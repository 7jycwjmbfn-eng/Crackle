"""Track A — neural temporal point process vs parametric Hawkes (Phase 2.2).

PRE-REGISTERED SUCCESS CRITERION (fixed before any training run; see git
history of this file): the discrete-time transformer-Hawkes model counts
as a positive result iff it beats the parametric Hawkes referee on TEST
total log-likelihood per case AND on >= 2 of the 3 component
log-likelihoods (count, kind, tile), each by a margin exceeding the
3-seed std. Anything else is a negative readout and is reported as such.

Pipeline:
1. cache per-case global curves (recomputed from quantized shards, ROI on)
2. pack event streams from the Phase 1.3 catalog (splits from
   case_summary.csv: train/val/test + held-out 4-notch OOD)
3. referee: exponential-kernel Hawkes (grid MLE on train) + empirical
   train kind/tile distributions
4. train DiscreteTHP (3 seeds), select epoch by val total LL
5. report per-split LL components, kind/tile accuracy, count MAE, and
   time-rescaling KS (fraction of cases with p > 0.05)

Example (PowerShell):
  python -m scripts.topo_track_a_ntpp `
    --dataset .\\datasets\\topo_synth_v1 --catalog .\\datasets\\topo_synth_v1\\catalog `
    --out .\\runs_topo\\phase2_track_a
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from crackle.data.common import write_json
from crackle.experiments.hetero_pinning import _git_commit
from crackle.metrics.point_process import ks_exp_pvalue
from crackle.topo.catalog import EVENT_KINDS, GLOBAL_CURVE_KEYS, RiskSetConfig

_W: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    _W.update(args_dict)


def _case_curves(case_id: str) -> tuple[str, np.ndarray]:
    from crackle.topo.catalog import case_events_and_curves

    data = np.load(Path(_W["dataset"]) / "shards" / f"{case_id}.npz")
    movie = data["movie_u8"].astype(np.float64) / 255.0
    _, curves, _ = case_events_and_curves(
        movie, config=RiskSetConfig(**_W["config"]))
    return case_id, np.stack([curves[k] for k in GLOBAL_CURVE_KEYS], axis=1)


def build_curve_cache(dataset: Path, case_ids: list[str], workers: int,
                      config: dict, cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path)
        if set(data.files) >= set(case_ids):
            return {c: data[c] for c in case_ids}
    out: dict[str, np.ndarray] = {}
    with Pool(processes=workers, initializer=_init_worker,
              initargs=({"dataset": str(dataset), "config": config},)) as pool:
        for i, (cid, arr) in enumerate(
                pool.imap_unordered(_case_curves, case_ids), 1):
            out[cid] = arr.astype(np.float32)
            if i % 250 == 0 or i == len(case_ids):
                print(f"curves [{i}/{len(case_ids)}]", flush=True)
    np.savez_compressed(cache_path, **out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    from crackle.topo.ntpp import (
        CaseTensors,
        DiscreteTHP,
        NTPPConfig,
        case_log_likelihood,
        fit_parametric_hawkes,
        hawkes_log_likelihood,
        pack_case,
        rescaled_intervals,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    summary = pd.read_csv(args.catalog / "case_summary.csv")
    events = pd.read_parquet(args.catalog / "topo_events.parquet")
    splits = {s: list(g["case_id"]) for s, g in summary.groupby("split")}
    n_steps = 81  # generated movies are (steps+1) frames
    config = NTPPConfig(epochs=args.epochs)

    curves = build_curve_cache(
        args.dataset, list(summary["case_id"]), args.workers,
        {"sig_tau": 0.08, "roi_margin_k": 1.5},
        args.out / "curves_cache.npz")

    by_case = dict(tuple(events.groupby("case_id")))
    empty = pd.DataFrame({"step": [], "kind": [], "y": [], "x": []})
    packed: dict[str, CaseTensors] = {
        cid: pack_case(by_case.get(cid, empty), n_steps, curves[cid], config)
        for cid in summary["case_id"]
    }

    # ---------------- referee: parametric Hawkes + empirical marks ------
    train_cases = splits["train"]
    mu, alpha, beta = fit_parametric_hawkes(
        [packed[c].steps for c in train_cases],
        [n_steps] * len(train_cases))
    print(f"hawkes MLE: mu={mu:.4f} alpha={alpha:.4f} beta={beta:.4f}",
          flush=True)
    train_kinds = np.concatenate([packed[c].kinds for c in train_cases])
    train_tiles = np.concatenate([packed[c].tiles for c in train_cases])
    p_kind = np.bincount(train_kinds, minlength=len(EVENT_KINDS)) + 1.0
    p_kind /= p_kind.sum()
    p_tile = np.bincount(train_tiles, minlength=config.n_tiles) + 1.0
    p_tile /= p_tile.sum()

    referee_rows = []
    for split, cids in splits.items():
        comp = {"count": 0.0, "kind": 0.0, "tile": 0.0}
        n_ev, ks_ok, ks_n = 0, 0, 0
        for cid in cids:
            p = packed[cid]
            ll_c, lam = hawkes_log_likelihood(p.steps, n_steps, mu, alpha, beta)
            comp["count"] += ll_c
            comp["kind"] += float(np.log(p_kind[p.kinds]).sum())
            comp["tile"] += float(np.log(p_tile[p.tiles]).sum())
            n_ev += p.steps.size
            _, pval = ks_exp_pvalue(rescaled_intervals(p.steps, lam))
            if pval is not None:
                ks_n += 1
                ks_ok += int(pval > 0.05)
        n = len(cids)
        referee_rows.append({
            "model": "parametric_hawkes", "seed": -1, "split": split,
            "ll_count_per_case": comp["count"] / n,
            "ll_kind_per_case": comp["kind"] / n,
            "ll_tile_per_case": comp["tile"] / n,
            "ll_total_per_case": sum(comp.values()) / n,
            "ks_frac_ok": ks_ok / max(ks_n, 1), "n_events": n_ev,
        })
    # kind_acc for the referee = majority-class rate, computed properly:
    for row, (split, cids) in zip(referee_rows, splits.items()):
        kinds_all = np.concatenate([packed[c].kinds for c in cids]) \
            if cids else np.zeros(0, dtype=np.int64)
        tiles_all = np.concatenate([packed[c].tiles for c in cids]) \
            if cids else np.zeros(0, dtype=np.int64)
        row["kind_acc"] = float((kinds_all == p_kind.argmax()).mean()) \
            if kinds_all.size else 0.0
        row["tile_acc"] = float((tiles_all == p_tile.argmax()).mean()) \
            if tiles_all.size else 0.0

    # ---------------- neural model -------------------------------------
    def make_batch(cids: list[str]) -> dict[str, torch.Tensor]:
        bos_kind, bos_tile = len(EVENT_KINDS), config.n_tiles
        e_max = max(p.steps.size for p in (packed[c] for c in cids)) + 1
        b = len(cids)
        ev_kind = np.full((b, e_max), bos_kind, dtype=np.int64)
        ev_tile = np.full((b, e_max), bos_tile, dtype=np.int64)
        ev_step = np.full((b, e_max), config.max_steps, dtype=np.int64)
        ev_mask = np.zeros((b, e_max), dtype=bool)
        tgt_mask = np.zeros((b, e_max), dtype=bool)
        counts = np.zeros((b, n_steps), dtype=np.float32)
        curves_arr = np.zeros((b, n_steps, len(GLOBAL_CURVE_KEYS)),
                              dtype=np.float32)
        for i, cid in enumerate(cids):
            p = packed[cid]
            e = p.steps.size
            ev_kind[i, 0], ev_tile[i, 0], ev_step[i, 0] = bos_kind, bos_tile, 0
            ev_mask[i, 0] = True
            ev_kind[i, 1 : e + 1] = p.kinds
            ev_tile[i, 1 : e + 1] = p.tiles
            ev_step[i, 1 : e + 1] = p.steps
            ev_mask[i, 1 : e + 1] = True
            tgt_mask[i, 1 : e + 1] = True
            np.add.at(counts[i], p.steps[p.steps < n_steps], 1.0)
            curves_arr[i] = p.curves
        step_valid = np.ones((b, n_steps), dtype=np.float32)
        t = {k: torch.as_tensor(v, device=device) for k, v in {
            "ev_kind": ev_kind, "ev_tile": ev_tile, "ev_step": ev_step,
            "ev_mask": ev_mask, "tgt_mask": tgt_mask, "counts": counts,
            "curves": curves_arr, "step_valid": step_valid,
        }.items()}
        return t

    def eval_split(model, cids: list[str]) -> dict[str, float]:
        model.eval()
        comp = {"count": 0.0, "kind": 0.0, "tile": 0.0}
        kind_hits = tile_hits = n_ev = 0
        ks_ok = ks_n = 0
        with torch.no_grad():
            for i in range(0, len(cids), config.batch_cases):
                chunk = cids[i : i + config.batch_cases]
                batch = make_batch(chunk)
                ll = case_log_likelihood(model, batch)
                comp["count"] += float(ll["count"])
                comp["kind"] += float(ll["kind"])
                comp["tile"] += float(ll["tile"])
                n_ev += int(ll["n_events"])
                # mark accuracy + KS via per-case intensities
                enc = model.encode_history(batch["ev_kind"], batch["ev_tile"],
                                           batch["ev_step"], batch["ev_mask"])
                lam_steps = []
                for t_step in range(1, n_steps):
                    tt = torch.full((len(chunk),), t_step, device=device,
                                    dtype=torch.long)
                    context = model.query(enc, batch["ev_step"],
                                          batch["ev_mask"], tt,
                                          batch["curves"][:, t_step - 1])
                    lam, kl, tl = model.step_terms(context)
                    lam_steps.append(lam.cpu().numpy())
                    ev_here = (batch["ev_step"] == t_step) & batch["tgt_mask"]
                    if ev_here.any():
                        rows, cols = ev_here.nonzero(as_tuple=True)
                        kind_hits += int((kl.argmax(-1)[rows]
                                          == batch["ev_kind"][rows, cols]
                                          ).sum())
                        tile_hits += int((tl.argmax(-1)[rows]
                                          == batch["ev_tile"][rows, cols]
                                          ).sum())
                lam_arr = np.stack(lam_steps, axis=1)  # (B, T-1)
                for j, cid in enumerate(chunk):
                    p = packed[cid]
                    lam_full = np.concatenate([[0.0], lam_arr[j]])
                    _, pval = ks_exp_pvalue(
                        rescaled_intervals(p.steps, lam_full))
                    if pval is not None:
                        ks_n += 1
                        ks_ok += int(pval > 0.05)
        n = len(cids)
        return {
            "ll_count_per_case": comp["count"] / n,
            "ll_kind_per_case": comp["kind"] / n,
            "ll_tile_per_case": comp["tile"] / n,
            "ll_total_per_case": sum(comp.values()) / n,
            "kind_acc": kind_hits / max(n_ev, 1),
            "tile_acc": tile_hits / max(n_ev, 1),
            "ks_frac_ok": ks_ok / max(ks_n, 1),
            "n_events": n_ev,
        }

    rows = list(referee_rows)
    started = time.perf_counter()
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DiscreteTHP(config).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
        best_val, best_state = -np.inf, None
        order = list(train_cases)
        for epoch in range(config.epochs):
            model.train()
            rng = np.random.default_rng(seed * 1000 + epoch)
            rng.shuffle(order)
            total = 0.0
            for i in range(0, len(order), config.batch_cases):
                batch = make_batch(order[i : i + config.batch_cases])
                ll = case_log_likelihood(model, batch)
                loss = -(ll["count"] + ll["kind"] + ll["tile"]) \
                    / batch["counts"].shape[0]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += float(loss)
            val = eval_split(model, splits["val"])["ll_total_per_case"]
            print(f"seed {seed} epoch {epoch}: train loss {total:.1f} "
                  f"val LL/case {val:.2f}", flush=True)
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        for split in ("test", "ood", "val"):
            rows.append({"model": "discrete_thp", "seed": seed,
                         "split": split, **eval_split(model, splits[split])})
        print(f"seed {seed} done ({time.perf_counter()-started:.0f}s)",
              flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(args.out / "track_a_results.csv", index=False)
    write_json(args.out / "config.json", {
        "hawkes": {"mu": mu, "alpha": alpha, "beta": beta},
        "ntpp": config.__dict__, "seeds": list(args.seeds),
        "git_commit": _git_commit(Path.cwd()),
        "wall_s": round(time.perf_counter() - started, 1),
    })
    print(json.dumps({"out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
