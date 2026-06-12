"""Track B — learned diagram vectorization vs fixed representations.

PRE-REGISTERED SUCCESS CRITERION (fixed before any training run; see git
history of this file): the PersLay-style learned vectorization counts as
a positive result iff it beats BOTH the fixed persistence image AND the
hand-crafted scalar curves on TEST binary NLL by a margin exceeding the
3-seed std on >= 2 of the 3 horizons {3, 5, 10}. Otherwise the readout
is negative ("hand-crafted summaries suffice in this world") and is
reported as such.

All three variants share the same hazard head (MLP), the same local
features, the same optimizer/schedule/seeds — the per-frame topology
REPRESENTATION is the only variable:
  scalar   : 12 hand-crafted curve features (Phase 0 summaries + deltas)
  pi_fixed : 8x8x2 fixed persistence image (Adams-style, pers-weighted)
  perslay  : sum-pooled learned point transform (Carriere-style)

Example (PowerShell):
  python -m scripts.topo_track_b_perslay `
    --dataset .\\datasets\\topo_synth_v1 --catalog .\\datasets\\topo_synth_v1\\catalog `
    --out .\\runs_topo\\phase2_track_b
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
from crackle.metrics.point_process import binary_nll, topk_precision_recall
from crackle.topo.catalog import (
    GLOBAL_FEATURES,
    LOCAL_FEATURES,
    RiskSetConfig,
)

_W: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    _W.update(args_dict)


def _case_diagrams(case_id: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Per-frame ROI-filtered significant diagram points for one case."""
    from crackle.topo.cubical import superlevel_persistence
    from crackle.topo.roi import apply_roi, horizon_margin_mask

    config = RiskSetConfig(**_W["config"])
    data = np.load(Path(_W["dataset"]) / "shards" / f"{case_id}.npz")
    movie = data["movie_u8"].astype(np.float64) / 255.0
    ny, nx = movie.shape[1], movie.shape[2]
    roi = horizon_margin_mask(ny, nx, height=config.height,
                              horizon=config.horizon_mm,
                              k=config.roi_margin_k)
    pts_all, offsets = [], [0]
    for t in range(movie.shape[0]):
        diag = apply_roi(
            superlevel_persistence(movie[t], connectivity=config.connectivity),
            roi,
        ).significant(config.sig_tau)
        pts = np.stack([diag.dim.astype(np.float64), diag.birth,
                        diag.persistence], axis=1) if diag.dim.size else \
            np.zeros((0, 3))
        pts_all.append(pts.astype(np.float32))
        offsets.append(offsets[-1] + pts.shape[0])
    return case_id, np.concatenate(pts_all, axis=0), np.asarray(offsets,
                                                                dtype=np.int64)


def build_diagram_cache(dataset: Path, case_ids: list[str], workers: int,
                        config: dict, cache_path: Path) -> dict[str, tuple]:
    if cache_path.exists():
        data = np.load(cache_path)
        if all(f"{c}_pts" in data.files for c in case_ids):
            return {c: (data[f"{c}_pts"], data[f"{c}_off"]) for c in case_ids}
    out: dict[str, tuple] = {}
    payload: dict[str, np.ndarray] = {}
    with Pool(processes=workers, initializer=_init_worker,
              initargs=({"dataset": str(dataset), "config": config},)) as pool:
        for i, (cid, pts, off) in enumerate(
                pool.imap_unordered(_case_diagrams, case_ids), 1):
            out[cid] = (pts, off)
            payload[f"{cid}_pts"] = pts
            payload[f"{cid}_off"] = off
            if i % 250 == 0 or i == len(case_ids):
                print(f"diagrams [{i}/{len(case_ids)}]", flush=True)
    np.savez_compressed(cache_path, **payload)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--horizons", type=int, nargs="*", default=[3, 5, 10])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-groups", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    from crackle.topo.perslay import TrackBModel, persistence_image

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    summary = pd.read_csv(args.catalog / "case_summary.csv")
    case_ids = list(summary["case_id"])
    diagrams = build_diagram_cache(
        args.dataset, case_ids, args.workers,
        {"sig_tau": 0.08, "roi_margin_k": 1.5},
        args.out / "diagram_cache.npz")

    # ---- group the risk sets by (case, step) --------------------------
    print("loading risk sets ...", flush=True)
    splits: dict[str, dict[str, np.ndarray]] = {}
    n_tiles = None
    for split in ("train", "test", "ood"):
        df = pd.read_parquet(args.catalog / "risksets" /
                             f"riskset_{split}.parquet")
        df = df.sort_values(["case_id", "step", "tile_y", "tile_x"],
                            kind="stable")
        counts = df.groupby(["case_id", "step"], sort=False).size()
        n_tiles = int(counts.iloc[0])
        assert (counts == n_tiles).all(), "ragged tile groups"
        g = len(counts)
        local = df[LOCAL_FEATURES].to_numpy(np.float32).reshape(g, n_tiles, -1)
        scalar = df[GLOBAL_FEATURES].to_numpy(np.float32) \
            .reshape(g, n_tiles, -1)[:, 0, :]
        labels = {h: df[f"label_any_H{h}"].to_numpy(np.int8)
                  .reshape(g, n_tiles) for h in args.horizons}
        keys = counts.index.to_frame(index=False)
        splits[split] = {
            "case": keys["case_id"].to_numpy(), "step":
                keys["step"].to_numpy(np.int64),
            "local": local, "scalar": scalar, "labels": labels,
        }
        print(f"{split}: {g} groups x {n_tiles} tiles", flush=True)

    # ---- per-group topo inputs ----------------------------------------
    def group_pts(split: str, idx: np.ndarray, max_p: int
                  ) -> tuple[np.ndarray, np.ndarray]:
        s = splits[split]
        pts = np.zeros((idx.size, max_p, 3), dtype=np.float32)
        mask = np.zeros((idx.size, max_p), dtype=bool)
        for j, i in enumerate(idx):
            p, off = diagrams[s["case"][i]]
            a, b = off[s["step"][i]], off[s["step"][i] + 1]
            n = min(b - a, max_p)
            pts[j, :n] = p[a : a + n]
            mask[j, :n] = True
        return pts, mask

    print("precomputing fixed persistence images ...", flush=True)
    pi_cache: dict[str, np.ndarray] = {}
    for split, s in splits.items():
        g = s["case"].size
        pi = np.zeros((g, 128), dtype=np.float32)
        for i in range(g):
            p, off = diagrams[s["case"][i]]
            a, b = off[s["step"][i]], off[s["step"][i] + 1]
            pi[i] = persistence_image(p[a:b])
        pi_cache[split] = pi
        print(f"  {split} done", flush=True)

    max_pts = max(int(np.diff(off).max()) for _, off in diagrams.values())
    print(f"max diagram points/frame: {max_pts}", flush=True)

    # ---- training loop -------------------------------------------------
    def standardize(train_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = train_arr.mean(axis=0, keepdims=True)
        std = train_arr.std(axis=0, keepdims=True) + 1e-6
        return mean, std

    results = []
    started = time.perf_counter()
    loc_mean, loc_std = standardize(
        splits["train"]["local"].reshape(-1, len(LOCAL_FEATURES)))
    sc_mean, sc_std = standardize(splits["train"]["scalar"])
    pi_mean, pi_std = standardize(pi_cache["train"])

    for horizon in args.horizons:
        for variant in ("scalar", "pi_fixed", "perslay"):
            for seed in args.seeds:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = TrackBModel(variant,
                                    n_local=len(LOCAL_FEATURES)).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                s = splits["train"]
                lab = s["labels"][horizon]
                usable = np.flatnonzero((lab >= 0).any(axis=1))
                for epoch in range(args.epochs):
                    order = rng.permutation(usable)
                    for i in range(0, order.size, args.batch_groups):
                        idx = order[i : i + args.batch_groups]
                        local = torch.as_tensor(
                            (s["local"][idx] - loc_mean) / loc_std,
                            device=device)
                        y = torch.as_tensor(
                            lab[idx].astype(np.float32), device=device)
                        valid = y >= 0
                        kw = {}
                        if variant == "scalar":
                            topo = torch.as_tensor(
                                (s["scalar"][idx] - sc_mean) / sc_std,
                                device=device)
                        elif variant == "pi_fixed":
                            topo = torch.as_tensor(
                                (pi_cache["train"][idx] - pi_mean) / pi_std,
                                device=device)
                        else:
                            topo = torch.zeros((idx.size, 0), device=device)
                            pts, mask = group_pts("train", idx, max_pts)
                            kw = {"diag_pts": torch.as_tensor(pts,
                                                              device=device),
                                  "diag_mask": torch.as_tensor(mask,
                                                               device=device)}
                        logits = model(local, topo, **kw)
                        loss = nn_bce(logits, y, valid)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                # evaluation
                model.eval()
                for split in ("test", "ood"):
                    s_ev = splits[split]
                    lab_ev = s_ev["labels"][horizon]
                    probs, ys = [], []
                    with torch.no_grad():
                        for i in range(0, s_ev["case"].size, 2048):
                            idx = np.arange(i, min(i + 2048,
                                                   s_ev["case"].size))
                            local = torch.as_tensor(
                                (s_ev["local"][idx] - loc_mean) / loc_std,
                                device=device)
                            kw = {}
                            if variant == "scalar":
                                topo = torch.as_tensor(
                                    (s_ev["scalar"][idx] - sc_mean) / sc_std,
                                    device=device)
                            elif variant == "pi_fixed":
                                topo = torch.as_tensor(
                                    (pi_cache[split][idx] - pi_mean) / pi_std,
                                    device=device)
                            else:
                                topo = torch.zeros((idx.size, 0),
                                                   device=device)
                                pts, mask = group_pts(split, idx, max_pts)
                                kw = {"diag_pts": torch.as_tensor(
                                          pts, device=device),
                                      "diag_mask": torch.as_tensor(
                                          mask, device=device)}
                            p = torch.sigmoid(model(local, topo, **kw))
                            valid = lab_ev[idx] >= 0
                            probs.append(p.cpu().numpy()[valid])
                            ys.append(lab_ev[idx][valid])
                    prob = np.concatenate(probs)
                    y_all = np.concatenate(ys).astype(np.int8)
                    k = max(1, int(0.01 * y_all.size))
                    prec, rec = topk_precision_recall(prob, y_all, k)
                    results.append({
                        "variant": variant, "horizon": horizon, "seed": seed,
                        "split": split,
                        "nll": binary_nll(prob, y_all),
                        "top1pct_recall": rec, "top1pct_precision": prec,
                    })
                print(f"H{horizon} {variant} seed {seed} done "
                      f"({time.perf_counter()-started:.0f}s)", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(args.out / "track_b_results.csv", index=False)
    write_json(args.out / "config.json", {
        "horizons": args.horizons, "seeds": list(args.seeds),
        "epochs": args.epochs, "git_commit": _git_commit(Path.cwd()),
        "wall_s": round(time.perf_counter() - started, 1),
    })
    print(json.dumps({"out": str(args.out)}))
    return 0


def nn_bce(logits, y, valid):
    import torch

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y.clamp(min=0.0), reduction="none")
    return (loss * valid.float()).sum() / valid.float().sum().clamp(min=1.0)


if __name__ == "__main__":
    sys.exit(main())
