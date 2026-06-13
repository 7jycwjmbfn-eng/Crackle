"""Phase 2.1 robustness — causal onset under DIC-like measurement noise.

This reactivates the false-alarm axis that degenerated in the noiseless
world (runs_topo/phase2_onset_full: FA ~ 0 everywhere). For each case and
each noise level sigma, measurement noise is added to the damage movie,
the ROI-filtered significant precursor signals are recomputed FROM THE
NOISY field, and the causal detectors (rolling z, CUSUM) are swept. The
failure-time reference (t*, growth_start) is taken from the CLEAN
ground-truth simulation, because in deployment you score a noisy detector
against the true failure time. The control total_damage is recomputed
from the SAME noisy field, so neither side is privileged.

Event / false-alarm definition (CORRECTED after the first smoke run --
see git history). The kinematic world loads monotonically, so damage
grows from step ~2 and the original "alarm before growth_start" false-
alarm definition was degenerate (no quiet pre-growth window; FA stuck at
0 with or without noise). The meaningful event is the INSTABILITY t*
(argmax damage increment). We therefore score each detector's FIRST alarm
at step t_a against a warning window W:
  - premature (t* - t_a > W)  -> FALSE ALARM (fired during routine
        sub-critical loading, not tracking the impending instability);
        its apparent "lead" is NOT credited
  - useful   (0 <= t* - t_a <= W) -> detection, lead = t* - t_a
  - late     (t_a > t*)        -> miss
This penalizes early noise-firing as a false alarm rather than rewarding
it as a long lead, so it cannot be gamed by lowering the threshold under
noise. W is swept over {15, 25, 40}.

PRE-REGISTERED QUESTION (the scientific question is unchanged from the
committed v1; only the degenerate FA reference was fixed): under
measurement noise that lifts the control's false-alarm rate off zero, at
MATCHED false-alarm rate, does at least one topological precursor signal
achieve a strictly longer median useful lead than total_damage on >= 2 of
the 3 noise levels {0.02, 0.05, 0.10}? A "no" is a real, reportable
negative.

Comparison method: rolling-z and standardized CUSUM are scale-free, so a
"matched false-alarm rate" is realized by reading each signal's lead at
the detector threshold whose measured FA rate is closest to a shared
target FA from below, then comparing median useful leads. Both topo and
control use the identical detector, warning window, and FA target.

Outputs under --out:
  per_case.parquet     (case, sigma, signal, detector, threshold) rows
  fa_sweep.csv         FA rate / median lead per (sigma, signal, detector, threshold)
  matched_fa.csv       lead at matched FA target, topo vs control
  lead_vs_fa_noise.png  curves per sigma
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
from crackle.topo.catalog import RiskSetConfig, case_events_and_curves
from crackle.topo.causal_onset import evaluate_case
from crackle.topo.events import event_count_curves
from crackle.topo.instability import instability_step
from crackle.topo.noise import add_measurement_noise

Z_THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)
CUSUM_THRESHOLDS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 30.0)
TOPO_SIGNALS = ("n_h0_sig", "entropy_h0", "total_pers_h0", "cum_events")
CONTROL = "total_damage"

_W: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    _W.update(args_dict)


def _signals_from_movie(movie: np.ndarray, config: RiskSetConfig
                        ) -> dict[str, np.ndarray]:
    events, curves, _ = case_events_and_curves(movie, config=config)
    counts = event_count_curves(events, movie.shape[0])
    return {
        "n_h0_sig": curves["n_h0_sig"],
        "entropy_h0": curves["entropy_h0"],
        "total_pers_h0": curves["total_pers_h0"],
        "cum_events": np.cumsum(counts["all"]).astype(np.float64),
        CONTROL: movie.reshape(movie.shape[0], -1).sum(axis=1),
    }


def _one_case(case_id: str) -> list[dict[str, Any]]:
    config = RiskSetConfig(**_W["config"])
    data = np.load(Path(_W["dataset"]) / "shards" / f"{case_id}.npz")
    clean = data["movie_u8"].astype(np.float64) / 255.0
    # ground-truth failure time from the CLEAN field
    clean_damage = clean.reshape(clean.shape[0], -1).sum(axis=1)
    t_star = instability_step(clean_damage)
    case_seed = int(case_id.split("_")[-1])
    rows: list[dict[str, Any]] = []
    for s_i, sigma in enumerate(_W["sigmas"]):
        movie = add_measurement_noise(
            clean, sigma=sigma, corr_cells=_W["corr_cells"],
            seed=_W["seed"] + 1000 * case_seed + s_i)
        signals = _signals_from_movie(movie, config)
        for name, sig in signals.items():
            for detector, thresholds in (("z", Z_THRESHOLDS),
                                         ("cusum", CUSUM_THRESHOLDS)):
                for thr in thresholds:
                    ev = evaluate_case(case_id, name, sig, clean_damage,
                                       detector=detector, threshold=thr,
                                       window=_W["window"])
                    rows.append({
                        "case_id": case_id, "sigma": sigma, "signal": name,
                        "detector": detector, "threshold": thr,
                        "first_alarm": -1 if ev.first_alarm is None
                        else int(ev.first_alarm),
                        "t_star": int(t_star),
                    })
    return rows


def _classify(df: pd.DataFrame, warn_window: int) -> pd.DataFrame:
    """Add false_alarm / detected / lead columns for a warning window W."""
    out = df.copy()
    alarmed = out["first_alarm"] >= 0
    gap = out["t_star"] - out["first_alarm"]
    out["false_alarm"] = alarmed & (gap > warn_window)      # premature
    out["detected"] = alarmed & (gap >= 0) & (gap <= warn_window)
    out["lead"] = np.where(out["detected"], gap.astype(float), np.nan)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--sigmas", type=float, nargs="*",
                        default=[0.02, 0.05, 0.10])
    parser.add_argument("--corr-cells", type=float, default=1.5)
    parser.add_argument("--sig-tau", type=float, default=0.08)
    parser.add_argument("--roi-margin-k", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--fa-targets", type=float, nargs="*",
                        default=[0.05, 0.10, 0.20])
    parser.add_argument("--warn-windows", type=int, nargs="*",
                        default=[15, 25, 40])
    parser.add_argument("--only-hetero", action="store_true",
                        help="drop contrast<=1.5 cases (trivial topology)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.dataset / "manifest.csv")
    if args.only_hetero and "contrast" in manifest.columns:
        manifest = manifest[manifest["contrast"] > 1.5]
    case_ids = list(manifest["case_id"])
    if args.limit:
        case_ids = case_ids[: args.limit]
    print(f"{len(case_ids)} cases, sigmas {args.sigmas}", flush=True)

    worker = {"dataset": str(args.dataset),
              "config": {"sig_tau": args.sig_tau,
                         "roi_margin_k": args.roi_margin_k},
              "sigmas": args.sigmas, "corr_cells": args.corr_cells,
              "window": args.window, "seed": args.seed}
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(worker,)) as pool:
        for i, rows in enumerate(pool.imap_unordered(_one_case, case_ids), 1):
            all_rows.extend(rows)
            if i % 100 == 0 or i == len(case_ids):
                print(f"[{i}/{len(case_ids)}] "
                      f"{i/(time.perf_counter()-started):.1f} case/s",
                      flush=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out / "per_case.parquet", index=False)

    sweeps, matched = [], []
    for W in args.warn_windows:
        cls = _classify(df, W)
        sweep = (cls.groupby(["sigma", "signal", "detector", "threshold"])
                 .agg(fa_rate=("false_alarm", "mean"),
                      detect_rate=("detected", "mean"),
                      median_lead=("lead", "median"),
                      n_detected=("detected", "sum"))
                 .reset_index())
        sweep["warn_window"] = W
        sweeps.append(sweep)

        def pick(sigma, sig, det, target):
            sub = sweep[(sweep.sigma == sigma) & (sweep.signal == sig)
                        & (sweep.detector == det)]
            below = sub[sub.fa_rate <= target]
            cand = below if len(below) else sub
            if not len(cand):
                return None
            ref = target if len(below) else cand.fa_rate.min()
            return cand.loc[(cand.fa_rate - ref).abs().idxmin()]

        for sigma in args.sigmas:
            for det in ("z", "cusum"):
                for target in args.fa_targets:
                    ctrl = pick(sigma, CONTROL, det, target)
                    if ctrl is None:
                        continue
                    cr = cls[(cls.sigma == sigma) & (cls.signal == CONTROL)
                             & (cls.detector == det)
                             & (cls.threshold == ctrl.threshold)
                             ].set_index("case_id")
                    for sig in TOPO_SIGNALS:
                        p = pick(sigma, sig, det, target)
                        if p is None:
                            continue
                        tr = cls[(cls.sigma == sigma) & (cls.signal == sig)
                                 & (cls.detector == det)
                                 & (cls.threshold == p.threshold)
                                 ].set_index("case_id")
                        j = tr.join(cr, lsuffix="_t", rsuffix="_c")
                        tl, cl_ = j["lead_t"], j["lead_c"]
                        wins = ((tl > cl_) & tl.notna() & cl_.notna()
                                | (tl.notna() & cl_.isna()))
                        losses = ((tl < cl_) & tl.notna() & cl_.notna()
                                  | (cl_.notna() & tl.isna()))
                        decided = int(wins.sum() + losses.sum())
                        matched.append({
                            "warn_window": W, "sigma": sigma, "detector": det,
                            "fa_target": target, "signal": sig,
                            "fa_topo": float(p.fa_rate),
                            "fa_ctrl": float(ctrl.fa_rate),
                            "detect_topo": float(p.detect_rate),
                            "detect_ctrl": float(ctrl.detect_rate),
                            "median_lead_topo": float(p.median_lead),
                            "median_lead_ctrl": float(ctrl.median_lead),
                            "win_frac": float(wins.sum() / decided)
                            if decided else np.nan, "n_decided": decided})
    pd.concat(sweeps, ignore_index=True).to_csv(
        args.out / "fa_sweep.csv", index=False)
    matched_df = pd.DataFrame(matched)
    matched_df.to_csv(args.out / "matched_fa.csv", index=False)
    sweep = sweeps[len(args.warn_windows) // 2]  # middle W for the plot

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = args.sigmas
    fig, axes = plt.subplots(1, len(sigmas), figsize=(6 * len(sigmas), 4.5),
                             sharey=True)
    for ax, sigma in zip(np.atleast_1d(axes), sigmas):
        for sig in (*TOPO_SIGNALS, CONTROL):
            sub = sweep[(sweep.sigma == sigma) & (sweep.signal == sig)
                        & (sweep.detector == "z")].sort_values("fa_rate")
            style = dict(lw=2.5, color="k") if sig == CONTROL else {}
            ax.plot(sub.fa_rate, sub.median_lead, "o-", label=sig, **style)
        ax.set_xlabel("false-alarm rate"); ax.set_title(f"sigma={sigma} (z)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("median lead (steps)")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out / "lead_vs_fa_noise.png", dpi=130)

    write_json(args.out / "config.json", {
        "dataset": str(args.dataset), "sigmas": args.sigmas,
        "corr_cells": args.corr_cells, "window": args.window,
        "only_hetero": args.only_hetero, "git_commit": _git_commit(Path.cwd()),
        "wall_s": round(time.perf_counter() - started, 1)})
    print(json.dumps({"cases": len(case_ids), "out": str(args.out),
                      "wall_s": round(time.perf_counter() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
