"""Phase 2.1 — causal onset detection sweep on a generated dataset.

For every case: recompute the topological precursor curves from the
quantized shard, run the causal detectors (rolling z, CUSUM) over a
threshold sweep on each signal, and evaluate lead vs false alarm against
the control signal total_damage.

Signals (spec 2.1): n_h0_sig, entropy_h0, total_pers_h0, cum_events.
Control: total_damage (same detectors, same sweep — the bar to beat).

Outputs under --out:
  sweep.csv          one row per (signal, detector, threshold): FA rate,
                     alarm rate, median/p25/p75 lead
  per_case.parquet   one row per (case, signal, detector, threshold)
  lead_vs_fa.png     curves per detector
  report fragment    printed to stdout as JSON; full report assembled by
                     reports/topo_phase2_causal_onset_<date>.md
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

from crackle.topo.catalog import RiskSetConfig, case_events_and_curves
from crackle.topo.causal_onset import evaluate_case
from crackle.topo.events import event_count_curves

Z_THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
CUSUM_THRESHOLDS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 30.0)
TOPO_SIGNALS = ("n_h0_sig", "entropy_h0", "total_pers_h0", "cum_events")
CONTROL = "total_damage"

_W: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    _W.update(args_dict)


def _one_case(case_id: str) -> list[dict[str, Any]]:
    config = RiskSetConfig(**_W["config"])
    data = np.load(Path(_W["dataset"]) / "shards" / f"{case_id}.npz")
    movie = data["movie_u8"].astype(np.float64) / 255.0
    events, curves, _ = case_events_and_curves(movie, config=config)
    counts = event_count_curves(events, movie.shape[0])
    signals: dict[str, np.ndarray] = {
        "n_h0_sig": curves["n_h0_sig"],
        "entropy_h0": curves["entropy_h0"],
        "total_pers_h0": curves["total_pers_h0"],
        "cum_events": np.cumsum(counts["all"]).astype(np.float64),
        CONTROL: movie.reshape(movie.shape[0], -1).sum(axis=1),
    }
    total_damage = signals[CONTROL]
    rows = []
    # cusum_k2: drift above the steady-ramp z baseline (~1.9 for a linear
    # ramp at window=10), so the accumulator only grows under acceleration
    for name, sig in signals.items():
        for detector, thresholds, kw in (
            ("z", Z_THRESHOLDS, {}),
            ("cusum", CUSUM_THRESHOLDS, {"cusum_k": 0.5}),
            ("cusum_k2", CUSUM_THRESHOLDS, {"cusum_k": 2.0}),
        ):
            for thr in thresholds:
                ev = evaluate_case(case_id, name, sig, total_damage,
                                   detector=detector.split("_")[0],
                                   threshold=thr, window=_W["window"], **kw)
                rows.append({
                    "case_id": case_id, "signal": name, "detector": detector,
                    "threshold": thr,
                    "first_alarm": -1 if ev.first_alarm is None else ev.first_alarm,
                    "t_star": ev.t_star, "growth_start": ev.growth_start,
                    "false_alarm": bool(ev.is_false_alarm),
                    "lead": np.nan if ev.lead is None else float(ev.lead),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = all cases")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--sig-tau", type=float, default=0.08)
    parser.add_argument("--roi-margin-k", type=float, default=1.5)
    parser.add_argument("--fa-targets", type=float, nargs="*",
                        default=[0.05, 0.10, 0.20])
    args = parser.parse_args(argv)

    manifest = pd.read_csv(args.dataset / "manifest.csv")
    case_ids = list(manifest["case_id"])
    if args.limit:
        case_ids = case_ids[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    config = dict(sig_tau=args.sig_tau, roi_margin_k=args.roi_margin_k)
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=({"dataset": str(args.dataset), "config": config,
                         "window": args.window},)) as pool:
        for i, rows in enumerate(pool.imap_unordered(_one_case, case_ids), 1):
            all_rows.extend(rows)
            if i % 200 == 0 or i == len(case_ids):
                print(f"[{i}/{len(case_ids)}]", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out / "per_case.parquet", index=False)

    sweep = (
        df.groupby(["signal", "detector", "threshold"])
        .agg(fa_rate=("false_alarm", "mean"),
             alarm_rate=("first_alarm", lambda s: float((s >= 0).mean())),
             median_lead=("lead", "median"),
             p25_lead=("lead", lambda s: s.quantile(0.25)),
             p75_lead=("lead", lambda s: s.quantile(0.75)),
             n_valid_lead=("lead", "count"))
        .reset_index()
    )
    sweep.to_csv(args.out / "sweep.csv", index=False)

    # Same-setting duel: z-scores and standardized CUSUM are scale-free, so
    # "equal false-alarm rate" is realized by comparing each topo signal to
    # the control at the IDENTICAL (detector, threshold) setting and
    # reporting both FA rates alongside. Low z thresholds (< ~2) sit below
    # the steady-ramp baseline of a linearly growing signal and detect
    # loading itself (trivial alarms); the informative regime starts where
    # detectors only respond to acceleration.
    matched = []
    for det in sorted(df["detector"].unique()):
        thresholds = sorted(df[df.detector == det]["threshold"].unique())
        for thr in thresholds:
            ctrl = df[(df.signal == CONTROL) & (df.detector == det)
                      & (df.threshold == thr)].set_index("case_id")
            for sig in TOPO_SIGNALS:
                topo = df[(df.signal == sig) & (df.detector == det)
                          & (df.threshold == thr)].set_index("case_id")
                joined = topo.join(ctrl, lsuffix="_topo", rsuffix="_ctrl")
                t_lead, c_lead = joined["lead_topo"], joined["lead_ctrl"]
                wins = ((t_lead > c_lead) & t_lead.notna() & c_lead.notna()
                        | (t_lead.notna() & c_lead.isna()))
                losses = ((t_lead < c_lead) & t_lead.notna() & c_lead.notna()
                          | (c_lead.notna() & t_lead.isna()))
                decided = int(wins.sum() + losses.sum())
                matched.append({
                    "detector": det, "threshold": thr, "signal": sig,
                    "fa_topo": float(topo["false_alarm"].mean()),
                    "fa_ctrl": float(ctrl["false_alarm"].mean()),
                    "alarm_rate_topo": float((topo["first_alarm"] >= 0).mean()),
                    "alarm_rate_ctrl": float((ctrl["first_alarm"] >= 0).mean()),
                    "median_lead_topo": float(t_lead.median()),
                    "median_lead_ctrl": float(c_lead.median()),
                    "win_frac": float(wins.sum() / decided) if decided else np.nan,
                    "n_decided": decided,
                })
    matched_df = pd.DataFrame(matched)
    matched_df.to_csv(args.out / "duel_per_threshold.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = sorted(df["detector"].unique())
    fig, axes = plt.subplots(1, len(detectors), figsize=(6 * len(detectors), 4.5),
                             sharey=True)
    for ax, det in zip(np.atleast_1d(axes), detectors):
        for sig in (*TOPO_SIGNALS, CONTROL):
            sub = sweep[(sweep.signal == sig) & (sweep.detector == det)]
            sub = sub.sort_values("fa_rate")
            style = dict(lw=2.5, color="k") if sig == CONTROL else {}
            ax.plot(sub.fa_rate, sub.median_lead, "o-", label=sig, **style)
        ax.set_xlabel("false-alarm rate"); ax.set_title(f"detector: {det}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("median lead (steps)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "lead_vs_fa.png", dpi=130)

    print(json.dumps({"cases": len(case_ids),
                      "wall_s": round(time.perf_counter() - started, 1),
                      "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
