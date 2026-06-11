"""Mass-generate a multi-notch synthetic damage-movie dataset (parallel).

Each case: random notch set (count/position/angle/length), random toughness
contrast and correlation length, kinematic rollout -> damage movie
(T+1, ny, nx), stored uint8-quantized (damage in [0,1], 1/255 resolution is
far below sig_tau=0.08). Optionally runs the TDA pipeline per case and
emits the topological event catalog inline.

Output layout under --out:
  manifest.csv                       one row per case (geometry, physics, stats)
  shards/case_<id>.npz               movie_u8, gc_field (float16), meta json
  catalogs/case_<id>_events.csv      (if --with-tda)
  dataset_card.md                    generation parameters + honest scope note

Throughput reference (measured): ~0.7 s/case at 48x29x80 incl. TDA, single
core. Scale with --workers; 1000 cases on 12-16 cores lands in minutes.
Larger grids scale roughly linearly in nodes*steps.

Example (PowerShell):
  python -m scripts.generate_topo_dataset `
    --out .\\datasets\\topo_synth_v1 --n-cases 2000 --workers 12 `
    --nx 64 --ny 39 --steps 100 --with-tda
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from crackle.topo import extract_events, sequence_summaries
from crackle.topo.synth import sample_case, simulate_multinotch

_WORKER_ARGS: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    _WORKER_ARGS.update(args_dict)


def _one_case(case_index: int) -> dict[str, Any]:
    a = _WORKER_ARGS
    rng = np.random.default_rng(a["seed"] + case_index)
    case = sample_case(
        rng,
        case_seed=a["seed"] + case_index,
        nx=a["nx"], ny=a["ny"], steps=a["steps"],
        contrast_range=tuple(a["contrast_range"]),
        n_notch_range=tuple(a["n_notch_range"]),
        edge_notch_prob=a["edge_notch_prob"],
    )
    out = simulate_multinotch(case)
    movie = out["movie"]
    case_id = f"case_{case_index:06d}"
    shard = Path(a["out"]) / "shards" / f"{case_id}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard,
        movie_u8=np.clip(np.rint(movie * 255.0), 0, 255).astype(np.uint8),
        gc_field=out["gc_field"].astype(np.float16),
        meta=json.dumps({**case.meta, "contrast": case.contrast,
                         "corr_length_mm": case.corr_length_mm,
                         "steps": case.steps, "nx": case.nx, "ny": case.ny}),
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "n_notches": case.meta["n_notches"],
        "include_edge": case.meta["include_edge"],
        "contrast": round(case.contrast, 4),
        "corr_length_mm": case.corr_length_mm,
        "final_broken_frac": round(float(out["final_broken_frac"]), 5),
        "n_bonds": int(out["n_bonds"]),
    }
    if a["with_tda"]:
        _, diags = sequence_summaries(movie, sig_tau=a["sig_tau"])
        events = extract_events(diags, sig_tau=a["sig_tau"])
        cat = Path(a["out"]) / "catalogs" / f"{case_id}_events.csv"
        cat.parent.mkdir(parents=True, exist_ok=True)
        with cat.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["step", "kind", "y", "x", "persistence", "birth_value"]
            )
            writer.writeheader()
            writer.writerows([e.as_row() for e in events])
        row["n_events"] = len(events)
        row["n_h1_events"] = sum(1 for e in events if e.kind.startswith("h1"))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=29)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--contrast-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--n-notch-range", type=int, nargs=2, default=[1, 4])
    parser.add_argument("--edge-notch-prob", type=float, default=0.5)
    parser.add_argument("--with-tda", action="store_true")
    parser.add_argument("--sig-tau", type=float, default=0.08)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    worker_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(worker_args,)) as pool:
        for index, row in enumerate(pool.imap_unordered(_one_case, range(args.n_cases)), 1):
            rows.append(row)
            if index % 25 == 0 or index == args.n_cases:
                rate = index / (time.perf_counter() - started)
                print(f"[{index}/{args.n_cases}] {rate:.1f} case/s", flush=True)
    rows.sort(key=lambda r: r["case_id"])

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    card = [
        "# topo_synth dataset card",
        "",
        f"cases: {len(rows)}  grid: {args.ny}x{args.nx}  steps: {args.steps}",
        f"contrast: U{tuple(args.contrast_range)}  notches: {tuple(args.n_notch_range)}"
        f"  edge_notch_prob: {args.edge_notch_prob}  seed: {args.seed}",
        "",
        "Storage: movies uint8-quantized (1/255 << sig_tau); gc float16.",
        "",
        "Scope note (do not delete): generated by a KINEMATIC multi-notch proxy "
        "(prescribed displacement field + bond breaking), not a solved mechanics "
        "model. Valid as a topology/event-mining world; invalid for quantitative "
        "mechanics claims.",
    ]
    if args.with_tda:
        ev = np.array([r.get("n_events", 0) for r in rows], dtype=float)
        h1 = np.array([r.get("n_h1_events", 0) for r in rows], dtype=float)
        card += [
            "",
            f"TDA (sig_tau={args.sig_tau}): events/case mean {ev.mean():.1f} "
            f"(p10 {np.percentile(ev,10):.0f}, p90 {np.percentile(ev,90):.0f}); "
            f"H1 events/case mean {h1.mean():.1f}.",
        ]
    (args.out / "dataset_card.md").write_text("\n".join(card), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "manifest": str(manifest),
                      "wall_s": round(time.perf_counter() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
