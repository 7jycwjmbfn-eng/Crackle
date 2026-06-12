"""Generate a bond-level dataset for Track C (bond-graph GNN).

Same multi-notch case distribution as generate_topo_dataset, but the
saved payload is the simulator's native representation: the bond graph
with per-step alive states (bit-packed) and stretch/critical ratios at a
strided subset of steps. Movies are NOT saved here — Track C deliberately
sidesteps grid resampling (addendum v1.1 B, Track C).

Per-case npz under --out/shards/:
  points (N,2) f32, bonds (B,2) i32, rest f16, critical f16, gc_bond f16,
  alive_packed (T+1, ceil(B/8)) u8 (np.packbits along bonds), n_bonds,
  sample_steps (S,) i16, ratio (S, B) f16  [stretch_t / critical]
  meta json

Example (PowerShell):
  python -m scripts.generate_bond_dataset `
    --out .\\datasets\\topo_bonds_v1 --n-cases 400 --workers 8
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

from crackle.topo.synth import bond_stretch_at, sample_case, simulate_multinotch

_W: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    _W.update(args_dict)


def _one_case(case_index: int) -> dict[str, Any]:
    a = _W
    rng = np.random.default_rng(a["seed"] + case_index)
    case = sample_case(rng, case_seed=a["seed"] + case_index,
                       nx=a["nx"], ny=a["ny"], steps=a["steps"])
    out = simulate_multinotch(case, return_bonds=True)
    case_id = f"case_{case_index:06d}"
    steps = np.arange(a["stride"], case.steps - 1, a["stride"], dtype=np.int16)
    critical = out["critical"]
    ratio = np.stack([
        bond_stretch_at(case, out["points"], out["bonds"], out["rest"], int(t))
        / np.maximum(critical, 1e-9)
        for t in steps
    ]).astype(np.float16)
    shard = Path(a["out"]) / "shards" / f"{case_id}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard,
        points=out["points"].astype(np.float32),
        bonds=out["bonds"].astype(np.int32),
        rest=out["rest"].astype(np.float16),
        critical=critical.astype(np.float16),
        gc_bond=out["gc_bond"].astype(np.float16),
        alive_packed=np.packbits(out["alive_steps"], axis=1),
        n_bonds=np.int64(out["bonds"].shape[0]),
        sample_steps=steps,
        ratio=ratio,
        meta=json.dumps({**case.meta, "contrast": case.contrast,
                         "corr_length_mm": case.corr_length_mm,
                         "steps": case.steps, "nx": case.nx, "ny": case.ny,
                         "length": case.length, "height": case.height,
                         "horizon": case.horizon}),
    )
    return {"case_id": case_id, "n_notches": case.meta["n_notches"],
            "contrast": round(case.contrast, 4),
            "n_bonds": int(out["bonds"].shape[0]),
            "final_broken_frac": round(float(out["final_broken_frac"]), 5)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=29)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rows = []
    worker_args = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(worker_args,)) as pool:
        for i, row in enumerate(pool.imap_unordered(_one_case,
                                                    range(args.n_cases)), 1):
            rows.append(row)
            if i % 100 == 0 or i == args.n_cases:
                print(f"[{i}/{args.n_cases}] "
                      f"{i/(time.perf_counter()-started):.1f} case/s",
                      flush=True)
    rows.sort(key=lambda r: r["case_id"])
    with (args.out / "manifest.csv").open("w", newline="",
                                          encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "dataset_card.md").write_text("\n".join([
        "# topo_bonds dataset card (Track C)",
        "",
        f"cases: {len(rows)}  grid: {args.ny}x{args.nx}  steps: {args.steps}"
        f"  stride: {args.stride}  seed: {args.seed}",
        "",
        "Native bond-graph payload (no movies). Scope note (do not "
        "delete): KINEMATIC multi-notch proxy, valid for representation "
        "comparisons, invalid for quantitative mechanics claims.",
    ]), encoding="utf-8")
    print(json.dumps({"cases": len(rows),
                      "wall_s": round(time.perf_counter() - started, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
