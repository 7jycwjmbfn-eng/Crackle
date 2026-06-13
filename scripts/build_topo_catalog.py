"""Build topological event catalogs + tile risk sets from a generated dataset.

Input: a dataset directory produced by scripts.generate_topo_dataset
(manifest.csv + shards/case_*.npz with movie_u8). TDA is recomputed here
from the QUANTIZED movies — the single source of truth for downstream
phases — with ROI and matcher recorded in every row.

Output layout under --out:
  catalogs/case_<id>_events.parquet    per-case event catalog (csv fallback)
  topo_events.parquet                  global concatenated catalog
  risksets/riskset_<split>.parquet     train / val / test / ood rows
  config.json                          full provenance (args + git commit)
  catalog_audit.md                     counts per kind, class balance per horizon

Splits are BY CASE (md5 hash, 70/15/15); every case with --ood-notches
notches goes to the held-out "ood" split (addendum v1.1 A).

Example (PowerShell):
  python -m scripts.build_topo_catalog `
    --dataset .\datasets\topo_synth_v1 --out .\datasets\topo_synth_v1\catalog `
    --workers 16
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
import pandas as pd

from crackle.experiments.hetero_pinning import _git_commit
from crackle.topo.catalog import (
    CATALOG_COLUMNS,
    EVENT_KINDS,
    FEATURE_COLUMNS,
    RiskSetConfig,
    case_events_and_curves,
    catalog_rows,
    riskset_rows,
    split_of_case,
)

_W: dict[str, Any] = {}


def _init_worker(args_dict: dict[str, Any]) -> None:
    _W.update(args_dict)


def _one_case(case: dict[str, Any]) -> dict[str, Any]:
    config = RiskSetConfig(**_W["config"])
    case_id = case["case_id"]
    shard = Path(_W["dataset"]) / "shards" / f"{case_id}.npz"
    data = np.load(shard)
    clean = data["movie_u8"].astype(np.float64) / 255.0
    # ground-truth events (labels) always come from the clean field
    clean_events, clean_curves, roi = case_events_and_curves(clean, config=config)

    sigma = float(_W.get("noise_sigma", 0.0))
    if sigma > 0.0:
        from crackle.topo.noise import add_measurement_noise
        case_seed = int(case_id.split("_")[-1])
        movie = add_measurement_noise(
            clean, sigma=sigma, corr_cells=_W.get("noise_corr", 1.5),
            seed=_W.get("noise_seed", 0) + 1000 * case_seed)
        events, curves, _ = case_events_and_curves(movie, config=config)
        label_events = clean_events  # forecast TRUE failure from noisy obs
        source = f"movie_u8+noise{sigma}"
    else:
        movie, events, curves, label_events = (
            clean, clean_events, clean_curves, None)
        source = "movie_u8"

    cat = pd.DataFrame(
        catalog_rows(events, case_id=case_id, config=config,
                     source_field_key=source),
        columns=CATALOG_COLUMNS,
    )
    cat_dir = Path(_W["out"]) / "catalogs"
    cat_dir.mkdir(parents=True, exist_ok=True)
    cat_path = cat_dir / f"{case_id}_events.parquet"
    cat.to_parquet(cat_path, index=False)

    rs = pd.DataFrame(
        riskset_rows(movie, events, curves, roi, case_id=case_id,
                     config=config, label_events=label_events)
    )
    feat_cols = [c for c in rs.columns if c in FEATURE_COLUMNS]
    rs[feat_cols] = rs[feat_cols].astype(np.float32)
    rs_path = Path(_W["out"]) / "risksets" / "cases" / f"{case_id}.parquet"
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    rs.to_parquet(rs_path, index=False)

    label_cols = [c for c in rs.columns if c.startswith("label_")]
    label_stats = {
        c: {
            "pos": int((rs[c] == 1).sum()),
            "neg": int((rs[c] == 0).sum()),
            "censored": int((rs[c] == -1).sum()),
        }
        for c in label_cols
    }
    return {
        "case_id": case_id,
        "split": split_of_case(case_id, int(case["n_notches"]),
                               ood_notches=_W["ood_notches"]),
        "n_events": len(events),
        "kind_counts": {k: sum(1 for e in events if e.kind == k)
                        for k in EVENT_KINDS},
        "n_riskset_rows": len(rs),
        "label_stats": label_stats,
        "events_per_step": len(events) / max(movie.shape[0] - 1, 1),
        "catalog_path": str(cat_path),
        "riskset_path": str(rs_path),
    }


def _concat_parquets(paths: list[Path], out_path: Path) -> int:
    import pyarrow.parquet as pq

    writer = None
    total = 0
    for p in paths:
        table = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total += table.num_rows
    if writer is not None:
        writer.close()
    return total


def write_audit(out: Path, results: list[dict[str, Any]],
                args: argparse.Namespace, wall_s: float) -> None:
    kinds_total = {k: sum(r["kind_counts"][k] for r in results) for k in EVENT_KINDS}
    ev_rate = np.array([r["events_per_step"] for r in results])
    splits = sorted({r["split"] for r in results})
    lines = [
        "# Topo Phase 1.3 catalog audit",
        "",
        f"dataset: {args.dataset}  cases: {len(results)}  wall: {wall_s:.0f}s",
        f"config: sig_tau={args.sig_tau} roi_margin_k={args.roi_margin_k} "
        f"matcher={args.matcher} tile={args.tile} horizons={args.horizons} "
        f"decay={args.decay}",
        f"git: {_git_commit(Path.cwd())}",
        "",
        "## Event stream",
        "",
        f"events/step per case: mean {ev_rate.mean():.3f}, p10 "
        f"{np.percentile(ev_rate, 10):.3f}, p90 {np.percentile(ev_rate, 90):.3f}",
        f"KILL RULE 1 (>= 0.3 events/step mean): "
        f"{'PASS' if ev_rate.mean() >= 0.3 else 'FAIL'}",
        "",
        "| kind | events |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in kinds_total.items()]
    lines += ["", "## Splits (by case)", "", "| split | cases | riskset rows |",
              "|---|---|---|"]
    for s in splits:
        rs = [r for r in results if r["split"] == s]
        lines.append(f"| {s} | {len(rs)} | {sum(r['n_riskset_rows'] for r in rs)} |")
    lines += ["", "## Class balance (positives / non-censored, all cases)", "",
              "| label | pos | neg | censored | pos rate |", "|---|---|---|---|---|"]
    label_cols = sorted(results[0]["label_stats"])
    for c in label_cols:
        pos = sum(r["label_stats"][c]["pos"] for r in results)
        neg = sum(r["label_stats"][c]["neg"] for r in results)
        cen = sum(r["label_stats"][c]["censored"] for r in results)
        rate = pos / max(pos + neg, 1)
        lines.append(f"| {c} | {pos} | {neg} | {cen} | {rate:.4f} |")
    lines += [
        "",
        "Claim boundary: this is an event/feature CATALOG audit; it licenses",
        "Phase 2 model training, no forecasting claim. Features use frames <= t",
        "only (causality unit-tested); labels use (t, t+H].",
    ]
    (out / "catalog_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sig-tau", type=float, default=0.08)
    parser.add_argument("--roi-margin-k", type=float, default=1.5)
    parser.add_argument("--matcher", type=str, default="wasserstein",
                        choices=["greedy", "wasserstein"])
    parser.add_argument("--tile", type=int, default=6)
    parser.add_argument("--horizons", type=int, nargs="*", default=[3, 5, 10])
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--ood-notches", type=int, default=4)
    parser.add_argument("--noise-sigma", type=float, default=0.0,
                        help="measurement noise on features; labels stay "
                             "ground-truth (clean). 0 = clean build")
    parser.add_argument("--noise-corr", type=float, default=1.5)
    parser.add_argument("--noise-seed", type=int, default=20260612)
    args = parser.parse_args(argv)

    manifest = list(csv.DictReader((args.dataset / "manifest.csv").open()))
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(
        sig_tau=args.sig_tau, roi_margin_k=args.roi_margin_k,
        matcher=args.matcher, tile=args.tile,
        horizons=tuple(args.horizons), decay=args.decay,
    )
    (args.out / "config.json").write_text(json.dumps(
        {**config, "dataset": str(args.dataset),
         "ood_notches": args.ood_notches,
         "noise_sigma": args.noise_sigma, "noise_corr": args.noise_corr,
         "noise_seed": args.noise_seed,
         "git_commit": _git_commit(Path.cwd())}, indent=2), encoding="utf-8")

    started = time.perf_counter()
    worker_args = {"dataset": str(args.dataset), "out": str(args.out),
                   "config": config, "ood_notches": args.ood_notches,
                   "noise_sigma": args.noise_sigma,
                   "noise_corr": args.noise_corr,
                   "noise_seed": args.noise_seed}
    results: list[dict[str, Any]] = []
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(worker_args,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one_case, manifest), 1):
            results.append(r)
            if i % 100 == 0 or i == len(manifest):
                rate = i / (time.perf_counter() - started)
                print(f"[{i}/{len(manifest)}] {rate:.1f} case/s", flush=True)
    results.sort(key=lambda r: r["case_id"])

    n = _concat_parquets([Path(r["catalog_path"]) for r in results],
                         args.out / "topo_events.parquet")
    print(f"global catalog: {n} events")
    for split in sorted({r["split"] for r in results}):
        paths = [Path(r["riskset_path"]) for r in results if r["split"] == split]
        n = _concat_parquets(paths, args.out / "risksets" /
                             f"riskset_{split}.parquet")
        print(f"riskset_{split}: {n} rows from {len(paths)} cases")

    wall = time.perf_counter() - started
    write_audit(args.out, results, args, wall)
    summary = pd.DataFrame([
        {"case_id": r["case_id"], "split": r["split"], "n_events": r["n_events"],
         "events_per_step": round(r["events_per_step"], 4)} for r in results
    ])
    summary.to_csv(args.out / "case_summary.csv", index=False)
    print(json.dumps({"cases": len(results), "wall_s": round(wall, 1),
                      "audit": str(args.out / "catalog_audit.md")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
