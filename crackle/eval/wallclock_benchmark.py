from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from crackle.baselines.deterministic import DETERMINISTIC_MODELS, deterministic_logits
from crackle.data.common import load_json, load_labels, ok_samples, safe_div, write_csv, write_json
from crackle.data.features import feature_matrix, frontier_bond_mask, material_toughness, update_history_trigger
from crackle.eval.eval_crackle import _load_model, _predict_proba
from crackle.metrics.point_process import topk_precision_recall


def _source_manifest(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    meta = load_json(data_dir / "metadata.json")
    return Path(meta["source_manifest"]), meta


def causal_candidate_bonds(
    labels: dict[str, np.ndarray],
    step: int,
    *,
    active_topk: int,
    bonds: np.ndarray | None = None,
    toughness: np.ndarray | None = None,
) -> np.ndarray:
    bonds = labels["bonds"].astype(np.int64) if bonds is None else bonds
    alive = labels["bond_alive"][step].astype(bool)
    alive_idx = np.flatnonzero(alive)
    if alive_idx.size <= active_topk:
        return alive_idx
    toughness = material_toughness(labels, bonds) if toughness is None else toughness
    stretch_ratio = labels["bond_stretch"][step, alive_idx] / np.maximum(toughness[alive_idx], 1e-12)
    frontier = frontier_bond_mask(labels["crack_tip_mask"][step], bonds)[alive_idx].astype(np.float64)
    score = stretch_ratio + 0.35 * frontier
    chosen = np.argpartition(score, -int(active_topk))[-int(active_topk) :]
    return alive_idx[chosen]


def benchmark_model(data_dir: Path, payload: dict[str, Any], *, split: str, active_topk: int, top_k: int) -> dict[str, Any]:
    manifest_path, meta = _source_manifest(data_dir)
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    if not samples:
        raise ValueError(f"no samples found for split {split!r}")
    model = str(payload["model"])
    total_ms = 0.0
    riskset_build_ms = 0.0
    mechanics_feature_update_ms = 0.0
    intensity_eval_ms = 0.0
    active_selection_ms = 0.0
    pd_correction_ms = 0.0
    total_steps = 0
    total_true = 0
    captured_true = 0
    top_recalls: list[float] = []
    top_precisions: list[float] = []
    dense_ms: list[float] = []
    bond_counts: list[int] = []
    for sample in samples:
        labels = load_labels(sample)
        sample_started = time.perf_counter()
        bonds = labels["bonds"].astype(np.int64)
        bond_counts.append(int(bonds.shape[0]))
        toughness = material_toughness(labels, bonds)
        alive = labels["bond_alive"].astype(bool)
        prep_ms = 1000.0 * (time.perf_counter() - sample_started)
        mechanics_feature_update_ms += prep_ms
        total_ms += prep_ms
        history = np.zeros((alive.shape[1],), dtype=np.float64)
        for step in range(alive.shape[0] - 1):
            started = time.perf_counter()
            candidates = causal_candidate_bonds(labels, step, active_topk=active_topk, bonds=bonds, toughness=toughness)
            selected_at = time.perf_counter()
            active_selection_ms += 1000.0 * (selected_at - started)
            riskset_build_ms += 1000.0 * (selected_at - started)
            if model in DETERMINISTIC_MODELS:
                scores = deterministic_logits(model, labels, step, candidates)
                prob = 1.0 / (1.0 + np.exp(-scores))
            else:
                prob = _predict_proba(payload, labels, step, candidates, history, sample=sample)
            scored_at = time.perf_counter()
            intensity_eval_ms += 1000.0 * (scored_at - selected_at)
            total_steps += 1
            true_mask = alive[step] & ~alive[step + 1]
            true_count = int(np.count_nonzero(true_mask))
            if model not in DETERMINISTIC_MODELS:
                history_started = time.perf_counter()
                history = update_history_trigger(history, true_mask, bonds)
                mechanics_feature_update_ms += 1000.0 * (time.perf_counter() - history_started)
            total_ms += 1000.0 * (time.perf_counter() - started)
            if true_count:
                y = true_mask[candidates]
                total_true += true_count
                captured_true += int(np.count_nonzero(y))
                precision, recall = topk_precision_recall(prob, y, max(top_k, true_count))
                top_precisions.append(precision)
                # Recall against the full event set, not only events inside candidates.
                top_recalls.append(recall * safe_div(float(np.count_nonzero(y)), float(true_count)))
        if sample.get("dense_step_ms") is not None:
            dense_ms.append(float(sample["dense_step_ms"]))
    total_step_ms = safe_div(total_ms, float(max(total_steps, 1)))
    riskset_build_step_ms = safe_div(riskset_build_ms, float(max(total_steps, 1)))
    mechanics_feature_step_ms = safe_div(mechanics_feature_update_ms, float(max(total_steps, 1)))
    intensity_eval_step_ms = safe_div(intensity_eval_ms, float(max(total_steps, 1)))
    active_selection_step_ms = safe_div(active_selection_ms, float(max(total_steps, 1)))
    pd_correction_step_ms = safe_div(pd_correction_ms, float(max(total_steps, 1)))
    dense_step_ms = float(np.mean(dense_ms)) if dense_ms else None
    return {
        "model": model,
        "run_variant": payload.get("run_variant") or "",
        "seed": payload.get("seed"),
        "split_id": split,
        "active_topk": int(active_topk),
        "top_k": int(top_k),
        "test_samples": len(samples),
        "event_recall_in_active_set": safe_div(float(captured_true), float(max(total_true, 1))),
        "missed_event_rate": 1.0 - safe_div(float(captured_true), float(max(total_true, 1))),
        "topK_event_precision": float(np.mean(top_precisions)) if top_precisions else None,
        "topK_event_recall_full_domain": float(np.mean(top_recalls)) if top_recalls else None,
        "active_ratio": safe_div(float(active_topk), float(np.mean(bond_counts))),
        "riskset_build_ms": riskset_build_step_ms,
        "mechanics_feature_update_ms": mechanics_feature_step_ms,
        "intensity_eval_ms": intensity_eval_step_ms,
        "active_selection_ms": active_selection_step_ms,
        "pd_correction_ms": pd_correction_step_ms,
        "table_write_ms": 0.0,
        "total_step_ms": total_step_ms,
        "total_step_ms_without_table_write": total_step_ms,
        "dense_step_ms": dense_step_ms,
        "wallclock_speedup_vs_dense_PD": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "wallclock_speedup_vs_dense_PD_without_table_write": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "pd_correction_ratio": safe_div(pd_correction_step_ms, total_step_ms),
        "uses_future_labels": bool(payload.get("uses_future_labels", False)),
        "dataset_hash": meta.get("dataset_hash") or manifest.get("dataset_hash"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crackle active-candidate wall-clock benchmark.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--active-topk", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [benchmark_model(args.data, _load_model(path), split=args.split, active_topk=args.active_topk, top_k=args.top_k) for path in args.model_dir]
    write_csv(args.out_dir / "crackle_wallclock_table.csv", rows)
    write_json(args.out_dir / "wallclock_metrics.json", {"schema": "crackle_wallclock_benchmark_v1", "rows": rows})
    print(json.dumps({"out_dir": str(args.out_dir), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
