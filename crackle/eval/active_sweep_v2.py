from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from crackle.baselines.deterministic import DETERMINISTIC_MODELS, deterministic_logits
from crackle.baselines.gbm import GBMHazard
from crackle.baselines.survival import LogisticHazard
from crackle.data.common import load_json, load_labels, ok_samples, safe_div, write_csv, write_json
from crackle.data.features import bond_geometry, feature_matrix, hybrid_feature_matrix, material_toughness, ranker_feature_matrix, update_history_trigger
from crackle.eval.eval_crackle import _load_model, _predict_proba
from crackle.eval.wallclock_benchmark import causal_candidate_bonds
from crackle.metrics.point_process import topk_precision_recall


_GBM_CACHE: dict[str, GBMHazard] = {}
_LOGISTIC_CACHE: dict[str, LogisticHazard] = {}


def _source_manifest(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    meta = load_json(data_dir / "metadata.json")
    return Path(meta["source_manifest"]), meta


def _nearest_loc_error(pred_centers: np.ndarray, true_centers: np.ndarray) -> float | None:
    if pred_centers.size == 0 or true_centers.size == 0:
        return None
    d2 = np.sum((pred_centers[:, None, :] - true_centers[None, :, :]) ** 2, axis=2)
    return float(np.mean(np.sqrt(np.min(d2, axis=0))))


def _gbm_predict(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    key = str(payload["model_dir"])
    hazard = _GBM_CACHE.get(key)
    if hazard is None:
        hazard = GBMHazard.from_json(payload)
        _GBM_CACHE[key] = hazard
    return hazard.predict_proba(features)


def _logistic_predict(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    key = str(payload["model_dir"])
    hazard = _LOGISTIC_CACHE.get(key)
    if hazard is None:
        hazard = LogisticHazard.from_json(payload)
        _LOGISTIC_CACHE[key] = hazard
    return hazard.predict_proba(features)


def _predict_feature_matrix(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if payload.get("kind") == "gbm_hazard":
        return _gbm_predict(payload, features)
    return _logistic_predict(payload, features)


def _feature_matrix_for_payload(
    payload: dict[str, Any],
    labels: dict[str, np.ndarray],
    sample: dict[str, Any],
    step: int,
    indices: np.ndarray,
    history: np.ndarray,
) -> np.ndarray:
    model = str(payload["model"])
    if model == "crackle_cox_mechanics_ranker_v1":
        return ranker_feature_matrix(labels, sample, step, indices, history_trigger=history, include_history=True)
    if model in {"cox_mechanics_hawkes_hybrid_v1", "crackle_cox_fast_ranker_v1"}:
        return hybrid_feature_matrix(labels, step, indices, history_trigger=history, include_history=True)
    return feature_matrix(labels, step, indices, history_trigger=history, include_history=True)


def benchmark_threshold_sweep_for_budget_batched_gbm(
    data_dir: Path,
    payload: dict[str, Any],
    *,
    split: str,
    active_topk: int,
    top_k: int,
    thresholds: list[float | None],
) -> list[dict[str, Any]]:
    manifest_path, meta = _source_manifest(data_dir)
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    if not samples:
        raise ValueError(f"no samples found for split {split!r}")
    model = str(payload["model"])
    threshold_keys = [""] + [str(float(t)) for t in thresholds if t is not None]
    threshold_values: dict[str, float | None] = {"": None}
    threshold_values.update({str(float(t)): float(t) for t in thresholds if t is not None})
    top_recalls: list[float] = []
    top_precisions: list[float] = []
    totals = {key: {"precisions": [], "recalls": [], "loc_errors": [], "selected_total": 0, "selected_true": 0} for key in threshold_keys}
    total_ms = 0.0
    active_selection_ms = 0.0
    mechanics_feature_update_ms = 0.0
    intensity_eval_ms = 0.0
    total_steps = 0
    total_true = 0
    captured_true = 0
    dense_ms: list[float] = []
    bond_counts: list[int] = []
    for sample in samples:
        sample_step_ms = 0.0
        labels = load_labels(sample)
        prep_started = time.perf_counter()
        bonds = labels["bonds"].astype(np.int64)
        bond_counts.append(int(bonds.shape[0]))
        toughness = material_toughness(labels, bonds)
        geom = bond_geometry(labels["reference_x"], bonds)
        alive = labels["bond_alive"].astype(bool)
        prep_ms = 1000.0 * (time.perf_counter() - prep_started)
        mechanics_feature_update_ms += prep_ms
        sample_step_ms += prep_ms
        history = np.zeros((alive.shape[1],), dtype=np.float64)
        feature_chunks: list[np.ndarray] = []
        step_records: list[dict[str, Any]] = []
        for step in range(alive.shape[0] - 1):
            step_started = time.perf_counter()
            candidates = causal_candidate_bonds(labels, step, active_topk=active_topk, bonds=bonds, toughness=toughness)
            selected_at = time.perf_counter()
            active_selection_ms += 1000.0 * (selected_at - step_started)
            feature_started = time.perf_counter()
            feature_chunks.append(_feature_matrix_for_payload(payload, labels, sample, step, candidates, history))
            intensity_eval_ms += 1000.0 * (time.perf_counter() - feature_started)
            true_mask = alive[step] & ~alive[step + 1]
            true_count = int(np.count_nonzero(true_mask))
            step_records.append({"candidates": candidates, "true_mask": true_mask, "true_count": true_count})
            history_started = time.perf_counter()
            history = update_history_trigger(history, true_mask, bonds)
            mechanics_feature_update_ms += 1000.0 * (time.perf_counter() - history_started)
            sample_step_ms += 1000.0 * (time.perf_counter() - step_started)
            total_steps += 1
        if feature_chunks:
            pred_started = time.perf_counter()
            prob_all = _predict_feature_matrix(payload, np.concatenate(feature_chunks, axis=0))
            pred_ms = 1000.0 * (time.perf_counter() - pred_started)
            intensity_eval_ms += pred_ms
            sample_step_ms += pred_ms
            offset = 0
            for record in step_records:
                candidates = record["candidates"]
                prob = prob_all[offset : offset + candidates.size]
                offset += candidates.size
                true_mask = record["true_mask"]
                true_count = int(record["true_count"])
                if true_count == 0:
                    continue
                y = true_mask[candidates]
                total_true += true_count
                captured = int(np.count_nonzero(y))
                captured_true += captured
                precision, recall = topk_precision_recall(prob, y, max(top_k, true_count))
                top_precisions.append(precision)
                top_recalls.append(recall * safe_div(float(captured), float(true_count)))
                true_centers = geom.centers[np.flatnonzero(true_mask)]
                for key in threshold_keys:
                    threshold = threshold_values[key]
                    if threshold is None:
                        final_k = min(max(top_k, true_count), prob.size)
                        selected_local = np.argpartition(prob, -final_k)[-final_k:]
                    else:
                        selected_local = np.flatnonzero(prob >= threshold)
                    bucket = totals[key]
                    bucket["selected_total"] += int(selected_local.size)
                    if selected_local.size:
                        hits = int(np.count_nonzero(y[selected_local]))
                        bucket["selected_true"] += hits
                        bucket["precisions"].append(safe_div(float(hits), float(selected_local.size)))
                        bucket["recalls"].append(safe_div(float(hits), float(true_count)))
                        loc = _nearest_loc_error(geom.centers[candidates[selected_local]], true_centers)
                        if loc is not None:
                            bucket["loc_errors"].append(loc)
                    else:
                        bucket["precisions"].append(0.0)
                        bucket["recalls"].append(0.0)
        total_ms += sample_step_ms
        if sample.get("dense_step_ms") is not None:
            dense_ms.append(float(sample["dense_step_ms"]))
    total_step_ms = safe_div(total_ms, float(max(total_steps, 1)))
    dense_step_ms = float(np.mean(dense_ms)) if dense_ms else None
    active_ratio = safe_div(float(active_topk), float(np.mean(bond_counts)))
    common = {
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
        "active_ratio": active_ratio,
        "active_selection_ms": safe_div(active_selection_ms, float(max(total_steps, 1))),
        "mechanics_feature_update_ms": safe_div(mechanics_feature_update_ms, float(max(total_steps, 1))),
        "intensity_eval_ms": safe_div(intensity_eval_ms, float(max(total_steps, 1))),
        "riskset_build_ms": safe_div(active_selection_ms, float(max(total_steps, 1))),
        "pd_correction_ms": 0.0,
        "table_write_ms": 0.0,
        "total_step_ms": total_step_ms,
        "total_step_ms_without_table_write": total_step_ms,
        "dense_step_ms": dense_step_ms,
        "wallclock_speedup_vs_dense_PD": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "wallclock_speedup_vs_dense_PD_without_table_write": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "uses_future_labels": bool(payload.get("uses_future_labels", False)),
        "dataset_hash": meta.get("dataset_hash") or manifest.get("dataset_hash"),
        "inference_batching": "per_sample_all_steps",
    }
    rows: list[dict[str, Any]] = []
    mean_bonds = max(float(np.mean(bond_counts)), 1.0)
    for key in threshold_keys:
        bucket = totals[key]
        correction_ratio = safe_div(float(bucket["selected_total"]), float(max(total_steps, 1)) * mean_bonds)
        rows.append(
            {
                **common,
                "prob_threshold": key,
                "threshold_event_precision": float(np.mean(bucket["precisions"])) if bucket["precisions"] else None,
                "threshold_event_recall_full_domain": float(np.mean(bucket["recalls"])) if bucket["recalls"] else None,
                "next_event_loc_err": float(np.mean(bucket["loc_errors"])) if bucket["loc_errors"] else None,
                "pd_correction_ratio": correction_ratio,
            }
        )
    return rows


def benchmark_operating_point(
    data_dir: Path,
    payload: dict[str, Any],
    *,
    split: str,
    active_topk: int,
    top_k: int,
    prob_threshold: float | None,
) -> dict[str, Any]:
    manifest_path, meta = _source_manifest(data_dir)
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    if not samples:
        raise ValueError(f"no samples found for split {split!r}")
    model = str(payload["model"])
    total_ms = 0.0
    active_selection_ms = 0.0
    mechanics_feature_update_ms = 0.0
    intensity_eval_ms = 0.0
    total_steps = 0
    total_true = 0
    captured_true = 0
    selected_true = 0
    selected_total = 0
    top_recalls: list[float] = []
    top_precisions: list[float] = []
    threshold_recalls: list[float] = []
    threshold_precisions: list[float] = []
    loc_errors: list[float] = []
    dense_ms: list[float] = []
    bond_counts: list[int] = []
    for sample in samples:
        labels = load_labels(sample)
        prep_started = time.perf_counter()
        bonds = labels["bonds"].astype(np.int64)
        bond_counts.append(int(bonds.shape[0]))
        toughness = material_toughness(labels, bonds)
        geom = bond_geometry(labels["reference_x"], bonds)
        alive = labels["bond_alive"].astype(bool)
        prep_ms = 1000.0 * (time.perf_counter() - prep_started)
        mechanics_feature_update_ms += prep_ms
        total_ms += prep_ms
        history = np.zeros((alive.shape[1],), dtype=np.float64)
        for step in range(alive.shape[0] - 1):
            step_started = time.perf_counter()
            candidates = causal_candidate_bonds(labels, step, active_topk=active_topk, bonds=bonds, toughness=toughness)
            selected_at = time.perf_counter()
            active_selection_ms += 1000.0 * (selected_at - step_started)
            if model in DETERMINISTIC_MODELS:
                prob = 1.0 / (1.0 + np.exp(-deterministic_logits(model, labels, step, candidates)))
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
            total_ms += 1000.0 * (time.perf_counter() - step_started)
            if true_count == 0:
                continue
            y = true_mask[candidates]
            total_true += true_count
            captured = int(np.count_nonzero(y))
            captured_true += captured
            precision, recall = topk_precision_recall(prob, y, max(top_k, true_count))
            top_precisions.append(precision)
            top_recalls.append(recall * safe_div(float(captured), float(true_count)))
            if prob_threshold is None:
                final_k = min(max(top_k, true_count), prob.size)
                selected_local = np.argpartition(prob, -final_k)[-final_k:]
            else:
                selected_local = np.flatnonzero(prob >= float(prob_threshold))
            selected_total += int(selected_local.size)
            if selected_local.size:
                hits = int(np.count_nonzero(y[selected_local]))
                selected_true += hits
                threshold_precisions.append(safe_div(float(hits), float(selected_local.size)))
                threshold_recalls.append(safe_div(float(hits), float(true_count)))
                true_centers = geom.centers[np.flatnonzero(true_mask)]
                pred_centers = geom.centers[candidates[selected_local]]
                loc = _nearest_loc_error(pred_centers, true_centers)
                if loc is not None:
                    loc_errors.append(loc)
            else:
                threshold_precisions.append(0.0)
                threshold_recalls.append(0.0)
        if sample.get("dense_step_ms") is not None:
            dense_ms.append(float(sample["dense_step_ms"]))
    total_step_ms = safe_div(total_ms, float(max(total_steps, 1)))
    dense_step_ms = float(np.mean(dense_ms)) if dense_ms else None
    active_ratio = safe_div(float(active_topk), float(np.mean(bond_counts)))
    correction_ratio = safe_div(float(selected_total), float(max(total_steps, 1) * max(float(np.mean(bond_counts)), 1.0)))
    return {
        "model": model,
        "run_variant": payload.get("run_variant") or "",
        "seed": payload.get("seed"),
        "split_id": split,
        "active_topk": int(active_topk),
        "top_k": int(top_k),
        "prob_threshold": prob_threshold if prob_threshold is not None else "",
        "test_samples": len(samples),
        "event_recall_in_active_set": safe_div(float(captured_true), float(max(total_true, 1))),
        "missed_event_rate": 1.0 - safe_div(float(captured_true), float(max(total_true, 1))),
        "topK_event_precision": float(np.mean(top_precisions)) if top_precisions else None,
        "topK_event_recall_full_domain": float(np.mean(top_recalls)) if top_recalls else None,
        "threshold_event_precision": float(np.mean(threshold_precisions)) if threshold_precisions else None,
        "threshold_event_recall_full_domain": float(np.mean(threshold_recalls)) if threshold_recalls else None,
        "next_event_loc_err": float(np.mean(loc_errors)) if loc_errors else None,
        "active_ratio": active_ratio,
        "pd_correction_ratio": correction_ratio,
        "active_selection_ms": safe_div(active_selection_ms, float(max(total_steps, 1))),
        "mechanics_feature_update_ms": safe_div(mechanics_feature_update_ms, float(max(total_steps, 1))),
        "intensity_eval_ms": safe_div(intensity_eval_ms, float(max(total_steps, 1))),
        "riskset_build_ms": safe_div(active_selection_ms, float(max(total_steps, 1))),
        "pd_correction_ms": 0.0,
        "table_write_ms": 0.0,
        "total_step_ms": total_step_ms,
        "total_step_ms_without_table_write": total_step_ms,
        "dense_step_ms": dense_step_ms,
        "wallclock_speedup_vs_dense_PD": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "wallclock_speedup_vs_dense_PD_without_table_write": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "uses_future_labels": bool(payload.get("uses_future_labels", False)),
        "dataset_hash": meta.get("dataset_hash") or manifest.get("dataset_hash"),
    }


def benchmark_threshold_sweep_for_budget(
    data_dir: Path,
    payload: dict[str, Any],
    *,
    split: str,
    active_topk: int,
    top_k: int,
    thresholds: list[float | None],
) -> list[dict[str, Any]]:
    if payload.get("kind") == "gbm_hazard" and os.environ.get("CRACKLE_GBM_BATCHED") == "1":
        return benchmark_threshold_sweep_for_budget_batched_gbm(
            data_dir,
            payload,
            split=split,
            active_topk=active_topk,
            top_k=top_k,
            thresholds=thresholds,
        )
    if payload.get("kind") == "logistic_hazard" and os.environ.get("CRACKLE_LOGISTIC_BATCHED", "1") == "1":
        return benchmark_threshold_sweep_for_budget_batched_gbm(
            data_dir,
            payload,
            split=split,
            active_topk=active_topk,
            top_k=top_k,
            thresholds=thresholds,
        )
    manifest_path, meta = _source_manifest(data_dir)
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    if not samples:
        raise ValueError(f"no samples found for split {split!r}")
    model = str(payload["model"])
    threshold_keys = [""] + [str(float(t)) for t in thresholds if t is not None]
    threshold_values: dict[str, float | None] = {"": None}
    threshold_values.update({str(float(t)): float(t) for t in thresholds if t is not None})
    top_recalls: list[float] = []
    top_precisions: list[float] = []
    totals = {key: {"precisions": [], "recalls": [], "loc_errors": [], "selected_total": 0, "selected_true": 0} for key in threshold_keys}
    total_ms = 0.0
    active_selection_ms = 0.0
    mechanics_feature_update_ms = 0.0
    intensity_eval_ms = 0.0
    total_steps = 0
    total_true = 0
    captured_true = 0
    dense_ms: list[float] = []
    bond_counts: list[int] = []
    for sample in samples:
        labels = load_labels(sample)
        prep_started = time.perf_counter()
        bonds = labels["bonds"].astype(np.int64)
        bond_counts.append(int(bonds.shape[0]))
        toughness = material_toughness(labels, bonds)
        geom = bond_geometry(labels["reference_x"], bonds)
        alive = labels["bond_alive"].astype(bool)
        prep_ms = 1000.0 * (time.perf_counter() - prep_started)
        mechanics_feature_update_ms += prep_ms
        total_ms += prep_ms
        history = np.zeros((alive.shape[1],), dtype=np.float64)
        for step in range(alive.shape[0] - 1):
            step_started = time.perf_counter()
            candidates = causal_candidate_bonds(labels, step, active_topk=active_topk, bonds=bonds, toughness=toughness)
            selected_at = time.perf_counter()
            active_selection_ms += 1000.0 * (selected_at - step_started)
            if model in DETERMINISTIC_MODELS:
                prob = 1.0 / (1.0 + np.exp(-deterministic_logits(model, labels, step, candidates)))
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
            total_ms += 1000.0 * (time.perf_counter() - step_started)
            if true_count == 0:
                continue
            y = true_mask[candidates]
            total_true += true_count
            captured = int(np.count_nonzero(y))
            captured_true += captured
            precision, recall = topk_precision_recall(prob, y, max(top_k, true_count))
            top_precisions.append(precision)
            top_recalls.append(recall * safe_div(float(captured), float(true_count)))
            true_centers = geom.centers[np.flatnonzero(true_mask)]
            for key in threshold_keys:
                threshold = threshold_values[key]
                if threshold is None:
                    final_k = min(max(top_k, true_count), prob.size)
                    selected_local = np.argpartition(prob, -final_k)[-final_k:]
                else:
                    selected_local = np.flatnonzero(prob >= threshold)
                bucket = totals[key]
                bucket["selected_total"] += int(selected_local.size)
                if selected_local.size:
                    hits = int(np.count_nonzero(y[selected_local]))
                    bucket["selected_true"] += hits
                    bucket["precisions"].append(safe_div(float(hits), float(selected_local.size)))
                    bucket["recalls"].append(safe_div(float(hits), float(true_count)))
                    loc = _nearest_loc_error(geom.centers[candidates[selected_local]], true_centers)
                    if loc is not None:
                        bucket["loc_errors"].append(loc)
                else:
                    bucket["precisions"].append(0.0)
                    bucket["recalls"].append(0.0)
        if sample.get("dense_step_ms") is not None:
            dense_ms.append(float(sample["dense_step_ms"]))
    total_step_ms = safe_div(total_ms, float(max(total_steps, 1)))
    dense_step_ms = float(np.mean(dense_ms)) if dense_ms else None
    active_ratio = safe_div(float(active_topk), float(np.mean(bond_counts)))
    common = {
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
        "active_ratio": active_ratio,
        "active_selection_ms": safe_div(active_selection_ms, float(max(total_steps, 1))),
        "mechanics_feature_update_ms": safe_div(mechanics_feature_update_ms, float(max(total_steps, 1))),
        "intensity_eval_ms": safe_div(intensity_eval_ms, float(max(total_steps, 1))),
        "riskset_build_ms": safe_div(active_selection_ms, float(max(total_steps, 1))),
        "pd_correction_ms": 0.0,
        "table_write_ms": 0.0,
        "total_step_ms": total_step_ms,
        "total_step_ms_without_table_write": total_step_ms,
        "dense_step_ms": dense_step_ms,
        "wallclock_speedup_vs_dense_PD": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "wallclock_speedup_vs_dense_PD_without_table_write": safe_div(dense_step_ms, total_step_ms) if dense_step_ms is not None else None,
        "uses_future_labels": bool(payload.get("uses_future_labels", False)),
        "dataset_hash": meta.get("dataset_hash") or manifest.get("dataset_hash"),
    }
    rows: list[dict[str, Any]] = []
    mean_bonds = max(float(np.mean(bond_counts)), 1.0)
    for key in threshold_keys:
        bucket = totals[key]
        correction_ratio = safe_div(float(bucket["selected_total"]), float(max(total_steps, 1)) * mean_bonds)
        rows.append(
            {
                **common,
                "prob_threshold": key,
                "threshold_event_precision": float(np.mean(bucket["precisions"])) if bucket["precisions"] else None,
                "threshold_event_recall_full_domain": float(np.mean(bucket["recalls"])) if bucket["recalls"] else None,
                "next_event_loc_err": float(np.mean(bucket["loc_errors"])) if bucket["loc_errors"] else None,
                "pd_correction_ratio": correction_ratio,
            }
        )
    return rows


def _mark_operating_modes(rows: list[dict[str, Any]]) -> None:
    eligible = [row for row in rows if row.get("wallclock_speedup_vs_dense_PD") is not None and row.get("topK_event_recall_full_domain") is not None]
    for row in rows:
        row["operating_mode"] = ""
    if not eligible:
        return
    speed = max(eligible, key=lambda row: float(row["wallclock_speedup_vs_dense_PD"]))
    accuracy = max(eligible, key=lambda row: float(row["topK_event_recall_full_domain"]))
    balanced_candidates = [row for row in eligible if float(row["wallclock_speedup_vs_dense_PD"]) >= 1.2]
    balanced = max(balanced_candidates or eligible, key=lambda row: float(row["topK_event_recall_full_domain"]))
    speed["operating_mode"] = (str(speed.get("operating_mode") or "") + "|speed").strip("|")
    accuracy["operating_mode"] = (str(accuracy.get("operating_mode") or "") + "|accuracy").strip("|")
    balanced["operating_mode"] = (str(balanced.get("operating_mode") or "") + "|balanced").strip("|")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crackle active sweep v2 over candidate budgets and probability thresholds.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--active-topk", type=int, action="append", default=None)
    parser.add_argument("--prob-threshold", type=float, action="append", default=None)
    parser.add_argument("--top-k", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = args.active_topk or [512, 1024, 2048, 4096]
    thresholds: list[float | None] = [None]
    thresholds.extend(args.prob_threshold or [0.001, 0.005, 0.01, 0.02])
    rows: list[dict[str, Any]] = []
    for model_dir in args.model_dir:
        payload = _load_model(model_dir)
        for budget in budgets:
            rows.extend(
                benchmark_threshold_sweep_for_budget(
                    args.data,
                    payload,
                    split=args.split,
                    active_topk=budget,
                    top_k=args.top_k,
                    thresholds=thresholds,
                )
            )
    _mark_operating_modes(rows)
    write_csv(args.out_dir / "crackle_active_sweep_v2.csv", rows)
    write_json(args.out_dir / "crackle_active_sweep_v2.json", {"schema": "crackle_active_sweep_v2", "rows": rows})
    print(json.dumps({"out_dir": str(args.out_dir), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
