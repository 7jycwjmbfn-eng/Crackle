from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from crackle.baselines.deterministic import DETERMINISTIC_MODELS, deterministic_logits
from crackle.baselines.gbm import GBMHazard
from crackle.baselines.hawkes import hawkes_history_by_step
from crackle.baselines.survival import LogisticHazard
from crackle.calibration import IntensityCalibrator, fit_intensity_calibrators, qq_max_deviation_exp
from crackle.data.common import load_json, load_labels, ok_samples, safe_div, sigmoid, write_csv, write_json
from crackle.data.features import bond_geometry, feature_matrix, hybrid_feature_matrix, ranker_feature_matrix
from crackle.metrics.point_process import ks_exp_pvalue, topk_precision_recall


SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "run_variant",
    "benchmark_case_id",
    "split_id",
    "seed",
    "dataset_hash",
    "teacher_version",
    "claim_allowed",
    "uses_future_labels",
    "oracle_row",
    "riskset_NLL",
    "censored_survival_NLL",
    "time_rescaling_KS_pvalue",
    "time_rescaling_KS_stat",
    "QQ_max_deviation",
    "Brier_next_event_window",
    "calibration_ECE_next_event",
    "calibration_method",
    "calibrated_riskset_NLL",
    "calibrated_Brier_next_event_window",
    "calibrated_ECE_next_event",
    "calibrated_time_rescaling_KS_pvalue",
    "calibrated_time_rescaling_KS_stat",
    "calibrated_QQ_max_deviation",
    "time_rescaling_not_a_gate_reason",
    "Brier_skill",
    "calibrated_Brier_skill",
    "topK_event_precision",
    "topK_event_recall",
    "next_event_time_err",
    "next_event_loc_err",
    "damage_iou_full_domain",
    "bond_f1_full_domain",
    "missed_crack_rate_full_domain",
    "false_crack_rate_full_domain",
    "crack_tip_error_mm",
    "crack_path_error_mm",
    "time_to_failure_err",
    "branch_detect_f1",
    "arrest_decision_acc",
    "wallclock_speedup_vs_dense_PD",
    "bond_count_speedup",
    "active_set_overhead_ratio",
    "total_step_ms",
    "gpu_util_avg",
    "gpu_mem_peak_mb",
    "excluded_sample_rate",
    "G0_pass",
    "G1a_event_forecasting_pass",
    "G1b_probability_scoring_pass",
    "G2_pass",
    "G_calibration_spatial_pass",
    "G3a_mechanics_vs_catalog_pass",
    "G3b_mechanics_vs_strong_survival_pass",
    "G4_pass",
    "G5_pass",
]


_GBM_CACHE: dict[str, GBMHazard] = {}
_LOGISTIC_CACHE: dict[str, LogisticHazard] = {}


def _source_manifest(data_dir: Path) -> tuple[Path, dict[str, Any]]:
    meta = load_json(data_dir / "metadata.json")
    return Path(meta["source_manifest"]), meta


def _load_model(model_dir: Path) -> dict[str, Any]:
    payload = load_json(model_dir / "model.json")
    payload["model_dir"] = str(model_dir)
    return payload


def _predict_proba(
    payload: dict[str, Any],
    labels: dict[str, np.ndarray],
    step: int,
    indices: np.ndarray,
    history: np.ndarray,
    *,
    sample: dict[str, Any] | None = None,
) -> np.ndarray:
    model = str(payload["model"])
    if model in DETERMINISTIC_MODELS:
        return sigmoid(deterministic_logits(model, labels, step, indices))
    if payload.get("kind") == "gbm_hazard":
        key = str(payload["model_dir"])
        hazard = _GBM_CACHE.get(key)
        if hazard is None:
            hazard = GBMHazard.from_json(payload)
            _GBM_CACHE[key] = hazard
        x = feature_matrix(labels, step, indices, history_trigger=history, include_history=True)
        return hazard.predict_proba(x)
    key = str(payload["model_dir"])
    hazard = _LOGISTIC_CACHE.get(key)
    if hazard is None:
        hazard = LogisticHazard.from_json(payload)
        _LOGISTIC_CACHE[key] = hazard
    if model == "crackle_cox_mechanics_ranker_v1":
        x = ranker_feature_matrix(labels, sample, step, indices, history_trigger=history, include_history=True)
    elif model in {"cox_mechanics_hawkes_hybrid_v1", "crackle_cox_fast_ranker_v1"}:
        x = hybrid_feature_matrix(labels, step, indices, history_trigger=history, include_history=True)
    else:
        x = feature_matrix(labels, step, indices, history_trigger=history, include_history=True)
    return hazard.predict_proba(x)


def _case_label(samples: list[dict[str, Any]], key: str) -> str:
    values = sorted({str(sample.get(key) or sample.get("split") or "") for sample in samples if sample.get(key) or sample.get("split")})
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "mixed:" + "|".join(values)


def _ece_from_bins(bin_total: np.ndarray, bin_prob: np.ndarray, bin_true: np.ndarray) -> float:
    n = float(np.sum(bin_total))
    if n <= 0.0:
        return 0.0
    ece = 0.0
    for count, p_sum, y_sum in zip(bin_total, bin_prob, bin_true):
        if count <= 0:
            continue
        ece += float(count) / n * abs(float(p_sum) / float(count) - float(y_sum) / float(count))
    return float(ece)


def _sample_indices_for_calibration(
    rng: np.random.Generator,
    alive_idx: np.ndarray,
    event_mask: np.ndarray,
    *,
    max_neg_per_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    positives = np.flatnonzero(event_mask)
    neg_candidates = alive_idx[~event_mask[alive_idx]]
    neg_count = min(int(max_neg_per_step), int(neg_candidates.size))
    if neg_count > 0:
        neg_idx = rng.choice(neg_candidates, size=neg_count, replace=False)
        neg_weight = float(neg_candidates.size) / float(max(neg_count, 1))
    else:
        neg_idx = np.empty((0,), dtype=np.int64)
        neg_weight = 1.0
    indices = np.concatenate([positives, neg_idx])
    weights = np.concatenate([np.ones((positives.size,), dtype=np.float64), np.full((neg_idx.size,), neg_weight, dtype=np.float64)])
    return indices, weights


def fit_validation_calibrator(
    data_dir: Path,
    payload: dict[str, Any],
    *,
    split: str,
    max_neg_per_step: int,
    seed: int,
) -> tuple[IntensityCalibrator, list[dict[str, float | str]]]:
    manifest_path, _ = _source_manifest(data_dir)
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    if not samples:
        return IntensityCalibrator(method="none"), [{"method": "none", "NLL": 0.0, "Brier": 0.0, "ECE": 0.0}]
    rng = np.random.default_rng(seed)
    prob_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    for sample in samples:
        labels = load_labels(sample)
        alive = labels["bond_alive"].astype(bool)
        histories = hawkes_history_by_step(labels)
        for step in range(alive.shape[0] - 1):
            current_alive = alive[step]
            alive_idx = np.flatnonzero(current_alive)
            if alive_idx.size == 0:
                continue
            event_mask = current_alive & ~alive[step + 1]
            indices, weights = _sample_indices_for_calibration(rng, alive_idx, event_mask, max_neg_per_step=max_neg_per_step)
            if indices.size == 0:
                continue
            prob = _predict_proba(payload, labels, step, indices, histories[step], sample=sample)
            target = event_mask[indices].astype(np.float64)
            prob_chunks.append(prob)
            target_chunks.append(target)
            weight_chunks.append(weights)
            time_chunks.append(np.full((indices.size,), float(step) / max(alive.shape[0] - 1, 1), dtype=np.float64))
    if not prob_chunks:
        return IntensityCalibrator(method="none"), [{"method": "none", "NLL": 0.0, "Brier": 0.0, "ECE": 0.0}]
    return fit_intensity_calibrators(
        np.concatenate(prob_chunks),
        np.concatenate(target_chunks),
        time_frac=np.concatenate(time_chunks),
        weight=np.concatenate(weight_chunks),
    )


def evaluate_model(
    data_dir: Path,
    payload: dict[str, Any],
    *,
    split: str,
    top_k: int,
    max_eval_samples: int | None = None,
    case_id: str | None = None,
    calibrator: IntensityCalibrator | None = None,
) -> dict[str, Any]:
    manifest_path, meta = _source_manifest(data_dir)
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    if case_id is not None:
        samples = [sample for sample in samples if str(sample.get("benchmark_case_id", "")) == str(case_id)]
    if max_eval_samples is not None:
        samples = samples[: int(max_eval_samples)]
    if not samples:
        raise ValueError(f"no samples found for split {split!r}")
    model = str(payload["model"])
    seed = int(payload.get("seed") or 0)
    nll_sum = 0.0
    brier_sum = 0.0
    cal_nll_sum = 0.0
    cal_brier_sum = 0.0
    climatology_brier_sum = 0.0
    risk_count = 0
    top_precisions: list[float] = []
    top_recalls: list[float] = []
    loc_errors: list[float] = []
    first_time_errors: list[float] = []
    failure_time_errors: list[float] = []
    rescaled: list[float] = []
    cal_rescaled: list[float] = []
    bin_total = np.zeros((12,), dtype=np.float64)
    bin_prob = np.zeros_like(bin_total)
    bin_true = np.zeros_like(bin_total)
    cal_bin_total = np.zeros((12,), dtype=np.float64)
    cal_bin_prob = np.zeros_like(cal_bin_total)
    cal_bin_true = np.zeros_like(cal_bin_total)
    intensity_ms = 0.0
    total_steps = 0
    dense_step_ms: list[float] = []
    for sample in samples:
        labels = load_labels(sample)
        bonds = labels["bonds"].astype(np.int64)
        alive = labels["bond_alive"].astype(bool)
        geom = bond_geometry(labels["reference_x"], bonds)
        histories = hawkes_history_by_step(labels)
        true_first: int | None = None
        pred_first: int | None = None
        true_last: int | None = None
        pred_last: int | None = None
        residual_integral = 0.0
        cal_residual_integral = 0.0
        for step in range(alive.shape[0] - 1):
            total_steps += 1
            current_alive = alive[step]
            event_mask = current_alive & ~alive[step + 1]
            alive_idx = np.flatnonzero(current_alive)
            if alive_idx.size == 0:
                continue
            started = time.perf_counter()
            prob = _predict_proba(payload, labels, step, alive_idx, histories[step], sample=sample)
            intensity_ms += 1000.0 * (time.perf_counter() - started)
            time_frac = float(step) / max(alive.shape[0] - 1, 1)
            cal_prob = calibrator.apply(prob, time_frac=time_frac) if calibrator is not None else prob
            y = event_mask[alive_idx].astype(np.float64)
            p = np.clip(prob, 1e-7, 1.0 - 1e-7)
            cp = np.clip(cal_prob, 1e-7, 1.0 - 1e-7)
            nll_sum += float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            brier_sum += float(np.sum((p - y) ** 2))
            cal_nll_sum += float(-np.sum(y * np.log(cp) + (1.0 - y) * np.log(1.0 - cp)))
            cal_brier_sum += float(np.sum((cp - y) ** 2))
            event_rate = float(np.mean(y)) if y.size else 0.0
            climatology_brier_sum += float(np.sum((event_rate - y) ** 2))
            risk_count += int(y.size)
            bins = np.minimum((p * bin_total.size).astype(np.int64), bin_total.size - 1)
            np.add.at(bin_total, bins, 1.0)
            np.add.at(bin_prob, bins, p)
            np.add.at(bin_true, bins, y)
            cal_bins = np.minimum((cp * cal_bin_total.size).astype(np.int64), cal_bin_total.size - 1)
            np.add.at(cal_bin_total, cal_bins, 1.0)
            np.add.at(cal_bin_prob, cal_bins, cp)
            np.add.at(cal_bin_true, cal_bins, y)
            event_count = int(np.count_nonzero(event_mask))
            expected_events = float(np.sum(prob))
            cal_expected_events = float(np.sum(cal_prob))
            residual_integral += expected_events
            cal_residual_integral += cal_expected_events
            if event_count:
                if true_first is None:
                    true_first = step + 1
                true_last = step + 1
                k = max(int(top_k), event_count)
                precision, recall = topk_precision_recall(prob, y, k)
                top_precisions.append(precision)
                top_recalls.append(recall)
                top_local = np.argpartition(prob, -min(k, prob.size))[-min(k, prob.size) :]
                pred_centers = geom.centers[alive_idx[top_local]]
                true_centers = geom.centers[np.flatnonzero(event_mask)]
                d2 = np.sum((pred_centers[:, None, :] - true_centers[None, :, :]) ** 2, axis=2)
                loc_errors.append(float(np.mean(np.sqrt(np.min(d2, axis=0)))))
                rescaled.append(residual_integral / max(event_count, 1))
                cal_rescaled.append(cal_residual_integral / max(event_count, 1))
                residual_integral = 0.0
                cal_residual_integral = 0.0
            if pred_first is None and expected_events >= 0.5:
                pred_first = step + 1
            if expected_events >= 0.5:
                pred_last = step + 1
        dt = float(sample.get("time_step_dt") or 1.0 / max(alive.shape[0] - 1, 1))
        if true_first is not None:
            first_time_errors.append(abs(float((pred_first or alive.shape[0]) - true_first)) * dt)
        if true_last is not None:
            failure_time_errors.append(abs(float((pred_last or alive.shape[0]) - true_last)) * dt)
        if sample.get("dense_step_ms") is not None:
            dense_step_ms.append(float(sample["dense_step_ms"]))
    ks_stat, ks_p = ks_exp_pvalue(rescaled)
    cal_ks_stat, cal_ks_p = ks_exp_pvalue(cal_rescaled)
    total_step_ms = safe_div(intensity_ms, float(max(total_steps, 1)))
    dense_ms = float(np.mean(dense_step_ms)) if dense_step_ms else None
    speedup = safe_div(dense_ms, total_step_ms) if dense_ms is not None else None
    brier = safe_div(brier_sum, float(max(risk_count, 1)))
    cal_brier = safe_div(cal_brier_sum, float(max(risk_count, 1)))
    clim_brier = safe_div(climatology_brier_sum, float(max(risk_count, 1)))
    brier_skill = 1.0 - safe_div(brier, clim_brier) if clim_brier > 0.0 else None
    cal_brier_skill = 1.0 - safe_div(cal_brier, clim_brier) if clim_brier > 0.0 else None
    cal_ece = _ece_from_bins(cal_bin_total, cal_bin_prob, cal_bin_true)
    return {
        "run_id": f"{model}_seed{seed}_{split}",
        "model": model,
        "run_variant": payload.get("run_variant") or "",
        "benchmark_case_id": _case_label(samples, "benchmark_case_id"),
        "split_id": _case_label(samples, "split_id"),
        "seed": seed,
        "dataset_hash": meta.get("dataset_hash") or manifest.get("dataset_hash"),
        "teacher_version": _case_label(samples, "teacher_version") or "dense_horizon_pd_v0",
        "claim_allowed": False,
        "uses_future_labels": bool(payload.get("uses_future_labels", False)),
        "oracle_row": bool(payload.get("oracle_row", False)),
        "riskset_NLL": safe_div(nll_sum, float(max(risk_count, 1))),
        "censored_survival_NLL": safe_div(nll_sum, float(max(risk_count, 1))),
        "time_rescaling_KS_pvalue": ks_p,
        "time_rescaling_KS_stat": ks_stat,
        "QQ_max_deviation": qq_max_deviation_exp(rescaled),
        "Brier_next_event_window": brier,
        "calibration_ECE_next_event": _ece_from_bins(bin_total, bin_prob, bin_true),
        "calibration_method": calibrator.method if calibrator is not None else "none",
        "calibrated_riskset_NLL": safe_div(cal_nll_sum, float(max(risk_count, 1))),
        "calibrated_Brier_next_event_window": cal_brier,
        "calibrated_ECE_next_event": cal_ece,
        "calibrated_time_rescaling_KS_pvalue": cal_ks_p,
        "calibrated_time_rescaling_KS_stat": cal_ks_stat,
        "calibrated_QQ_max_deviation": qq_max_deviation_exp(cal_rescaled),
        "time_rescaling_not_a_gate_reason": "degenerate_time_axis",
        "Brier_skill": brier_skill,
        "calibrated_Brier_skill": cal_brier_skill,
        "topK_event_precision": float(np.mean(top_precisions)) if top_precisions else None,
        "topK_event_recall": float(np.mean(top_recalls)) if top_recalls else None,
        "next_event_time_err": float(np.mean(first_time_errors)) if first_time_errors else None,
        "next_event_loc_err": float(np.mean(loc_errors)) if loc_errors else None,
        "damage_iou_full_domain": None,
        "bond_f1_full_domain": None,
        "missed_crack_rate_full_domain": None,
        "false_crack_rate_full_domain": None,
        "crack_tip_error_mm": None,
        "crack_path_error_mm": None,
        "time_to_failure_err": float(np.mean(failure_time_errors)) if failure_time_errors else None,
        "branch_detect_f1": None,
        "arrest_decision_acc": None,
        "wallclock_speedup_vs_dense_PD": speedup,
        "bond_count_speedup": None,
        "active_set_overhead_ratio": 1.0,
        "total_step_ms": total_step_ms,
        "gpu_util_avg": None,
        "gpu_mem_peak_mb": None,
        "excluded_sample_rate": 0.0,
        "G0_pass": (not bool(payload.get("uses_future_labels", False))) and (not bool(payload.get("oracle_row", False))),
        "G1a_event_forecasting_pass": None,
        "G1b_probability_scoring_pass": None,
        "G2_pass": bool(speedup is not None and speedup >= 1.2),
        "G3a_mechanics_vs_catalog_pass": None,
        "G3b_mechanics_vs_strong_survival_pass": None,
        "G_calibration_spatial_pass": bool(cal_ece <= 0.05 and (cal_brier_skill is not None and cal_brier_skill > 0.0)),
        "G4_pass": bool(cal_ece <= 0.05 and (cal_brier_skill is not None and cal_brier_skill > 0.0)),
        "G5_pass": bool(cal_ece <= 0.05 and (cal_brier_skill is not None and cal_brier_skill > 0.0)),
    }


def _apply_relative_gates(rows: list[dict[str, Any]]) -> None:
    event_baseline = next((row for row in rows if row["model"] == "deterministic_event_ranker_v1"), None)
    if event_baseline is None:
        event_baseline = next((row for row in rows if row["model"] == "deterministic_threshold_energy_v1b"), None)
    prob_baselines = [
        row
        for row in rows
        if row["model"]
        in {
            "deterministic_threshold_geometry_v1a",
            "deterministic_threshold_energy_v1b",
            "cox_discrete_time_v1",
            "parametric_hawkes_etas_graph_v1",
        }
    ]
    if event_baseline is None or not prob_baselines:
        return
    best_nll = min(float(row.get("calibrated_riskset_NLL") or row.get("riskset_NLL") or 1e9) for row in prob_baselines)
    best_brier = min(float(row.get("calibrated_Brier_next_event_window") or row.get("Brier_next_event_window") or 1e9) for row in prob_baselines)
    best_ece = min(float(row.get("calibrated_ECE_next_event") or row.get("calibration_ECE_next_event") or 1e9) for row in prob_baselines)
    catalog = next((row for row in rows if row["model"] == "catalog_only_hawkes_v1"), None)
    strongest_survival = next((row for row in rows if row["model"] == "cox_discrete_time_v1"), None)
    for row in rows:
        if row is event_baseline:
            row["G1a_event_forecasting_pass"] = None
        else:
            loc_ok = row.get("next_event_loc_err") is not None and event_baseline.get("next_event_loc_err") is not None and row["next_event_loc_err"] <= 0.5 * event_baseline["next_event_loc_err"]
            recall_ok = row.get("topK_event_recall") is not None and event_baseline.get("topK_event_recall") is not None and row["topK_event_recall"] >= event_baseline["topK_event_recall"] + 0.20
            ttf_ok = row.get("time_to_failure_err") is not None and event_baseline.get("time_to_failure_err") is not None and row["time_to_failure_err"] <= event_baseline["time_to_failure_err"]
            row["G1a_event_forecasting_pass"] = bool(loc_ok and recall_ok and ttf_ok)
        row["G1b_probability_scoring_pass"] = bool(
            row.get("calibrated_riskset_NLL") is not None
            and row.get("calibrated_Brier_next_event_window") is not None
            and row.get("calibrated_ECE_next_event") is not None
            and float(row["calibrated_riskset_NLL"]) <= 0.95 * best_nll
            and float(row["calibrated_Brier_next_event_window"]) <= 0.95 * best_brier
            and float(row["calibrated_ECE_next_event"]) <= 0.95 * best_ece
        )
        row["G_calibration_spatial_pass"] = bool(
            row.get("calibrated_ECE_next_event") is not None
            and float(row["calibrated_ECE_next_event"]) <= 0.05
            and row.get("calibrated_Brier_skill") is not None
            and float(row["calibrated_Brier_skill"]) > 0.0
        )
        if catalog is not None:
            row["G3a_mechanics_vs_catalog_pass"] = bool(
                row["model"] in {"mechanics_coupled_survival_v1", "cox_mechanics_hawkes_hybrid_v1"}
                and row.get("topK_event_recall") is not None
                and catalog.get("topK_event_recall") is not None
                and row["topK_event_recall"] > catalog["topK_event_recall"]
                and row.get("next_event_loc_err") is not None
                and catalog.get("next_event_loc_err") is not None
                and row["next_event_loc_err"] < catalog["next_event_loc_err"]
            )
        if strongest_survival is not None:
            row["G3b_mechanics_vs_strong_survival_pass"] = bool(
                row["model"] == "cox_mechanics_hawkes_hybrid_v1"
                and row.get("topK_event_recall") is not None
                and strongest_survival.get("topK_event_recall") is not None
                and row["topK_event_recall"] >= strongest_survival["topK_event_recall"]
                and row.get("next_event_loc_err") is not None
                and strongest_survival.get("next_event_loc_err") is not None
                and row["next_event_loc_err"] <= strongest_survival["next_event_loc_err"]
            )


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    metrics = [
        "riskset_NLL",
        "calibrated_riskset_NLL",
        "Brier_next_event_window",
        "calibrated_Brier_next_event_window",
        "calibration_ECE_next_event",
        "calibrated_ECE_next_event",
        "topK_event_precision",
        "topK_event_recall",
        "next_event_loc_err",
        "next_event_time_err",
        "time_to_failure_err",
        "wallclock_speedup_vs_dense_PD",
        "total_step_ms",
        "calibrated_time_rescaling_KS_pvalue",
        "calibrated_QQ_max_deviation",
        "Brier_skill",
        "calibrated_Brier_skill",
    ]
    out: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        agg: dict[str, Any] = {"model": model, "seeds": len(model_rows)}
        methods = sorted({str(row.get("calibration_method") or "none") for row in model_rows})
        agg["calibration_methods"] = "|".join(methods)
        for metric in metrics:
            vals = [_numeric(row.get(metric)) for row in model_rows]
            nums = np.asarray([val for val in vals if val is not None], dtype=np.float64)
            agg[f"{metric}_mean"] = float(np.mean(nums)) if nums.size else None
            agg[f"{metric}_std"] = float(np.std(nums, ddof=1)) if nums.size > 1 else 0.0 if nums.size == 1 else None
        out.append(agg)
    return out


def _best_by(rows: list[dict[str, Any]], metric: str, *, higher: bool) -> dict[str, Any] | None:
    candidates = [row for row in rows if _numeric(row.get(metric)) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[metric])) if higher else min(candidates, key=lambda row: float(row[metric]))


def build_gate_report(rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    best_recall = _best_by(rows, "topK_event_recall", higher=True)
    best_loc = _best_by(rows, "next_event_loc_err", higher=False)
    best_cal_nll = _best_by(rows, "calibrated_riskset_NLL", higher=False)
    speed_best = _best_by(rows, "wallclock_speedup_vs_dense_PD", higher=True)
    any_g1a = any(bool(row.get("G1a_event_forecasting_pass")) for row in rows)
    any_g1b = any(bool(row.get("G1b_probability_scoring_pass")) for row in rows)
    any_g2 = any(bool(row.get("G2_pass")) for row in rows)
    any_g3a = any(bool(row.get("G3a_mechanics_vs_catalog_pass")) for row in rows)
    any_g3b = any(bool(row.get("G3b_mechanics_vs_strong_survival_pass")) for row in rows)
    any_spatial_cal = any(bool(row.get("G_calibration_spatial_pass") or row.get("G4_pass")) for row in rows)
    return {
        "schema": "crackle_gate_report_v1_3_spatial_calibration",
        "G0_data_validity": {"pass": all(bool(row.get("G0_pass")) for row in rows), "note": "all evaluated rows are causal, non-oracle rows"},
        "G1a_event_forecasting": {
            "pass": any_g1a,
            "baseline": "deterministic_event_ranker_v1 if available, otherwise deterministic_threshold_energy_v1b",
            "criterion": "recall >= baseline + 0.20, loc_err <= 0.5 baseline, time_err <= baseline",
        },
        "G1b_probability_scoring": {
            "pass": any_g1b,
            "criterion": "calibrated NLL/Brier/ECE each improve >=5% over the best calibrated baseline pool",
        },
        "G2_speed": {"pass": any_g2, "criterion": "wallclock_speedup_vs_dense_PD >= 1.2"},
        "G3a_mechanics_vs_catalog": {"pass": any_g3a, "criterion": "mechanics/hybrid beats catalog-only on recall and localization"},
        "G3b_mechanics_vs_strong_survival": {"pass": any_g3b, "criterion": "hybrid beats cox_discrete_time_v1 on recall and localization"},
        "G_calibration_spatial": {
            "pass": any_spatial_cal,
            "criterion": "calibrated_ECE_next_event <= 0.05 and calibrated_Brier_skill > 0",
        },
        "G4_calibration": {
            "pass": any_spatial_cal,
            "criterion": "spatial ECE/Brier gate; time-rescaling KS retained only as diagnostic",
            "not_a_gate_reason": "degenerate_time_axis",
        },
        "claim_allowed": bool(any_g1a and any_g1b and any_g2 and any_g3a and any_spatial_cal),
        "best_models": {
            "topK_event_recall": best_recall,
            "next_event_loc_err": best_loc,
            "calibrated_riskset_NLL": best_cal_nll,
            "wallclock_speedup_vs_dense_PD": speed_best,
        },
        "aggregate": aggregate,
    }


def write_gate_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Crackle Gate Report v1.3",
        "",
        f"- G0 data validity: {report['G0_data_validity']['pass']}",
        f"- G1a event forecasting: {report['G1a_event_forecasting']['pass']}",
        f"- G1b probability scoring: {report['G1b_probability_scoring']['pass']}",
        f"- G2 speed: {report['G2_speed']['pass']}",
        f"- G3a mechanics vs catalog: {report['G3a_mechanics_vs_catalog']['pass']}",
        f"- G3b mechanics vs strong survival: {report['G3b_mechanics_vs_strong_survival']['pass']}",
        f"- G calibration spatial: {report['G_calibration_spatial']['pass']}",
        f"- time-rescaling KS gate status: diagnostic only ({report['G4_calibration']['not_a_gate_reason']})",
        f"- claim_allowed: {report['claim_allowed']}",
        "",
        "## Best Models",
    ]
    for metric, row in report["best_models"].items():
        if row is None:
            continue
        lines.append(f"- {metric}: {row.get('model')} = {row.get(metric)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Crackle baselines and write crackle_summary_table.csv.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--calibration-max-neg-per-step", type=int, default=4096)
    parser.add_argument("--calibration-seed", type=int, default=20260605)
    parser.add_argument("--case-breakdown", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibrators: dict[str, IntensityCalibrator] = {}
    calibration_scores: dict[str, list[dict[str, float | str]]] = {}
    payloads = [_load_model(path) for path in args.model_dir]
    for payload in payloads:
        key = str(payload["model_dir"])
        calibrator, scores = fit_validation_calibrator(
            args.data,
            payload,
            split=args.calibration_split,
            max_neg_per_step=args.calibration_max_neg_per_step,
            seed=args.calibration_seed + int(payload.get("seed") or 0),
        )
        calibrators[key] = calibrator
        calibration_scores[key] = scores
    rows = [
        evaluate_model(
            args.data,
            payload,
            split=args.split,
            top_k=args.top_k,
            max_eval_samples=args.max_eval_samples,
            calibrator=calibrators[str(payload["model_dir"])],
        )
        for payload in payloads
    ]
    _apply_relative_gates(rows)
    write_csv(args.out_dir / "crackle_summary_table.csv", rows, SUMMARY_COLUMNS)
    aggregate = aggregate_rows(rows)
    write_csv(args.out_dir / "crackle_model_aggregate.csv", aggregate)
    gate_report = build_gate_report(rows, aggregate)
    write_json(args.out_dir / "crackle_gate_report_v1_2.json", gate_report)
    write_json(args.out_dir / "gate_report.json", gate_report)
    write_gate_markdown(args.out_dir / "crackle_gate_report_v1_2.md", gate_report)
    case_rows: list[dict[str, Any]] = []
    if args.case_breakdown:
        manifest_path, _ = _source_manifest(args.data)
        manifest = load_json(manifest_path)
        case_ids = sorted({str(sample.get("benchmark_case_id")) for sample in ok_samples(manifest, split=args.split) if sample.get("benchmark_case_id")})
        for case_id in case_ids:
            for payload in payloads:
                case_rows.append(
                    evaluate_model(
                        args.data,
                        payload,
                        split=args.split,
                        top_k=args.top_k,
                        max_eval_samples=args.max_eval_samples,
                        case_id=case_id,
                        calibrator=calibrators[str(payload["model_dir"])],
                    )
        )
        _apply_relative_gates(case_rows)
        write_csv(args.out_dir / "crackle_case_breakdown.csv", case_rows, SUMMARY_COLUMNS)
    write_json(args.out_dir / "calibration_scores.json", {"schema": "crackle_calibration_scores_v1", "scores": calibration_scores})
    write_json(args.out_dir / "metrics.json", {"schema": "crackle_eval_metrics_v1_2", "split": args.split, "rows": rows, "case_rows": case_rows})
    print(json.dumps({"out_dir": str(args.out_dir), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
