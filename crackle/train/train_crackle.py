from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from crackle.baselines.deterministic import DETERMINISTIC_MODELS, deterministic_model_payload
from crackle.baselines.gbm import fit_gbm_hazard
from crackle.baselines.hawkes import hawkes_history_by_step
from crackle.baselines.survival import fit_logistic_hazard, fit_logistic_hazard_with_pairwise
from crackle.data.common import load_json, load_labels, ok_samples, write_json
from crackle.data.features import FEATURE_NAMES, HYBRID_FEATURE_NAMES, RANKER_FEATURE_NAMES, feature_matrix, hybrid_feature_matrix, ranker_feature_matrix


TRAINABLE_MODELS = {
    "cox_discrete_time_v1",
    "weibull_survival_v1",
    "catalog_only_hawkes_v1",
    "parametric_hawkes_etas_graph_v1",
    "mechanics_coupled_survival_v1",
    "cox_mechanics_hawkes_hybrid_v1",
    "gbm_survival_v1",
    "crackle_cox_mechanics_ranker_v1",
    "crackle_cox_fast_ranker_v1",
}


def _source_manifest(data_dir: Path) -> Path:
    meta = load_json(data_dir / "metadata.json")
    return Path(meta["source_manifest"])


def _selected_features(model: str) -> tuple[list[int], list[str], bool]:
    if model in ("cox_discrete_time_v1", "weibull_survival_v1"):
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return indices, [FEATURE_NAMES[i] for i in indices], False
    if model == "catalog_only_hawkes_v1":
        indices = [0, 9, 10]
        return indices, [FEATURE_NAMES[i] for i in indices], True
    if model == "parametric_hawkes_etas_graph_v1":
        indices = [0, 4, 6, 9, 10]
        return indices, [FEATURE_NAMES[i] for i in indices], True
    if model == "mechanics_coupled_survival_v1":
        indices = list(range(len(FEATURE_NAMES)))
        return indices, FEATURE_NAMES.copy(), True
    if model == "cox_mechanics_hawkes_hybrid_v1":
        indices = list(range(len(HYBRID_FEATURE_NAMES)))
        return indices, HYBRID_FEATURE_NAMES.copy(), True
    if model == "gbm_survival_v1":
        indices = list(range(len(FEATURE_NAMES)))
        return indices, FEATURE_NAMES.copy(), True
    if model == "crackle_cox_mechanics_ranker_v1":
        indices = list(range(len(RANKER_FEATURE_NAMES)))
        return indices, RANKER_FEATURE_NAMES.copy(), True
    if model == "crackle_cox_fast_ranker_v1":
        indices = list(range(len(HYBRID_FEATURE_NAMES)))
        return indices, HYBRID_FEATURE_NAMES.copy(), True
    raise ValueError(f"unknown trainable model: {model}")


def _features_for_model(
    model: str,
    labels: dict[str, np.ndarray],
    sample: dict[str, Any],
    step: int,
    indices: np.ndarray,
    history: np.ndarray,
    include_history: bool,
) -> np.ndarray:
    if model == "crackle_cox_mechanics_ranker_v1":
        return ranker_feature_matrix(labels, sample, step, indices, history_trigger=history, include_history=include_history)
    if model in {"cox_mechanics_hawkes_hybrid_v1", "crackle_cox_fast_ranker_v1"}:
        return hybrid_feature_matrix(labels, step, indices, history_trigger=history, include_history=include_history)
    return feature_matrix(labels, step, indices, history_trigger=history, include_history=include_history)


def build_training_matrix(
    data_dir: Path,
    *,
    split: str,
    model: str,
    seed: int,
    neg_ratio: int,
    max_neg_per_step: int,
    max_pairwise_per_step: int = 0,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    manifest_path = _source_manifest(data_dir)
    manifest = load_json(manifest_path)
    samples = ok_samples(manifest, split=split)
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    if not samples:
        raise ValueError(f"no samples found for split {split!r}")
    selected, _, include_history = _selected_features(model)
    rng = np.random.default_rng(seed)
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    pairwise_chunks: list[np.ndarray] = []
    pos_total = 0
    neg_total = 0
    pair_total = 0
    for sample_index, sample in enumerate(samples):
        labels = load_labels(sample)
        alive = labels["bond_alive"].astype(bool)
        histories = hawkes_history_by_step(labels) if include_history else [np.zeros((alive.shape[1],), dtype=np.float64) for _ in range(alive.shape[0] - 1)]
        for step in range(alive.shape[0] - 1):
            current_alive = alive[step]
            positives = current_alive & ~alive[step + 1]
            pos_idx = np.flatnonzero(positives)
            alive_idx = np.flatnonzero(current_alive)
            neg_candidates = alive_idx[~positives[alive_idx]]
            if pos_idx.size:
                neg_count = min(neg_candidates.size, max(int(neg_ratio) * pos_idx.size, min(int(max_neg_per_step), neg_candidates.size)))
            else:
                neg_count = min(neg_candidates.size, max(16, int(max_neg_per_step) // 12))
            if neg_count > 0:
                neg_idx = rng.choice(neg_candidates, size=neg_count, replace=False)
                indices = np.concatenate([pos_idx, neg_idx])
            else:
                indices = pos_idx
            if indices.size == 0:
                continue
            x_full = _features_for_model(model, labels, sample, step, indices, histories[step], include_history)
            y = positives[indices].astype(np.float64)
            x_selected = x_full[:, selected]
            chunks_x.append(x_selected)
            chunks_y.append(y)
            pos_total += int(np.count_nonzero(y))
            neg_total += int(y.size - np.count_nonzero(y))
            if model in {"crackle_cox_mechanics_ranker_v1", "crackle_cox_fast_ranker_v1"} and max_pairwise_per_step > 0 and np.any(y > 0.5) and np.any(y <= 0.5):
                local_pos = np.flatnonzero(y > 0.5)
                local_neg = np.flatnonzero(y <= 0.5)
                pair_count = min(int(max_pairwise_per_step), int(local_pos.size * local_neg.size))
                pos_pick = rng.choice(local_pos, size=pair_count, replace=True)
                neg_pick = rng.choice(local_neg, size=pair_count, replace=True)
                pairwise_chunks.append(x_selected[pos_pick] - x_selected[neg_pick])
                pair_total += int(pair_count)
    if not chunks_x:
        raise ValueError("training matrix is empty")
    x = np.concatenate(chunks_x, axis=0)
    y = np.concatenate(chunks_y, axis=0)
    stats = {
        "source_manifest": str(manifest_path),
        "split": split,
        "num_samples": len(samples),
        "num_rows": int(y.size),
        "positive_rows": pos_total,
        "negative_rows": neg_total,
        "pairwise_rows": pair_total,
        "neg_ratio": int(neg_ratio),
        "max_neg_per_step": int(max_neg_per_step),
        "max_pairwise_per_step": int(max_pairwise_per_step),
        "selected_feature_indices": selected,
    }
    pairwise = np.concatenate(pairwise_chunks, axis=0) if pairwise_chunks else None
    return x, y, pairwise, stats


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir or (args.data / "models" / f"{args.model}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model in DETERMINISTIC_MODELS:
        payload = deterministic_model_payload(args.model)
        payload["seed"] = int(args.seed)
        payload["data_dir"] = str(args.data)
        write_json(out_dir / "model.json", payload)
        write_json(out_dir / "training_stats.json", {"model": args.model, "trained": False, "reason": "handcoded deterministic causal baseline"})
        return {"model_dir": str(out_dir), "model": args.model, "trained": False}
    if args.model not in TRAINABLE_MODELS:
        raise ValueError(f"unknown model {args.model!r}")
    selected, names, _ = _selected_features(args.model)
    x, y, pairwise, stats = build_training_matrix(
        args.data,
        split=args.train_split,
        model=args.model,
        seed=args.seed,
        neg_ratio=args.neg_ratio,
        max_neg_per_step=args.max_neg_per_step,
        max_pairwise_per_step=args.max_pairwise_per_step,
        max_samples=args.max_train_samples,
    )
    if args.model == "gbm_survival_v1":
        hazard = fit_gbm_hazard(
            x,
            y,
            model_name=args.model,
            feature_names=names,
            selected_feature_indices=selected,
            out_dir=out_dir,
            seed=args.seed,
            n_estimators=args.gbm_estimators,
            max_depth=args.gbm_max_depth,
            learning_rate=args.gbm_learning_rate,
            subsample=args.gbm_subsample,
            colsample_bytree=args.gbm_colsample_bytree,
            n_jobs=args.n_jobs,
            device=args.device,
        )
        stats["backend"] = hazard.backend
    elif args.model in {"crackle_cox_mechanics_ranker_v1", "crackle_cox_fast_ranker_v1"}:
        hazard = fit_logistic_hazard_with_pairwise(
            x,
            y,
            pairwise,
            model=args.model,
            feature_names=names,
            selected_feature_indices=selected,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            rank_weight=args.rank_weight,
            focal_gamma=args.focal_gamma,
        )
        stats["objective"] = "class_balanced_bce_plus_pairwise_rank"
        stats["rank_weight"] = float(args.rank_weight)
        stats["focal_gamma"] = float(args.focal_gamma)
    else:
        hazard = fit_logistic_hazard(
            x,
            y,
            model=args.model,
            feature_names=names,
            selected_feature_indices=selected,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
        )
    payload = hazard.to_json()
    payload["seed"] = int(args.seed)
    payload["data_dir"] = str(args.data)
    if args.model in {"crackle_cox_mechanics_ranker_v1", "crackle_cox_fast_ranker_v1"}:
        payload["training_objective"] = stats.get("objective")
        payload["rank_weight"] = float(args.rank_weight)
        payload["focal_gamma"] = float(args.focal_gamma)
        payload["run_variant"] = f"rankw{float(args.rank_weight):.3g}_focal{float(args.focal_gamma):.3g}"
    write_json(out_dir / "model.json", payload)
    write_json(out_dir / "training_stats.json", stats)
    return {"model_dir": str(out_dir), "model": args.model, "trained": True, "training_stats": stats}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Crackle deterministic/survival/Hawkes baselines.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--neg-ratio", type=int, default=16)
    parser.add_argument("--max-neg-per-step", type=int, default=320)
    parser.add_argument("--max-pairwise-per-step", type=int, default=128)
    parser.add_argument("--rank-weight", type=float, default=0.35)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--device", default="cpu", help="Accepted for command compatibility; these baselines are numpy CPU models.")
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--gbm-estimators", type=int, default=360)
    parser.add_argument("--gbm-max-depth", type=int, default=5)
    parser.add_argument("--gbm-learning-rate", type=float, default=0.045)
    parser.add_argument("--gbm-subsample", type=float, default=0.90)
    parser.add_argument("--gbm-colsample-bytree", type=float, default=0.90)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_model(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
