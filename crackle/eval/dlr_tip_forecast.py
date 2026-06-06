from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from crackle.eval.dlr_dic_spatial_validation import build_dlr_cache
from crackle.eval.crackmnist_xgb_baseline import _features_for_images


def _local_feature_vector(image: np.ndarray, tip_xy_256: np.ndarray) -> np.ndarray:
    h = image.shape[-2]
    scale = (h - 1) / 255.0
    x = int(np.clip(round(float(tip_xy_256[0]) * scale), 0, h - 1))
    y = int(np.clip(round(float(tip_xy_256[1]) * scale), 0, h - 1))
    feat = _features_for_images(image[None]).reshape(h, h, 10)
    patch = feat[max(0, y - 1) : min(h, y + 2), max(0, x - 1) : min(h, x + 2)]
    return np.concatenate([feat[y, x], patch.mean(axis=(0, 1)), patch.max(axis=(0, 1))]).astype(np.float32)


def _build_transitions(cache: dict[str, Any]) -> dict[str, np.ndarray]:
    images = cache["images"]
    stage = cache["stage"].astype(np.int32)
    side = cache["side"]
    tip = cache["tip_xy_256"].astype(np.float32)
    force = cache["force_n"].astype(np.float32)
    crack_len = cache["crack_length_mm"].astype(np.float32)
    rows = []
    for side_name in sorted(set(side.tolist())):
        idx = np.flatnonzero(side == side_name)
        idx = idx[np.argsort(stage[idx])]
        by_stage = {int(stage[i]): int(i) for i in idx}
        sorted_stages = sorted(by_stage)
        for pos in range(1, len(sorted_stages) - 1):
            prev_i = by_stage[sorted_stages[pos - 1]]
            cur_i = by_stage[sorted_stages[pos]]
            next_i = by_stage[sorted_stages[pos + 1]]
            cur_tip = tip[cur_i]
            prev_tip = tip[prev_i]
            next_tip = tip[next_i]
            local = _local_feature_vector(images[cur_i], cur_tip)
            side_sign = -1.0 if side_name == "left" else 1.0
            base = np.asarray(
                [
                    float(stage[cur_i]),
                    side_sign,
                    float(cur_tip[0]),
                    float(cur_tip[1]),
                    float(cur_tip[0] - prev_tip[0]),
                    float(cur_tip[1] - prev_tip[1]),
                    float(force[cur_i]) if np.isfinite(force[cur_i]) else 0.0,
                    float(crack_len[cur_i]) if np.isfinite(crack_len[cur_i]) else 0.0,
                ],
                dtype=np.float32,
            )
            rows.append(
                {
                    "index": cur_i,
                    "next_index": next_i,
                    "stage": int(stage[cur_i]),
                    "next_stage": int(stage[next_i]),
                    "side": side_name,
                    "x": np.concatenate([base, local]),
                    "current_tip": cur_tip,
                    "prev_tip": prev_tip,
                    "next_tip": next_tip,
                    "delta": next_tip - cur_tip,
                    "last_delta": cur_tip - prev_tip,
                }
            )
    return {
        "x": np.stack([r["x"] for r in rows]).astype(np.float32),
        "current_tip": np.stack([r["current_tip"] for r in rows]).astype(np.float32),
        "next_tip": np.stack([r["next_tip"] for r in rows]).astype(np.float32),
        "delta": np.stack([r["delta"] for r in rows]).astype(np.float32),
        "last_delta": np.stack([r["last_delta"] for r in rows]).astype(np.float32),
        "stage": np.asarray([r["stage"] for r in rows], dtype=np.int32),
        "next_stage": np.asarray([r["next_stage"] for r in rows], dtype=np.int32),
        "side": np.asarray([r["side"] for r in rows], dtype="U5"),
    }


def _metrics(pred: np.ndarray, truth: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    err = np.sqrt(np.sum((pred[indices] - truth[indices]) ** 2, axis=1))
    return {
        "samples": int(indices.size),
        "mean_tip_error_px256": float(np.mean(err)),
        "median_tip_error_px256": float(np.median(err)),
        "p90_tip_error_px256": float(np.percentile(err, 90)),
        "hit_rate_4px": float(np.mean(err <= 4.0)),
        "hit_rate_8px": float(np.mean(err <= 8.0)),
    }


def _fit_linear_front30(transitions: dict[str, np.ndarray], train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(transitions["next_tip"])
    for side_name in sorted(set(transitions["side"].tolist())):
        train_side = train_idx[transitions["side"][train_idx] == side_name]
        all_side = np.flatnonzero(transitions["side"] == side_name)
        for coord in [0, 1]:
            coeff = np.polyfit(transitions["next_stage"][train_side].astype(np.float64), transitions["next_tip"][train_side, coord], 1)
            pred[all_side, coord] = np.polyval(coeff, transitions["next_stage"][all_side])
    pred[:, 0] = np.clip(pred[:, 0], 0, 255)
    pred[:, 1] = np.clip(pred[:, 1], 0, 255)
    return pred


def _fit_xgb_delta(transitions: dict[str, np.ndarray], train_idx: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str, float]:
    from xgboost import XGBRegressor

    x_train = transitions["x"][train_idx]
    y_train = transitions["delta"][train_idx]
    preds = []
    backend = f"xgboost_{args.device}"
    started = time.perf_counter()
    for coord in [0, 1]:
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device=args.device,
            n_estimators=args.estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=1.5,
            n_jobs=args.n_jobs,
            random_state=args.seed + coord,
        )
        try:
            model.fit(x_train, y_train[:, coord])
        except Exception:
            model.set_params(device="cpu")
            backend = "xgboost_cpu_fallback"
            model.fit(x_train, y_train[:, coord])
        preds.append(model.predict(transitions["x"]).astype(np.float32))
    elapsed = time.perf_counter() - started
    delta = np.stack(preds, axis=1)
    pred_tip = transitions["current_tip"] + delta
    pred_tip[:, 0] = np.clip(pred_tip[:, 0], 0, 255)
    pred_tip[:, 1] = np.clip(pred_tip[:, 1], 0, 255)
    return pred_tip, backend, elapsed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = build_dlr_cache(args.dlr_zip, args.cache_path, model_pixels=args.model_pixels)
    transitions = _build_transitions(cache)
    stages = np.asarray(sorted(np.unique(transitions["stage"])), dtype=np.int32)
    split_stage = stages[int(math.ceil(stages.size * args.front_fraction)) - 1]
    train_idx = np.flatnonzero(transitions["stage"] <= split_stage)
    test_idx = np.flatnonzero(transitions["stage"] > split_stage)
    truth = transitions["next_tip"]

    rows: list[dict[str, Any]] = []
    candidates = {
        "persistence_current_tip": transitions["current_tip"],
        "last_delta_extrapolation": transitions["current_tip"] + transitions["last_delta"],
        "front30_linear_tip_fit": _fit_linear_front30(transitions, train_idx, test_idx),
    }
    for name, pred in candidates.items():
        pred = pred.astype(np.float32)
        pred[:, 0] = np.clip(pred[:, 0], 0, 255)
        pred[:, 1] = np.clip(pred[:, 1], 0, 255)
        rows.append({"model": name, "backend": "geometry_cpu", "train_seconds": 0.0, **_metrics(pred, truth, test_idx)})

    pred, backend, train_seconds = _fit_xgb_delta(transitions, train_idx, args)
    rows.append({"model": "xgb_temporal_dic_next_tip_v1", "backend": backend, "train_seconds": train_seconds, **_metrics(pred, truth, test_idx)})

    for side_name in sorted(set(transitions["side"].tolist())):
        side_test = test_idx[transitions["side"][test_idx] == side_name]
        pred, backend, train_seconds = _fit_xgb_delta(transitions, train_idx[transitions["side"][train_idx] == side_name], args)
        rows.append(
            {
                "model": f"xgb_temporal_dic_next_tip_v1_{side_name}",
                "backend": backend,
                "train_seconds": train_seconds,
                "side": side_name,
                **_metrics(pred, truth, side_test),
            }
        )

    metadata = {
        "schema": "dlr_tip_forecast_v1",
        "num_transitions": int(transitions["stage"].size),
        "train_transitions": int(train_idx.size),
        "test_transitions": int(test_idx.size),
        "split_stage": int(split_stage),
        "front_fraction": args.front_fraction,
        "model_pixels": args.model_pixels,
        "causal_features": [
            "current_tip_xy",
            "previous_tip_delta",
            "current_force",
            "current_crack_length",
            "current_local_dic_uv_gradient_features",
        ],
        "target": "next_stage_tip_xy",
    }
    _write_csv(args.out_dir / "dlr_tip_forecast_metrics.csv", rows)
    (args.out_dir / "dlr_tip_forecast_metrics.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2))
    return {"metadata": metadata, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-step DLR DIC crack-tip forecasting.")
    parser.add_argument("--dlr-zip", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--model-pixels", type=int, default=64)
    parser.add_argument("--front-fraction", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.95)
    parser.add_argument("--colsample-bytree", type=float, default=0.95)
    parser.add_argument("--n-jobs", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
