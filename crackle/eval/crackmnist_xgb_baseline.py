from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _features_for_images(images: np.ndarray) -> np.ndarray:
    imgs = np.asarray(images, dtype=np.float32)
    ux = imgs[:, 0]
    uy = imgs[:, 1]
    mag = np.sqrt(ux * ux + uy * uy)
    gux_y, gux_x = np.gradient(ux, axis=(1, 2))
    guy_y, guy_x = np.gradient(uy, axis=(1, 2))
    grad_mag = np.sqrt(gux_x * gux_x + gux_y * gux_y + guy_x * guy_x + guy_y * guy_y)
    n, h, w = ux.shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    xx = np.broadcast_to(xx[None, :, :], (n, h, w))
    yy = np.broadcast_to(yy[None, :, :], (n, h, w))
    return np.stack([ux, uy, mag, gux_x, gux_y, guy_x, guy_y, grad_mag, xx, yy], axis=-1).reshape(-1, 10)


def _sample_training_rows(
    h5: h5py.File,
    *,
    max_images: int,
    neg_per_pos: int,
    seed: int,
    batch_images: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    images = h5["train_images"]
    masks = h5["train_masks"]
    total_images = min(int(max_images), int(images.shape[0]))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pos_total = 0
    neg_total = 0
    chunk = max(1, int(batch_images))
    for start in range(0, total_images, chunk):
        stop = min(total_images, start + chunk)
        image_batch = np.asarray(images[start:stop])
        mask_batch = np.asarray(masks[start:stop]) > 0
        feat_batch = _features_for_images(image_batch).reshape(stop - start, -1, 10)
        mask_flat = mask_batch.reshape(stop - start, -1)
        for local in range(stop - start):
            mask = mask_flat[local]
            pos = np.flatnonzero(mask)
            neg = np.flatnonzero(~mask)
            if pos.size == 0:
                neg_take = min(64, neg.size)
                chosen = rng.choice(neg, size=neg_take, replace=False)
            else:
                neg_take = min(neg.size, max(32, int(pos.size) * int(neg_per_pos)))
                chosen = np.concatenate([pos, rng.choice(neg, size=neg_take, replace=False)])
            feat = feat_batch[local, chosen]
            label = mask[chosen].astype(np.int32)
            xs.append(feat)
            ys.append(label)
            pos_count = int(np.count_nonzero(label))
            pos_total += pos_count
            neg_total += int(label.size - pos_count)
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.int32),
        {
            "train_images_used": total_images,
            "train_rows": int(sum(y.size for y in ys)),
            "positive_rows": pos_total,
            "negative_rows": neg_total,
            "neg_per_pos": int(neg_per_pos),
        },
    )


def _centroid(mask2d: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask2d)
    if xs.size == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _evaluate_split(
    model: Any,
    h5: h5py.File,
    split: str,
    *,
    threshold: float,
    max_images: int | None,
    batch_images: int,
) -> dict[str, Any]:
    images = h5[f"{split}_images"]
    masks = h5[f"{split}_masks"]
    n = int(images.shape[0]) if max_images is None else min(int(max_images), int(images.shape[0]))
    tp = fp = fn = 0
    topk_intersections = []
    topk_unions = []
    loc_errors = []
    top1_hits = 0
    valid_masks = 0
    started = time.perf_counter()
    for start in range(0, n, batch_images):
        stop = min(n, start + batch_images)
        feat = _features_for_images(np.asarray(images[start:stop]))
        prob = model.predict_proba(feat)[:, 1].reshape(stop - start, images.shape[2], images.shape[3])
        truth = np.asarray(masks[start:stop]) > 0
        pred = prob >= threshold
        tp += int(np.count_nonzero(pred & truth))
        fp += int(np.count_nonzero(pred & ~truth))
        fn += int(np.count_nonzero(~pred & truth))
        for local in range(stop - start):
            true_mask = truth[local]
            count = int(np.count_nonzero(true_mask))
            if count <= 0:
                continue
            valid_masks += 1
            flat_prob = prob[local].reshape(-1)
            top = np.argpartition(flat_prob, -count)[-count:]
            pred_top = np.zeros_like(true_mask.reshape(-1), dtype=bool)
            pred_top[top] = True
            pred_top = pred_top.reshape(true_mask.shape)
            inter = int(np.count_nonzero(pred_top & true_mask))
            union = int(np.count_nonzero(pred_top | true_mask))
            topk_intersections.append(inter)
            topk_unions.append(union)
            top1 = int(np.argmax(flat_prob))
            if true_mask.reshape(-1)[top1]:
                top1_hits += 1
            pred_y, pred_x = divmod(top1, true_mask.shape[1])
            center = _centroid(true_mask)
            if center is not None:
                cx, cy = center
                loc_errors.append(math.sqrt((float(pred_x) - cx) ** 2 + (float(pred_y) - cy) ** 2))
    elapsed = time.perf_counter() - started
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "split": split,
        "images": n,
        "threshold": threshold,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "topk_mask_iou": float(np.sum(topk_intersections) / max(np.sum(topk_unions), 1)),
        "top1_hit_rate": float(top1_hits / max(valid_masks, 1)),
        "tip_loc_error_px": float(np.mean(loc_errors)) if loc_errors else None,
        "eval_seconds": elapsed,
        "images_per_second": n / max(elapsed, 1e-9),
    }


def _calibrate_threshold(
    model: Any,
    h5: h5py.File,
    *,
    max_images: int | None,
    batch_images: int,
) -> dict[str, Any]:
    images = h5["val_images"]
    masks = h5["val_masks"]
    n = int(images.shape[0]) if max_images is None else min(int(max_images), int(images.shape[0]))
    prob_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    for start in range(0, n, batch_images):
        stop = min(n, start + batch_images)
        feat = _features_for_images(np.asarray(images[start:stop]))
        prob_parts.append(model.predict_proba(feat)[:, 1].astype(np.float32))
        truth_parts.append(np.asarray(masks[start:stop]).reshape(-1) > 0)
    probs = np.concatenate(prob_parts)
    truth = np.concatenate(truth_parts)
    quantiles = np.concatenate(
        [
            np.linspace(0.90, 0.99, 40, dtype=np.float64),
            np.linspace(0.991, 0.9999, 60, dtype=np.float64),
        ]
    )
    thresholds = np.unique(np.quantile(probs, quantiles))
    best = {"threshold": 0.5, "pixel_f1": -1.0}
    for threshold in thresholds:
        pred = probs >= float(threshold)
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["pixel_f1"]:
            best = {
                "threshold": float(threshold),
                "pixel_precision": precision,
                "pixel_recall": recall,
                "pixel_f1": f1,
            }
    return best


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from xgboost import XGBClassifier

    args.out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.data_root / f"crackmnist_{args.pixels}_{args.size}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as h5:
        x_train, y_train, stats = _sample_training_rows(
            h5,
            max_images=args.max_train_images,
            neg_per_pos=args.neg_per_pos,
            seed=args.seed,
            batch_images=args.batch_images,
        )
        pos = max(float(np.count_nonzero(y_train == 1)), 1.0)
        neg = max(float(np.count_nonzero(y_train == 0)), 1.0)
        started = time.perf_counter()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=args.device,
            n_estimators=args.estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=1.0,
            reg_lambda=1.5,
            scale_pos_weight=neg / pos,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        backend = f"xgboost_{args.device}"
        try:
            model.fit(x_train, y_train)
        except Exception:
            model.set_params(device="cpu")
            backend = "xgboost_cpu_fallback"
            model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started
        threshold_info = (
            _calibrate_threshold(model, h5, max_images=args.max_eval_images, batch_images=args.batch_images)
            if args.calibrate_threshold
            else {"threshold": 0.5, "pixel_f1": None}
        )
        rows = []
        for split in args.eval_split:
            rows.append(
                _evaluate_split(
                    model,
                    h5,
                    split,
                    threshold=float(threshold_info["threshold"]),
                    max_images=args.max_eval_images,
                    batch_images=args.batch_images,
                )
            )
    payload = {
        "schema": "crackmnist_xgb_baseline_v1",
        "h5": str(h5_path),
        "size": args.size,
        "pixels": args.pixels,
        "backend": backend,
        "train_seconds": train_seconds,
        "threshold_info": threshold_info,
        "training_stats": stats,
        "rows": rows,
    }
    _write_csv(args.out_dir / "crackmnist_xgb_metrics.csv", [{**{"backend": backend, "train_seconds": train_seconds}, **row} for row in rows])
    (args.out_dir / "crackmnist_xgb_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XGBoost baseline for CrackMNIST crack-tip segmentation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--size", default="L")
    parser.add_argument("--pixels", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--max-train-images", type=int, default=42056)
    parser.add_argument("--max-eval-images", type=int)
    parser.add_argument("--neg-per-pos", type=int, default=80)
    parser.add_argument("--estimators", type=int, default=700)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--subsample", type=float, default=0.92)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-images", type=int, default=256)
    parser.add_argument("--eval-split", action="append", default=None)
    parser.add_argument("--calibrate-threshold", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.eval_split is None:
        args.eval_split = ["val", "test"]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
