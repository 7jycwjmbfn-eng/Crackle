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


def _score_maps(images: np.ndarray, rule: str) -> np.ndarray:
    imgs = np.asarray(images, dtype=np.float32)
    ux = imgs[:, 0]
    uy = imgs[:, 1]
    mag = np.sqrt(ux * ux + uy * uy)
    gux_y, gux_x = np.gradient(ux, axis=(1, 2))
    guy_y, guy_x = np.gradient(uy, axis=(1, 2))
    grad_mag = np.sqrt(gux_x * gux_x + gux_y * gux_y + guy_x * guy_x + guy_y * guy_y)
    if rule == "dic_displacement_magnitude_rule":
        return mag
    if rule == "dic_gradient_magnitude_rule":
        return grad_mag
    if rule == "dic_energy_gradient_proxy_rule":
        return mag * grad_mag
    if rule == "dic_max_abs_gradient_rule":
        return np.maximum.reduce([np.abs(gux_x), np.abs(gux_y), np.abs(guy_x), np.abs(guy_y)])
    raise ValueError(rule)


def _centroid(mask2d: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask2d)
    if xs.size == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _threshold_stats(scores: np.ndarray, truth: np.ndarray, threshold: float) -> tuple[int, int, int]:
    pred = scores >= threshold
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    return tp, fp, fn


def _find_val_threshold(
    h5: h5py.File,
    rule: str,
    *,
    max_images: int | None,
    batch_images: int,
) -> dict[str, Any]:
    images = h5["val_images"]
    masks = h5["val_masks"]
    n = int(images.shape[0]) if max_images is None else min(int(max_images), int(images.shape[0]))
    score_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    for start in range(0, n, batch_images):
        stop = min(n, start + batch_images)
        score_parts.append(_score_maps(np.asarray(images[start:stop]), rule).reshape(-1))
        truth_parts.append((np.asarray(masks[start:stop]).reshape(-1) > 0))
    scores = np.concatenate(score_parts)
    truth = np.concatenate(truth_parts)
    quantiles = np.linspace(0.985, 0.9998, 80, dtype=np.float64)
    thresholds = np.unique(np.quantile(scores, quantiles))
    best = {"threshold": float(thresholds[0]), "pixel_f1": -1.0}
    for threshold in thresholds:
        tp, fp, fn = _threshold_stats(scores, truth, float(threshold))
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


def _evaluate_split(
    h5: h5py.File,
    rule: str,
    *,
    threshold: float,
    split: str,
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
        scores = _score_maps(np.asarray(images[start:stop]), rule)
        truth = np.asarray(masks[start:stop]) > 0
        pred = scores >= threshold
        tp += int(np.count_nonzero(pred & truth))
        fp += int(np.count_nonzero(pred & ~truth))
        fn += int(np.count_nonzero(~pred & truth))
        for local in range(stop - start):
            true_mask = truth[local]
            count = int(np.count_nonzero(true_mask))
            if count <= 0:
                continue
            valid_masks += 1
            flat_score = scores[local].reshape(-1)
            top = np.argpartition(flat_score, -count)[-count:]
            pred_top = np.zeros_like(true_mask.reshape(-1), dtype=bool)
            pred_top[top] = True
            pred_top = pred_top.reshape(true_mask.shape)
            inter = int(np.count_nonzero(pred_top & true_mask))
            union = int(np.count_nonzero(pred_top | true_mask))
            topk_intersections.append(inter)
            topk_unions.append(union)
            top1 = int(np.argmax(flat_score))
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
        "model": rule,
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.data_root / f"crackmnist_{args.pixels}_{args.size}.h5"
    rows = []
    thresholds = {}
    with h5py.File(h5_path, "r") as h5:
        for rule in args.rules:
            best = _find_val_threshold(
                h5,
                rule,
                max_images=args.max_eval_images,
                batch_images=args.batch_images,
            )
            thresholds[rule] = best
            for split in args.eval_split:
                rows.append(
                    _evaluate_split(
                        h5,
                        rule,
                        threshold=best["threshold"],
                        split=split,
                        max_images=args.max_eval_images,
                        batch_images=args.batch_images,
                    )
                )
    payload = {
        "schema": "crackmnist_dic_rule_baseline_v1",
        "h5": str(h5_path),
        "thresholds": thresholds,
        "rows": rows,
    }
    _write_csv(args.out_dir / "crackmnist_rule_metrics.csv", rows)
    (args.out_dir / "crackmnist_rule_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", dest="out_dir", type=Path, required=True)
    parser.add_argument("--size", default="L")
    parser.add_argument("--pixels", type=int, default=64)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--batch-images", type=int, default=512)
    parser.add_argument("--eval-split", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--rules",
        nargs="+",
        default=[
            "dic_displacement_magnitude_rule",
            "dic_gradient_magnitude_rule",
            "dic_energy_gradient_proxy_rule",
            "dic_max_abs_gradient_rule",
        ],
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
