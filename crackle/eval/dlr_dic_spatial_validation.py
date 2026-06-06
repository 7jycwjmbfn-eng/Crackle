from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage

from crackle.eval.crackmnist_xgb_baseline import _features_for_images, _sample_training_rows


NMAP_RE = re.compile(r"Nodemaps/S_160_4\.7_AllDataPoints_(\d+)\.txt$")
GT_RE = re.compile(r"GroundTruth/S_160_4\.7_AllDataPoints_(\d+)_(left|right)\.txt$")


def _parse_header_value(lines: list[str], name: str) -> float | None:
    prefix = f"# {name}:"
    for line in lines:
        if line.startswith(prefix):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _read_nodemap(zip_file: ZipFile, name: str, image_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    text = zip_file.read(name).decode("utf-8", errors="replace").splitlines()
    rows = []
    for line in text:
        if not line.strip() or line.startswith("#"):
            continue
        rows.append([float(x.strip()) for x in line.split(";")])
    arr = np.asarray(rows, dtype=np.float32)
    x = arr[:, 1]
    y = arr[:, 2]
    u = arr[:, 4]
    v = arr[:, 5]
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    px = np.rint((x - x_min) / max(x_max - x_min, 1e-6) * (image_size - 1)).astype(np.int32)
    py = np.rint((y_max - y) / max(y_max - y_min, 1e-6) * (image_size - 1)).astype(np.int32)
    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)
    sums = np.zeros((2, image_size, image_size), dtype=np.float32)
    counts = np.zeros((image_size, image_size), dtype=np.float32)
    np.add.at(sums[0], (py, px), u)
    np.add.at(sums[1], (py, px), v)
    np.add.at(counts, (py, px), 1.0)
    known = counts > 0
    sums[:, known] /= counts[known][None, :]
    if np.any(~known):
        _, nearest = ndimage.distance_transform_edt(~known, return_indices=True)
        for channel in range(2):
            sums[channel] = sums[channel, nearest[0], nearest[1]]
    meta = {
        "force_n": _parse_header_value(text, "Force [N]"),
        "potential_n": _parse_header_value(text, "Potential [N]"),
        "crack_length_mm": _parse_header_value(text, "Crack length [mm]"),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "nodemap_points": int(arr.shape[0]),
    }
    return sums.astype(np.float32), meta


def _resize_channels(image: np.ndarray, out_size: int) -> np.ndarray:
    if image.shape[-1] == out_size:
        return image.astype(np.float32)
    out = []
    for channel in image:
        pil = Image.fromarray(channel.astype(np.float32), mode="F")
        out.append(np.asarray(pil.resize((out_size, out_size), Image.Resampling.BILINEAR), dtype=np.float32))
    return np.stack(out, axis=0)


def _maxpool_mask(mask: np.ndarray, out_size: int) -> np.ndarray:
    if mask.shape[0] == out_size:
        return mask.astype(bool)
    factor = mask.shape[0] // out_size
    trimmed = mask[: out_size * factor, : out_size * factor]
    return trimmed.reshape(out_size, factor, out_size, factor).max(axis=(1, 3)).astype(bool)


def _tip_mask_from_gt(gt: np.ndarray, out_size: int) -> tuple[np.ndarray, tuple[float, float]]:
    ys, xs = np.nonzero(gt == 2)
    if xs.size == 0:
        ys, xs = np.nonzero(gt > 0)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    sx = cx * (out_size - 1) / (gt.shape[1] - 1)
    sy = cy * (out_size - 1) / (gt.shape[0] - 1)
    mask = np.zeros((out_size, out_size), dtype=bool)
    ix = int(round(sx))
    iy = int(round(sy))
    for yy in range(max(0, iy - 1), min(out_size, iy + 2)):
        for xx in range(max(0, ix - 1), min(out_size, ix + 2)):
            mask[yy, xx] = True
    return mask, (cx, cy)


def build_dlr_cache(zip_path: Path, cache_path: Path, *, model_pixels: int) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {
            "images": data["images"],
            "tip_masks": data["tip_masks"].astype(bool),
            "path_masks": data["path_masks"].astype(bool),
            "stage": data["stage"],
            "side": data["side"],
            "tip_xy_256": data["tip_xy_256"],
            "force_n": data["force_n"],
            "crack_length_mm": data["crack_length_mm"],
            "cache_path": str(cache_path),
            "from_cache": True,
        }
    with ZipFile(zip_path) as z:
        names = z.namelist()
        nodemaps: dict[int, str] = {}
        gt_names: dict[tuple[int, str], str] = {}
        for name in names:
            nmap = NMAP_RE.search(name)
            if nmap:
                nodemaps[int(nmap.group(1))] = name
            gt = GT_RE.search(name)
            if gt:
                gt_names[(int(gt.group(1)), gt.group(2))] = name
        images = []
        tip_masks = []
        path_masks = []
        stages = []
        sides = []
        tips = []
        forces = []
        crack_lengths = []
        for stage in sorted(nodemaps):
            if (stage, "left") not in gt_names and (stage, "right") not in gt_names:
                continue
            uv_256, meta = _read_nodemap(z, nodemaps[stage], 256)
            uv_model = _resize_channels(uv_256, model_pixels)
            for side in ["left", "right"]:
                gt_name = gt_names.get((stage, side))
                if gt_name is None:
                    continue
                gt = np.loadtxt(io.BytesIO(z.read(gt_name))).astype(np.int16)
                tip_mask, tip_xy = _tip_mask_from_gt(gt, model_pixels)
                path_mask = _maxpool_mask(gt > 0, model_pixels)
                images.append(uv_model)
                tip_masks.append(tip_mask)
                path_masks.append(path_mask)
                stages.append(stage)
                sides.append(side)
                tips.append(tip_xy)
                forces.append(float(meta["force_n"]) if meta["force_n"] is not None else np.nan)
                crack_lengths.append(float(meta["crack_length_mm"]) if meta["crack_length_mm"] is not None else np.nan)
    payload = {
        "images": np.asarray(images, dtype=np.float32),
        "tip_masks": np.asarray(tip_masks, dtype=np.uint8),
        "path_masks": np.asarray(path_masks, dtype=np.uint8),
        "stage": np.asarray(stages, dtype=np.int32),
        "side": np.asarray(sides, dtype="U5"),
        "tip_xy_256": np.asarray(tips, dtype=np.float32),
        "force_n": np.asarray(forces, dtype=np.float32),
        "crack_length_mm": np.asarray(crack_lengths, dtype=np.float32),
    }
    np.savez_compressed(cache_path, **payload)
    payload["cache_path"] = str(cache_path)
    payload["from_cache"] = False
    return payload


def _sample_rows_from_arrays(
    images: np.ndarray,
    masks: np.ndarray,
    indices: np.ndarray,
    *,
    neg_per_pos: int,
    seed: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if max_samples is not None and indices.size > max_samples:
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
    xs = []
    ys = []
    pos_total = 0
    neg_total = 0
    for idx in indices:
        mask = masks[int(idx)].reshape(-1).astype(bool)
        pos = np.flatnonzero(mask)
        neg = np.flatnonzero(~mask)
        if pos.size == 0:
            continue
        neg_take = min(neg.size, max(32, int(pos.size) * int(neg_per_pos)))
        chosen = np.concatenate([pos, rng.choice(neg, size=neg_take, replace=False)])
        feat = _features_for_images(images[int(idx) : int(idx) + 1]).reshape(-1, 10)[chosen]
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
            "samples_used": int(indices.size),
            "train_rows": int(sum(y.size for y in ys)),
            "positive_rows": pos_total,
            "negative_rows": neg_total,
            "neg_per_pos": int(neg_per_pos),
        },
    )


def _train_crackmnist_rows(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h5_path = args.crackmnist_root / f"crackmnist_{args.model_pixels}_L.h5"
    with h5py.File(h5_path, "r") as h5:
        return _sample_training_rows(
            h5,
            max_images=args.crackmnist_train_images,
            neg_per_pos=args.crackmnist_neg_per_pos,
            seed=args.seed,
            batch_images=args.batch_images,
        )


def _fit_xgb(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[Any, str, float]:
    from xgboost import XGBClassifier

    pos = max(float(np.count_nonzero(y == 1)), 1.0)
    neg = max(float(np.count_nonzero(y == 0)), 1.0)
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
    started = time.perf_counter()
    try:
        model.fit(x, y)
    except Exception:
        model.set_params(device="cpu")
        backend = "xgboost_cpu_fallback"
        model.fit(x, y)
    return model, backend, time.perf_counter() - started


def _predict_maps(model: Any, images: np.ndarray, *, batch_images: int) -> np.ndarray:
    probs = []
    h = images.shape[-2]
    w = images.shape[-1]
    for start in range(0, images.shape[0], batch_images):
        stop = min(images.shape[0], start + batch_images)
        feat = _features_for_images(images[start:stop])
        probs.append(model.predict_proba(feat)[:, 1].reshape(stop - start, h, w).astype(np.float32))
    return np.concatenate(probs, axis=0)


def _calibrate_threshold_from_probs(probs: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    flat = probs.reshape(-1)
    truth = masks.reshape(-1).astype(bool)
    thresholds = np.unique(
        np.quantile(
            flat,
            np.concatenate([np.linspace(0.90, 0.99, 30), np.linspace(0.991, 0.9999, 70)]),
        )
    )
    best = {"threshold": 0.5, "pixel_f1": -1.0}
    for threshold in thresholds:
        pred = flat >= float(threshold)
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["pixel_f1"]:
            best = {"threshold": float(threshold), "pixel_precision": precision, "pixel_recall": recall, "pixel_f1": f1}
    return best


def _metrics_for_probs(
    probs: np.ndarray,
    target_masks: np.ndarray,
    tip_xy_256: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    *,
    model_pixels: int,
    side: np.ndarray | None = None,
) -> dict[str, Any]:
    selected_probs = probs[indices]
    truth = target_masks[indices].astype(bool)
    pred = selected_probs >= threshold
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    topk_inters = []
    topk_unions = []
    loc_errors = []
    hit_4px = 0
    frontier_loc_errors = []
    frontier_hit_4px = 0
    scale = 255.0 / float(model_pixels - 1)
    for local, idx in enumerate(indices):
        mask = truth[local]
        count = max(int(np.count_nonzero(mask)), 1)
        flat_prob = selected_probs[local].reshape(-1)
        top = np.argpartition(flat_prob, -count)[-count:]
        pred_top = np.zeros(mask.size, dtype=bool)
        pred_top[top] = True
        pred_top = pred_top.reshape(mask.shape)
        topk_inters.append(int(np.count_nonzero(pred_top & mask)))
        topk_unions.append(int(np.count_nonzero(pred_top | mask)))
        top1 = int(np.argmax(flat_prob))
        py, px = divmod(top1, model_pixels)
        pred_x_256 = float(px) * scale
        pred_y_256 = float(py) * scale
        true_x, true_y = tip_xy_256[int(idx)]
        err = math.sqrt((pred_x_256 - float(true_x)) ** 2 + (pred_y_256 - float(true_y)) ** 2)
        loc_errors.append(err)
        if err <= 4.0:
            hit_4px += 1
        if side is not None:
            pred_mask = pred[local]
            ys, xs = np.nonzero(pred_mask)
            if xs.size > 0:
                if side[int(idx)] == "left":
                    frontier_x = int(np.min(xs))
                else:
                    frontier_x = int(np.max(xs))
                y_at_front = ys[xs == frontier_x]
                frontier_y = int(round(float(np.mean(y_at_front)))) if y_at_front.size else int(py)
                pred_fx_256 = float(frontier_x) * scale
                pred_fy_256 = float(frontier_y) * scale
            else:
                pred_fx_256 = pred_x_256
                pred_fy_256 = pred_y_256
            ferr = math.sqrt((pred_fx_256 - float(true_x)) ** 2 + (pred_fy_256 - float(true_y)) ** 2)
            frontier_loc_errors.append(ferr)
            if ferr <= 4.0:
                frontier_hit_4px += 1
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "topk_iou": float(np.sum(topk_inters) / max(np.sum(topk_unions), 1)),
        "tip_loc_error_px256": float(np.mean(loc_errors)) if loc_errors else None,
        "tip_hit_rate_4px": hit_4px / max(len(indices), 1),
        "frontier_tip_error_px256": float(np.mean(frontier_loc_errors)) if frontier_loc_errors else None,
        "frontier_tip_hit_rate_4px": frontier_hit_4px / max(len(frontier_loc_errors), 1) if frontier_loc_errors else None,
        "samples": int(len(indices)),
    }


def _paris_geometry_baseline(
    path_masks: np.ndarray,
    tip_xy_256: np.ndarray,
    stage: np.ndarray,
    side: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    model_pixels: int,
) -> dict[str, Any]:
    pred_masks = []
    pred_tips = []
    for idx in test_indices:
        s = side[int(idx)]
        train_side = train_indices[side[train_indices] == s]
        xs = tip_xy_256[train_side, 0]
        ys = tip_xy_256[train_side, 1]
        st = stage[train_side].astype(np.float64)
        if train_side.size >= 2:
            px_coef = np.polyfit(st, xs, 1)
            py_coef = np.polyfit(st, ys, 1)
            pred_x = float(np.polyval(px_coef, stage[int(idx)]))
            pred_y = float(np.polyval(py_coef, stage[int(idx)]))
        else:
            pred_x = float(np.mean(xs))
            pred_y = float(np.mean(ys))
        pred_x = float(np.clip(pred_x, 0, 255))
        pred_y = float(np.clip(pred_y, 0, 255))
        pred_tips.append((pred_x, pred_y))
        mask256 = np.zeros((256, 256), dtype=bool)
        edge_x = 255.0 if s == "left" else 0.0
        steps = max(2, int(abs(edge_x - pred_x)) + 1)
        for t in np.linspace(0.0, 1.0, steps):
            x = int(round(edge_x * (1.0 - t) + pred_x * t))
            y = int(round(pred_y))
            for yy in range(max(0, y - 1), min(256, y + 2)):
                if 0 <= x < 256:
                    mask256[yy, x] = True
        pred_masks.append(_maxpool_mask(mask256, model_pixels))
    truth = path_masks[test_indices].astype(bool)
    pred = np.asarray(pred_masks, dtype=bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
    loc = []
    for pxy, idx in zip(pred_tips, test_indices):
        true_x, true_y = tip_xy_256[int(idx)]
        loc.append(math.sqrt((pxy[0] - float(true_x)) ** 2 + (pxy[1] - float(true_y)) ** 2))
    return {
        "pixel_precision": tp / max(tp + fp, 1),
        "pixel_recall": tp / max(tp + fn, 1),
        "pixel_f1": f1,
        "pixel_iou": iou,
        "topk_iou": None,
        "tip_loc_error_px256": float(np.mean(loc)),
        "tip_hit_rate_4px": float(np.mean(np.asarray(loc) <= 4.0)),
        "samples": int(len(test_indices)),
    }


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
    images = cache["images"]
    tip_masks = cache["tip_masks"].astype(bool)
    path_masks = cache["path_masks"].astype(bool)
    stage = cache["stage"]
    side = cache["side"]
    tip_xy = cache["tip_xy_256"]
    unique_stages = np.asarray(sorted(np.unique(stage)), dtype=np.int32)
    split_at = unique_stages[int(math.ceil(len(unique_stages) * args.front_fraction)) - 1]
    train_indices = np.flatnonzero(stage <= split_at)
    test_indices = np.flatnonzero(stage > split_at)

    from xgboost import XGBClassifier  # noqa: F401

    rows: list[dict[str, Any]] = []
    # Cold transfer: train on CrackMNIST only, evaluate DLR after split.
    x_cm, y_cm, cm_stats = _train_crackmnist_rows(args)
    model, backend, train_seconds = _fit_xgb(x_cm, y_cm, args)
    train_probs = _predict_maps(model, images[train_indices], batch_images=args.batch_images)
    train_threshold = _calibrate_threshold_from_probs(train_probs, tip_masks[train_indices])
    test_started = time.perf_counter()
    all_probs = _predict_maps(model, images, batch_images=args.batch_images)
    eval_seconds = time.perf_counter() - test_started
    for target_name, target_masks in [("tip", tip_masks), ("path_tip", path_masks)]:
        metrics = _metrics_for_probs(
            all_probs,
            target_masks,
            tip_xy,
            test_indices,
            float(train_threshold["threshold"]),
            model_pixels=args.model_pixels,
            side=side,
        )
        rows.append(
            {
                "model": "xgb_crackmnist_cold_transfer",
                "target": target_name,
                "backend": backend,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "threshold": train_threshold["threshold"],
                "train_rows": cm_stats["train_rows"],
                "split": "front30_to_back70",
                **metrics,
            }
        )

    # Front-30 DLR domain adaptation with causal split.
    for target_name, target_masks in [("tip", tip_masks), ("path_tip", path_masks)]:
        x_dlr, y_dlr, dlr_stats = _sample_rows_from_arrays(
            images,
            target_masks,
            train_indices,
            neg_per_pos=args.dlr_neg_per_pos,
            seed=args.seed + (11 if target_name == "tip" else 17),
            max_samples=args.max_dlr_train_samples,
        )
        x_train = np.concatenate([x_cm, x_dlr], axis=0) if args.include_crackmnist_in_adapt else x_dlr
        y_train = np.concatenate([y_cm, y_dlr], axis=0) if args.include_crackmnist_in_adapt else y_dlr
        model, backend, train_seconds = _fit_xgb(x_train, y_train, args)
        train_probs = _predict_maps(model, images[train_indices], batch_images=args.batch_images)
        threshold = _calibrate_threshold_from_probs(train_probs, target_masks[train_indices])
        test_started = time.perf_counter()
        all_probs = _predict_maps(model, images, batch_images=args.batch_images)
        eval_seconds = time.perf_counter() - test_started
        metrics = _metrics_for_probs(
            all_probs,
            target_masks,
            tip_xy,
            test_indices,
            float(threshold["threshold"]),
            model_pixels=args.model_pixels,
            side=side,
        )
        rows.append(
            {
                "model": f"xgb_dlr_front30_adapt_{target_name}",
                "target": target_name,
                "backend": backend,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "threshold": threshold["threshold"],
                "train_rows": int(x_train.shape[0]),
                "dlr_train_rows": dlr_stats["train_rows"],
                "split": "front30_to_back70",
                **metrics,
            }
        )

    paris = _paris_geometry_baseline(path_masks, tip_xy, stage, side, train_indices, test_indices, model_pixels=args.model_pixels)
    rows.append(
        {
            "model": "paris_lefm_straight_tip_extrapolator",
            "target": "path_tip",
            "backend": "geometry_cpu",
            "train_seconds": 0.0,
            "eval_seconds": 0.0,
            "threshold": "",
            "train_rows": 0,
            "split": "front30_to_back70",
            **paris,
        }
    )
    metadata = {
        "schema": "dlr_dic_spatial_validation_v1",
        "dlr_zip": str(args.dlr_zip),
        "cache_path": str(args.cache_path),
        "model_pixels": args.model_pixels,
        "num_samples": int(images.shape[0]),
        "num_stages": int(unique_stages.size),
        "front_fraction": args.front_fraction,
        "split_stage_max_train": int(split_at),
        "train_samples": int(train_indices.size),
        "test_samples": int(test_indices.size),
    }
    _write_csv(args.out_dir / "dlr_spatial_metrics.csv", rows)
    (args.out_dir / "dlr_spatial_metrics.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2))
    return {"metadata": metadata, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DLR S_160_4.7 DIC spatial validation for crack-tip/path prediction.")
    parser.add_argument("--dlr-zip", type=Path, required=True)
    parser.add_argument("--crackmnist-root", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--model-pixels", type=int, default=64)
    parser.add_argument("--front-fraction", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--batch-images", type=int, default=256)
    parser.add_argument("--crackmnist-train-images", type=int, default=42056)
    parser.add_argument("--crackmnist-neg-per-pos", type=int, default=80)
    parser.add_argument("--dlr-neg-per-pos", type=int, default=80)
    parser.add_argument("--max-dlr-train-samples", type=int, default=None)
    parser.add_argument("--include-crackmnist-in-adapt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--estimators", type=int, default=700)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--subsample", type=float, default=0.92)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--n-jobs", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
