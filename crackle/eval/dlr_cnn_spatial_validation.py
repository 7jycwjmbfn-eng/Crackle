from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from crackle.eval.crackmnist_cnn_baseline import GpuSampler, _calibrate_threshold, _evaluate_prob
from crackle.eval.dlr_dic_spatial_validation import build_dlr_cache, _paris_geometry_baseline


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


def _tip_metrics_px256(prob: np.ndarray, tip_xy_256: np.ndarray, indices: np.ndarray, threshold: float, side: np.ndarray | None = None) -> dict[str, Any]:
    h, w = prob.shape[-2:]
    scale_x = 255.0 / float(w - 1)
    scale_y = 255.0 / float(h - 1)
    argmax_err = []
    frontier_err = []
    argmax_hit4 = 0
    frontier_hit4 = 0
    pred_mask = prob >= threshold
    for idx in indices:
        flat = prob[int(idx)].reshape(-1)
        py, px = divmod(int(np.argmax(flat)), w)
        true_x, true_y = tip_xy_256[int(idx)]
        err = math.sqrt((px * scale_x - float(true_x)) ** 2 + (py * scale_y - float(true_y)) ** 2)
        argmax_err.append(err)
        argmax_hit4 += int(err <= 4.0)
        if side is not None:
            ys, xs = np.nonzero(pred_mask[int(idx)])
            if xs.size:
                fx = int(xs.min()) if side[int(idx)] == "left" else int(xs.max())
                fy_vals = ys[xs == fx]
                fy = int(round(float(fy_vals.mean()))) if fy_vals.size else int(py)
            else:
                fx, fy = int(px), int(py)
            ferr = math.sqrt((fx * scale_x - float(true_x)) ** 2 + (fy * scale_y - float(true_y)) ** 2)
            frontier_err.append(ferr)
            frontier_hit4 += int(ferr <= 4.0)
    out = {
        "tip_loc_error_px256": float(np.mean(argmax_err)),
        "tip_hit_rate_4px": float(argmax_hit4 / max(len(argmax_err), 1)),
    }
    if frontier_err:
        out.update(
            {
                "frontier_tip_error_px256": float(np.mean(frontier_err)),
                "frontier_tip_hit_rate_4px": float(frontier_hit4 / max(len(frontier_err), 1)),
            }
        )
    return out


def _metrics(prob: np.ndarray, masks: np.ndarray, tip_xy_256: np.ndarray, indices: np.ndarray, threshold: float, side: np.ndarray) -> dict[str, Any]:
    subset_prob = prob[indices]
    subset_masks = masks[indices].astype(np.float32)
    out = _evaluate_prob(subset_prob, subset_masks, threshold)
    out.update(_tip_metrics_px256(prob, tip_xy_256, indices, threshold, side=side))
    out["samples"] = int(len(indices))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TinyUNet(nn.Module):
        def __init__(self, base: int = 32) -> None:
            super().__init__()
            self.enc1 = ConvBlock(2, base)
            self.enc2 = ConvBlock(base, base * 2)
            self.enc3 = ConvBlock(base * 2, base * 4)
            self.mid = ConvBlock(base * 4, base * 4)
            self.dec2 = ConvBlock(base * 6, base * 2)
            self.dec1 = ConvBlock(base * 3, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, 2))
            e3 = self.enc3(F.max_pool2d(e2, 2))
            mid = self.mid(e3)
            d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.head(d1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    cache = build_dlr_cache(args.dlr_zip, args.cache_path, model_pixels=args.model_pixels)
    images = cache["images"].astype(np.float32)
    tip_masks = cache["tip_masks"].astype(np.float32)
    path_masks = cache["path_masks"].astype(np.float32)
    stage = cache["stage"]
    side = cache["side"]
    tip_xy = cache["tip_xy_256"]
    unique_stages = np.asarray(sorted(np.unique(stage)), dtype=np.int32)
    split_stage = unique_stages[int(math.ceil(len(unique_stages) * args.front_fraction)) - 1]
    train_idx = np.flatnonzero(stage <= split_stage)
    test_idx = np.flatnonzero(stage > split_stage)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rows: list[dict[str, Any]] = []
    gpu_summaries: list[dict[str, Any]] = []

    def new_model() -> Any:
        model = TinyUNet(base=args.base_channels).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        return model

    x_t = torch.from_numpy(images).to(device=device, dtype=torch.float32)

    def predict(model: Any) -> np.ndarray:
        model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, x_t.shape[0], args.batch_size):
                probs.append(torch.sigmoid(model(x_t[start : start + args.batch_size])).squeeze(1).float().cpu().numpy())
        return np.concatenate(probs, axis=0)

    # Cold transfer: DLR front 30% is used only to calibrate threshold.
    model = new_model()
    started = time.perf_counter()
    prob = predict(model)
    eval_seconds = time.perf_counter() - started
    for target, masks in [("tip", tip_masks), ("path_tip", path_masks)]:
        threshold = _calibrate_threshold(prob[train_idx], masks[train_idx])["threshold"]
        rows.append(
            {
                "model": "tiny_unet_crackmnist_cold_transfer",
                "target": target,
                "backend": str(device),
                "train_seconds": 0.0,
                "eval_seconds": eval_seconds,
                "threshold": threshold,
                "split": "front30_to_back70",
                **_metrics(prob, masks, tip_xy, test_idx, float(threshold), side),
            }
        )

    def finetune(target: str, masks_np: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
        model = new_model()
        y = torch.from_numpy(masks_np[:, None]).to(device=device, dtype=torch.float32)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
        pos = float(masks_np[train_idx].sum())
        neg = float(masks_np[train_idx].size - masks_np[train_idx].sum())
        pos_weight = torch.tensor([min(args.max_pos_weight, neg / max(pos, 1.0))], device=device)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

        def loss_fn(logits: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
            bce = F.binary_cross_entropy_with_logits(logits, target_t, pos_weight=pos_weight)
            prob_t = torch.sigmoid(logits)
            inter = (prob_t * target_t).sum(dim=(1, 2, 3))
            denom = prob_t.sum(dim=(1, 2, 3)) + target_t.sum(dim=(1, 2, 3)) + 1e-6
            dice = 1.0 - ((2.0 * inter + 1e-6) / denom).mean()
            return bce + args.dice_weight * dice

        started_train = time.perf_counter()
        with GpuSampler(interval=0.5) as sampler:
            for _epoch in range(args.epochs):
                perm = train_idx_t[torch.randperm(train_idx_t.numel(), device=device)]
                model.train()
                for start in range(0, perm.numel(), args.batch_size):
                    idx = perm[start : start + args.batch_size]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                        logits = model(x_t[idx])
                        loss = loss_fn(logits, y[idx])
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
            train_seconds = time.perf_counter() - started_train
            gpu_summary = sampler.summary()
        return predict(model), train_seconds, gpu_summary

    for target, masks in [("tip", tip_masks), ("path_tip", path_masks)]:
        prob, train_seconds, gpu_summary = finetune(target, masks)
        gpu_summaries.append(gpu_summary)
        threshold = _calibrate_threshold(prob[train_idx], masks[train_idx])["threshold"]
        rows.append(
            {
                "model": f"tiny_unet_dlr_front30_finetune_{target}",
                "target": target,
                "backend": str(device),
                "train_seconds": train_seconds,
                "eval_seconds": 0.0,
                "threshold": threshold,
                "split": "front30_to_back70",
                **gpu_summary,
                **_metrics(prob, masks, tip_xy, test_idx, float(threshold), side),
            }
        )

    paris = _paris_geometry_baseline(path_masks.astype(bool), tip_xy, stage, side, train_idx, test_idx, model_pixels=args.model_pixels)
    rows.append(
        {
            "model": "paris_lefm_straight_tip_extrapolator",
            "target": "path_tip",
            "backend": "geometry_cpu",
            "train_seconds": 0.0,
            "eval_seconds": 0.0,
            "threshold": "",
            "split": "front30_to_back70",
            **paris,
        }
    )

    metadata = {
        "schema": "dlr_cnn_spatial_validation_v1",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "model_pixels": args.model_pixels,
        "num_samples": int(images.shape[0]),
        "train_samples": int(train_idx.size),
        "test_samples": int(test_idx.size),
        "split_stage": int(split_stage),
    }
    _write_csv(args.out_dir / "dlr_cnn_spatial_metrics.csv", rows)
    (args.out_dir / "dlr_cnn_spatial_metrics.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False, indent=2))
    return {"metadata": metadata, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DLR spatial validation with CrackMNIST CNN checkpoint.")
    parser.add_argument("--dlr-zip", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--model-pixels", type=int, default=64)
    parser.add_argument("--front-fraction", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--max-pos-weight", type=float, default=120.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
