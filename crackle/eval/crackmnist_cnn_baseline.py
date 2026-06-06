from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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


class GpuSampler:
    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
                util, mem, power, temp = [float(x.strip()) for x in out.split(",")[:4]]
                self.samples.append({"gpu_util": util, "gpu_mem_mb": mem, "gpu_power_w": power, "gpu_temp_c": temp})
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {
                "avg_gpu_util": None,
                "p95_gpu_util": None,
                "gpu_peak_memory_mb": None,
                "avg_gpu_power_w": None,
                "avg_gpu_temp_c": None,
            }
        vals = {k: np.asarray([s[k] for s in self.samples], dtype=np.float64) for k in self.samples[0]}
        return {
            "avg_gpu_util": float(vals["gpu_util"].mean()),
            "p95_gpu_util": float(np.percentile(vals["gpu_util"], 95)),
            "gpu_peak_memory_mb": float(vals["gpu_mem_mb"].max()),
            "avg_gpu_power_w": float(vals["gpu_power_w"].mean()),
            "avg_gpu_temp_c": float(vals["gpu_temp_c"].mean()),
        }


def _load_split(h5: h5py.File, split: str, max_images: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = int(h5[f"{split}_images"].shape[0])
    if max_images is not None:
        n = min(n, int(max_images))
    images = np.asarray(h5[f"{split}_images"][:n], dtype=np.float32)
    masks = (np.asarray(h5[f"{split}_masks"][:n]) > 0).astype(np.float32)
    return images, masks


def _centroid(mask2d: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask2d)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _evaluate_prob(prob: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = prob >= threshold
    truth_bool = truth.astype(bool)
    tp = int(np.count_nonzero(pred & truth_bool))
    fp = int(np.count_nonzero(pred & ~truth_bool))
    fn = int(np.count_nonzero(~pred & truth_bool))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    topk_intersections = []
    topk_unions = []
    loc_errors = []
    top1_hits = 0
    valid_masks = 0
    for i in range(prob.shape[0]):
        true_mask = truth_bool[i]
        count = int(np.count_nonzero(true_mask))
        if count <= 0:
            continue
        valid_masks += 1
        flat_prob = prob[i].reshape(-1)
        top = np.argpartition(flat_prob, -count)[-count:]
        pred_top = np.zeros(true_mask.size, dtype=bool)
        pred_top[top] = True
        pred_top = pred_top.reshape(true_mask.shape)
        topk_intersections.append(int(np.count_nonzero(pred_top & true_mask)))
        topk_unions.append(int(np.count_nonzero(pred_top | true_mask)))
        top1 = int(np.argmax(flat_prob))
        if true_mask.reshape(-1)[top1]:
            top1_hits += 1
        py, px = divmod(top1, true_mask.shape[1])
        center = _centroid(true_mask)
        if center is not None:
            cx, cy = center
            loc_errors.append(math.sqrt((float(px) - cx) ** 2 + (float(py) - cy) ** 2))
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "topk_mask_iou": float(np.sum(topk_intersections) / max(np.sum(topk_unions), 1)),
        "top1_hit_rate": float(top1_hits / max(valid_masks, 1)),
        "tip_loc_error_px": float(np.mean(loc_errors)) if loc_errors else None,
    }


def _calibrate_threshold(prob: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    flat = prob.reshape(-1)
    quantiles = np.concatenate([np.linspace(0.90, 0.99, 40), np.linspace(0.991, 0.99995, 80)])
    thresholds = np.unique(np.quantile(flat, quantiles))
    best: dict[str, Any] = {"threshold": 0.5, "pixel_f1": -1.0}
    for threshold in thresholds:
        metrics = _evaluate_prob(prob, truth, float(threshold))
        if metrics["pixel_f1"] > best["pixel_f1"]:
            best = {"threshold": float(threshold), **metrics}
    return best


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

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.data_root / f"crackmnist_{args.pixels}_{args.size}.h5"
    with h5py.File(h5_path, "r") as h5:
        train_x, train_y = _load_split(h5, "train", args.max_train_images)
        val_x, val_y = _load_split(h5, "val", args.max_eval_images)
        test_x, test_y = _load_split(h5, "test", args.max_eval_images)

    # Preload tensors to the selected device. This is intentional: it avoids the
    # external-SSD / CPU dataloader bottleneck that made earlier runs underuse the GPU.
    train_x_t = torch.from_numpy(train_x).to(device=device, dtype=torch.float32)
    train_y_t = torch.from_numpy(train_y[:, None]).to(device=device, dtype=torch.float32)
    val_x_t = torch.from_numpy(val_x).to(device=device, dtype=torch.float32)
    test_x_t = torch.from_numpy(test_x).to(device=device, dtype=torch.float32)
    model = TinyUNet(base=args.base_channels).to(device)
    pos = float(train_y.sum())
    neg = float(train_y.size - train_y.sum())
    pos_weight = torch.tensor([min(args.max_pos_weight, neg / max(pos, 1.0))], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
        dice = 1.0 - ((2.0 * inter + 1e-6) / denom).mean()
        return bce + args.dice_weight * dice

    history = []
    started = time.perf_counter()
    with GpuSampler(interval=1.0) as sampler:
        for epoch in range(1, args.epochs + 1):
            perm = torch.randperm(train_x_t.shape[0], device=device)
            model.train()
            losses = []
            for start in range(0, train_x_t.shape[0], args.batch_size):
                idx = perm[start : start + args.batch_size]
                xb = train_x_t[idx]
                yb = train_y_t[idx]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    probs = []
                    for start in range(0, val_x_t.shape[0], args.batch_size):
                        logits = model(val_x_t[start : start + args.batch_size])
                        probs.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
                val_prob = np.concatenate(probs, axis=0)
                thresh = _calibrate_threshold(val_prob, val_y)
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **thresh})
            else:
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        train_seconds = time.perf_counter() - started
        gpu_summary = sampler.summary()

    best = max((h for h in history if "threshold" in h), key=lambda h: h["pixel_f1"])
    threshold = float(best["threshold"])
    rows = []
    model.eval()
    for split, x_t, y_np in [("val", val_x_t, val_y), ("test", test_x_t, test_y)]:
        started_eval = time.perf_counter()
        with torch.no_grad():
            probs = []
            for start in range(0, x_t.shape[0], args.batch_size):
                logits = model(x_t[start : start + args.batch_size])
                probs.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
        prob = np.concatenate(probs, axis=0)
        eval_seconds = time.perf_counter() - started_eval
        rows.append(
            {
                "model": "tiny_unet_crackmnist_cuda_v1",
                "split": split,
                "images": int(y_np.shape[0]),
                "threshold": threshold,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "images_per_second": float(y_np.shape[0] / max(eval_seconds, 1e-9)),
                **_evaluate_prob(prob, y_np, threshold),
            }
        )
    model_path = args.out_dir / "tiny_unet_crackmnist_cuda_v1.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "history": history, "threshold": threshold}, model_path)
    payload = {
        "schema": "crackmnist_cnn_baseline_v1",
        "h5": str(h5_path),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "train_seconds": train_seconds,
        "best_threshold": threshold,
        "history": history,
        "gpu": gpu_summary,
        "rows": rows,
        "model_path": str(model_path),
    }
    _write_csv(args.out_dir / "crackmnist_cnn_metrics.csv", [{**gpu_summary, **row} for row in rows])
    (args.out_dir / "crackmnist_cnn_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUDA CNN baseline for CrackMNIST tip masks.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--size", default="L")
    parser.add_argument("--pixels", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
