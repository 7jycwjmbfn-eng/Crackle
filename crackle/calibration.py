from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crackle.data.common import logit_clip, safe_div, sigmoid


def weighted_binary_metrics(prob: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None, *, bins: int = 12) -> dict[str, float]:
    p = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if weight is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(weight, dtype=np.float64).reshape(-1)
    total = max(float(np.sum(w)), 1e-12)
    nll = float(np.sum(w * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))) / total)
    brier = float(np.sum(w * ((p - y) ** 2)) / total)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        bin_w = w[mask]
        bin_total = float(np.sum(bin_w))
        if bin_total <= 0.0:
            continue
        ece += bin_total / total * abs(float(np.sum(bin_w * p[mask]) / bin_total) - float(np.sum(bin_w * y[mask]) / bin_total))
    return {"NLL": nll, "Brier": brier, "ECE": float(ece)}


def qq_max_deviation_exp(rescaled_intervals: list[float]) -> float | None:
    vals = np.asarray([v for v in rescaled_intervals if np.isfinite(float(v)) and float(v) >= 0.0], dtype=np.float64)
    if vals.size < 8:
        return None
    vals = np.sort(vals)
    probs = (np.arange(1, vals.size + 1, dtype=np.float64) - 0.5) / vals.size
    expected = -np.log(np.maximum(1.0 - probs, 1e-12))
    scale = max(float(np.percentile(expected, 95)), 1e-12)
    return float(np.max(np.abs(vals - expected)) / scale)


def _fit_intercept_for_logits(logits: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    intercept = 0.0
    for _ in range(80):
        q = sigmoid(z + intercept)
        grad = float(np.sum(w * (q - y)))
        hess = float(np.sum(w * q * (1.0 - q))) + 1e-9
        step = grad / hess
        intercept -= float(np.clip(step, -4.0, 4.0))
        if abs(step) < 1e-8:
            break
    return float(np.clip(intercept, -40.0, 40.0))


def _fit_isotonic(scores: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[list[float], list[float]]:
    order = np.argsort(scores, kind="mergesort")
    x = np.asarray(scores, dtype=np.float64)[order]
    y = np.asarray(target, dtype=np.float64)[order]
    w = np.asarray(weight, dtype=np.float64)[order]
    blocks: list[dict[str, float]] = []
    for score, yy, ww in zip(x, y, w):
        block = {"x_hi": float(score), "w": float(max(ww, 0.0)), "sum": float(max(ww, 0.0) * yy)}
        blocks.append(block)
        while len(blocks) >= 2:
            a = blocks[-2]
            b = blocks[-1]
            avg_a = safe_div(a["sum"], a["w"])
            avg_b = safe_div(b["sum"], b["w"])
            if avg_a <= avg_b:
                break
            merged = {"x_hi": b["x_hi"], "w": a["w"] + b["w"], "sum": a["sum"] + b["sum"]}
            blocks[-2:] = [merged]
    xs = [0.0]
    ys = [safe_div(blocks[0]["sum"], blocks[0]["w"]) if blocks else float(np.mean(y))]
    for block in blocks:
        xs.append(float(block["x_hi"]))
        ys.append(float(np.clip(safe_div(block["sum"], block["w"]), 1e-7, 1.0 - 1e-7)))
    xs.append(1.0)
    ys.append(ys[-1])
    return xs, ys


@dataclass
class IntensityCalibrator:
    method: str
    intercept: float = 0.0
    temperature: float = 1.0
    isotonic_x: list[float] | None = None
    isotonic_y: list[float] | None = None
    time_bin_edges: list[float] | None = None
    time_bin_intercepts: list[float] | None = None

    def apply(self, prob: np.ndarray, time_frac: np.ndarray | float | None = None) -> np.ndarray:
        p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        if self.method == "none":
            return p
        if self.method == "isotonic":
            xs = np.asarray(self.isotonic_x or [0.0, 1.0], dtype=np.float64)
            ys = np.asarray(self.isotonic_y or [0.0, 1.0], dtype=np.float64)
            return np.clip(np.interp(p, xs, ys), 1e-7, 1.0 - 1e-7)
        logits = logit_clip(p)
        if self.method == "temperature":
            return sigmoid(logits / max(float(self.temperature), 1e-6) + float(self.intercept))
        if self.method == "time_bin":
            if time_frac is None:
                return sigmoid(logits + float(self.intercept))
            edges = np.asarray(self.time_bin_edges or [0.0, 1.0], dtype=np.float64)
            offsets = np.asarray(self.time_bin_intercepts or [self.intercept], dtype=np.float64)
            t = np.asarray(time_frac, dtype=np.float64)
            if t.ndim == 0:
                t = np.full_like(p, float(t))
            bins = np.clip(np.searchsorted(edges[1:-1], t, side="right"), 0, offsets.size - 1)
            return sigmoid(logits + offsets[bins])
        return sigmoid(logits + float(self.intercept))

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "intercept": self.intercept,
            "temperature": self.temperature,
            "isotonic_x": self.isotonic_x,
            "isotonic_y": self.isotonic_y,
            "time_bin_edges": self.time_bin_edges,
            "time_bin_intercepts": self.time_bin_intercepts,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "IntensityCalibrator":
        return cls(
            method=str(payload.get("method", "none")),
            intercept=float(payload.get("intercept", 0.0)),
            temperature=float(payload.get("temperature", 1.0)),
            isotonic_x=payload.get("isotonic_x"),
            isotonic_y=payload.get("isotonic_y"),
            time_bin_edges=payload.get("time_bin_edges"),
            time_bin_intercepts=payload.get("time_bin_intercepts"),
        )


def fit_intensity_calibrators(
    prob: np.ndarray,
    target: np.ndarray,
    *,
    time_frac: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    time_bins: int = 8,
) -> tuple[IntensityCalibrator, list[dict[str, float | str]]]:
    p = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.ones_like(p) if weight is None else np.asarray(weight, dtype=np.float64).reshape(-1)
    z = logit_clip(p)
    candidates: list[IntensityCalibrator] = [IntensityCalibrator(method="none")]
    intercept = _fit_intercept_for_logits(z, y, w)
    candidates.append(IntensityCalibrator(method="intercept", intercept=intercept))
    best_temp: IntensityCalibrator | None = None
    best_temp_nll = float("inf")
    for temp in np.geomspace(0.25, 8.0, 29):
        temp_z = z / float(temp)
        temp_intercept = _fit_intercept_for_logits(temp_z, y, w)
        q = sigmoid(temp_z + temp_intercept)
        score = weighted_binary_metrics(q, y, w)["NLL"]
        if score < best_temp_nll:
            best_temp_nll = score
            best_temp = IntensityCalibrator(method="temperature", intercept=temp_intercept, temperature=float(temp))
    if best_temp is not None:
        candidates.append(best_temp)
    xs, ys = _fit_isotonic(p, y, w)
    candidates.append(IntensityCalibrator(method="isotonic", isotonic_x=xs, isotonic_y=ys))
    if time_frac is not None:
        t = np.asarray(time_frac, dtype=np.float64).reshape(-1)
        edges = np.linspace(0.0, 1.0, int(time_bins) + 1)
        offsets = []
        global_intercept = intercept
        for bin_index in range(int(time_bins)):
            lo, hi = edges[bin_index], edges[bin_index + 1]
            mask = (t >= lo) & (t < hi if bin_index + 1 < len(edges) - 1 else t <= hi)
            if np.count_nonzero(mask) < 64 or np.count_nonzero(y[mask] > 0.5) == 0:
                offsets.append(global_intercept)
            else:
                offsets.append(_fit_intercept_for_logits(z[mask], y[mask], w[mask]))
        candidates.append(IntensityCalibrator(method="time_bin", intercept=global_intercept, time_bin_edges=edges.tolist(), time_bin_intercepts=offsets))
    scores: list[dict[str, float | str]] = []
    best = candidates[0]
    best_nll = float("inf")
    for candidate in candidates:
        q = candidate.apply(p, time_frac=time_frac)
        metrics = weighted_binary_metrics(q, y, w)
        row: dict[str, float | str] = {"method": candidate.method, **metrics}
        scores.append(row)
        if float(metrics["NLL"]) < best_nll:
            best_nll = float(metrics["NLL"])
            best = candidate
    return best, scores
