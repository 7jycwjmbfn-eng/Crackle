from __future__ import annotations

import math

import numpy as np


def binary_nll(prob: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(target, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def calibration_ece(prob: np.ndarray, target: np.ndarray, bins: int = 12) -> float:
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(p.size, 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        ece += float(np.count_nonzero(mask)) / total * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def topk_precision_recall(scores: np.ndarray, target: np.ndarray, k: int) -> tuple[float, float]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(target, dtype=bool)
    positives = int(np.count_nonzero(y))
    if positives == 0:
        return 1.0, 1.0
    k = max(1, min(int(k), s.size))
    top = np.argpartition(s, -k)[-k:]
    hits = int(np.count_nonzero(y[top]))
    return float(hits) / float(k), float(hits) / float(positives)


def ks_exp_pvalue(rescaled_intervals: list[float]) -> tuple[float | None, float | None]:
    vals = np.asarray([v for v in rescaled_intervals if math.isfinite(float(v)) and float(v) >= 0.0], dtype=np.float64)
    if vals.size < 8:
        return None, None
    vals = np.sort(vals)
    cdf = 1.0 - np.exp(-vals)
    empirical = np.arange(1, vals.size + 1, dtype=np.float64) / vals.size
    d_plus = np.max(empirical - cdf)
    d_minus = np.max(cdf - (np.arange(vals.size, dtype=np.float64) / vals.size))
    d = float(max(d_plus, d_minus))
    # Kolmogorov asymptotic approximation.
    z = (math.sqrt(vals.size) + 0.12 + 0.11 / math.sqrt(vals.size)) * d
    p = 2.0 * sum(((-1) ** (k - 1)) * math.exp(-2.0 * k * k * z * z) for k in range(1, 80))
    return d, float(max(0.0, min(1.0, p)))
