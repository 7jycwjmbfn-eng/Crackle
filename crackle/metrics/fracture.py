from __future__ import annotations

import numpy as np


def nearest_event_distance(pred_centers: np.ndarray, true_centers: np.ndarray) -> float:
    pred = np.asarray(pred_centers, dtype=np.float64)
    true = np.asarray(true_centers, dtype=np.float64)
    if pred.size == 0 and true.size == 0:
        return 0.0
    if pred.size == 0 or true.size == 0:
        return float("nan")
    d2 = np.sum((pred[:, None, :] - true[None, :, :]) ** 2, axis=2)
    return float(np.mean(np.sqrt(np.min(d2, axis=1))))
