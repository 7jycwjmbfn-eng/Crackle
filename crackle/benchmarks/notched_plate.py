from __future__ import annotations

import numpy as np


def build_bond_graph(points: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2D horizon bond graph with a simple cell-list broadphase."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"points must have shape [N, 2], got {x.shape}")
    h = float(horizon)
    if h <= 0.0:
        raise ValueError("horizon must be positive")

    cell = np.floor(x / h).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, key_arr in enumerate(cell):
        buckets.setdefault((int(key_arr[0]), int(key_arr[1])), []).append(index)

    bonds: list[tuple[int, int]] = []
    h2 = h * h
    for i, key_arr in enumerate(cell):
        cx, cy = int(key_arr[0]), int(key_arr[1])
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for j in buckets.get((cx + ox, cy + oy), []):
                    if j <= i:
                        continue
                    diff = x[j] - x[i]
                    dist2 = float(diff @ diff)
                    if 1e-18 < dist2 <= h2:
                        bonds.append((i, j))

    if not bonds:
        raise ValueError("horizon produced zero bonds")

    edges = np.asarray(bonds, dtype=np.int64)
    rest = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1).astype(np.float64)
    return edges, rest


def _segment_crosses_notch(a: np.ndarray, b: np.ndarray, notch_length: float, notch_y: float = 0.0) -> bool:
    ya = float(a[1]) - float(notch_y)
    yb = float(b[1]) - float(notch_y)
    if ya == 0.0 and yb == 0.0:
        return min(float(a[0]), float(b[0])) <= float(notch_length)
    if ya * yb > 0.0:
        return False
    denom = yb - ya
    if abs(denom) < 1e-12:
        return False
    alpha = -ya / denom
    if alpha < 0.0 or alpha > 1.0:
        return False
    x_cross = float(a[0] + alpha * (b[0] - a[0]))
    return 0.0 <= x_cross <= float(notch_length)


def initial_notch_alive(points: np.ndarray, bonds: np.ndarray, notch_length: float, notch_y: float = 0.0) -> np.ndarray:
    """Return an initial bond-alive mask with bonds cut by a horizontal notch removed."""
    alive = np.ones((bonds.shape[0],), dtype=bool)
    for index, (i, j) in enumerate(np.asarray(bonds, dtype=np.int64)):
        if _segment_crosses_notch(points[i], points[j], notch_length, notch_y=notch_y):
            alive[index] = False
    return alive

