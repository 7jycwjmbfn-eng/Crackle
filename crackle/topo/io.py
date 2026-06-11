"""Loaders bridging hetero_pinning outputs to (T, ny, nx) field movies.

hetero_pinning saves case_arrays.npz with:
- points            (N, 2)  meshgrid(xy indexing) row-major flatten -> row=y
- gc                (N,)
- reference_damage  (T+1, N)
- a_rollout_damage  (T+1, N)
- <model>_damage / <model>_u for each one-shot field model

Grid shape is inferred from points, so no nx/ny arguments are needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def infer_grid_shape(points: np.ndarray) -> tuple[int, int]:
    pts = np.asarray(points, dtype=np.float64)
    nx = int(np.unique(np.round(pts[:, 0], 9)).size)
    ny = int(np.unique(np.round(pts[:, 1], 9)).size)
    if nx * ny != pts.shape[0]:
        raise ValueError(
            f"points ({pts.shape[0]}) is not a tensor grid of {ny}x{nx}"
        )
    return ny, nx


def flat_to_fields(flat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(T, N) node damage -> (T, ny, nx) movies using the meshgrid layout."""
    ny, nx = infer_grid_shape(points)
    arr = np.asarray(flat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.reshape(arr.shape[0], ny, nx)


def load_case_npz(
    path: str | Path, key: str = "reference_damage"
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Returns (fields (T,ny,nx), points (N,2), raw npz dict)."""
    data = dict(np.load(Path(path)))
    if "points" not in data:
        raise KeyError(f"{path} has no 'points'; keys: {sorted(data)}")
    if key not in data:
        raise KeyError(f"{path} has no '{key}'; keys: {sorted(data)}")
    fields = flat_to_fields(data[key], data["points"])
    return fields, data["points"], data
