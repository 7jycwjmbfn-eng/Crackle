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


# --- real-data loaders: ordered image/mask sequence -> (T, ny, nx) ----------

import re


def natural_key(path: "str | Path") -> list:
    """Sort key that orders frame_2 before frame_10 (numeric-aware)."""
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_mask_sequence(
    paths: "list[str | Path]",
    *,
    threshold: float | None = 0.5,
    downscale: int = 1,
    invert: bool = False,
) -> np.ndarray:
    """Load an ORDERED list of crack-mask/image files into a (T, ny, nx) movie.

    Each file is read as grayscale in [0, 1]. With threshold set, the field is
    binarized (crack = 1) — the right input for cubical persistence on a
    segmentation mask; pass threshold=None to keep the continuous grayscale
    (e.g. a DIC strain field). downscale>1 block-averages to cut the grid
    (high-res masks can be thousands of px; topology is scale-robust, see the
    res96 probe). invert flips foreground/background. All frames must share a
    shape; the caller orders `paths` (use natural_key for frame-numbered files).

    No dependency on the simulator side, so it imports cleanly in the archive.
    """
    from PIL import Image

    frames: list[np.ndarray] = []
    shape0: tuple[int, int] | None = None
    for p in paths:
        img = Image.open(Path(p)).convert("L")
        a = np.asarray(img, dtype=np.float64) / 255.0
        if invert:
            a = 1.0 - a
        if downscale > 1:
            ny, nx = a.shape
            ny2, nx2 = ny // downscale, nx // downscale
            a = a[: ny2 * downscale, : nx2 * downscale].reshape(
                ny2, downscale, nx2, downscale).mean(axis=(1, 3))
        if threshold is not None:
            a = (a >= float(threshold)).astype(np.float64)
        if shape0 is None:
            shape0 = a.shape
        elif a.shape != shape0:
            raise ValueError(
                f"{p} shape {a.shape} != first frame {shape0}; "
                "frames must be co-registered and equal-sized"
            )
        frames.append(a)
    if not frames:
        raise ValueError("no frames given")
    return np.stack(frames, axis=0)
