"""Measurement-noise model for damage-field movies (Phase 2.1 robustness).

The Phase 0/2.1 synthetic world is noiseless, so the causal-onset
false-alarm axis degenerated (FA ~ 0 for every signal). To make the
"topological signals lead the macroscopic signal" claim survive contact
with reality, we add a DIC-like measurement-noise layer and re-run the
detectors on the NOISY observation while keeping the failure-time
reference (t*, growth_start) from the CLEAN ground-truth simulation.

Noise model
-----------
Per frame, spatially-correlated Gaussian noise (white noise passed through
a Gaussian blur of correlation length ``corr_cells``, then renormalized to
per-cell std ``sigma``) is added to the damage field and clipped to [0, 1].
Spatial correlation mimics DIC subset-based measurement error; iid noise
(corr_cells=0) is available as a harsher stress test. The model is
deterministic given a seed.

This degrades BOTH the topological signals (extrema-based, noise-sensitive)
and the macroscopic control total_damage (a sum, partly noise-averaging) on
the SAME noisy field, so the comparison stays fair.
"""
from __future__ import annotations

import numpy as np


def add_measurement_noise(
    movie: np.ndarray,
    *,
    sigma: float,
    corr_cells: float = 1.5,
    seed: int,
) -> np.ndarray:
    """Add DIC-like measurement noise to a (T, ny, nx) damage movie in [0,1].

    sigma is the target per-cell standard deviation of the added noise.
    corr_cells is the Gaussian spatial-correlation length in cells (0 = iid).
    Returns a new array; the input is not modified. sigma <= 0 is identity.
    """
    movie = np.asarray(movie, dtype=np.float64)
    if sigma <= 0.0:
        return movie.copy()
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(movie.shape)
    if corr_cells > 0.0:
        from scipy.ndimage import gaussian_filter

        for t in range(noise.shape[0]):
            noise[t] = gaussian_filter(noise[t], corr_cells, mode="reflect")
    # renormalize to the requested per-cell std (blurring shrinks variance)
    std = float(noise.std())
    if std > 0.0:
        noise *= sigma / std
    return np.clip(movie + noise, 0.0, 1.0)
