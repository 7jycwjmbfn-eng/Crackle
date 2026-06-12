"""Track B — differentiable persistence-diagram vectorization (PersLay-style).

Three representations of the per-frame significant diagram, compared under
an identical hazard head and training protocol (the representation is the
only variable):

- scalar   : the hand-crafted Phase-0 summary curves (12 features)
- pi_fixed : fixed-weight persistence image, 8x8 Gaussian grid over
             (birth, persistence) per homology dim -> 128 features
- perslay  : learned point transform phi(birth, pers, dim) with a learned
             persistence-dependent weight, sum-pooled (permutation
             invariant) -> d_out features, trained end-to-end

References: Adams et al. (persistence images), Carriere et al. (PersLay).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

PI_GRID = 8
PI_SIGMA = 0.06
PI_BIRTH_RANGE = (0.0, 1.0)
PI_PERS_RANGE = (0.0, 0.7)


def persistence_image(
    pts: np.ndarray, *, grid: int = PI_GRID, sigma: float = PI_SIGMA
) -> np.ndarray:
    """Fixed persistence image. pts: (P, 3) [dim, birth, persistence].

    Returns (2 * grid * grid,) float32 — one image per homology dim,
    persistence-weighted Gaussians, hard-clipped support for speed.
    """
    out = np.zeros((2, grid, grid), dtype=np.float32)
    if pts.size == 0:
        return out.reshape(-1)
    bx = np.linspace(*PI_BIRTH_RANGE, grid)
    py = np.linspace(*PI_PERS_RANGE, grid)
    for dim, birth, pers in pts:
        d = int(dim)
        if d > 1:
            continue
        gb = np.exp(-0.5 * ((bx - birth) / sigma) ** 2)
        gp = np.exp(-0.5 * ((py - pers) / sigma) ** 2)
        out[d] += float(pers) * np.outer(gp, gb).astype(np.float32)
    return out.reshape(-1)


class PersLayEncoder(nn.Module):
    """Permutation-invariant learned vectorization of a padded diagram."""

    def __init__(self, d_out: int = 64, hidden: int = 64):
        super().__init__()
        # point features: birth, persistence, one-hot dim
        self.phi = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, d_out))
        self.weight = nn.Linear(1, 1)

    def forward(self, pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """pts: (B, P, 3) [dim, birth, pers]; mask: (B, P) bool."""
        dim = pts[..., 0].long().clamp(0, 1)
        feats = torch.cat([
            pts[..., 1:3],
            nn.functional.one_hot(dim, 2).float(),
        ], dim=-1)
        w = nn.functional.softplus(self.weight(pts[..., 2:3]))
        emb = self.phi(feats) * w
        return (emb * mask.unsqueeze(-1)).sum(dim=1)


class HazardHead(nn.Module):
    """Shared MLP head: [local features + topo representation] -> logit."""

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TrackBModel(nn.Module):
    """Hazard model with a pluggable topology representation."""

    def __init__(self, variant: str, n_local: int, n_scalar: int = 12,
                 d_perslay: int = 64):
        super().__init__()
        self.variant = variant
        if variant == "scalar":
            d_topo = n_scalar
            self.encoder = None
        elif variant == "pi_fixed":
            d_topo = 2 * PI_GRID * PI_GRID
            self.encoder = None
        elif variant == "perslay":
            d_topo = d_perslay
            self.encoder = PersLayEncoder(d_out=d_perslay)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        self.head = HazardHead(n_local + d_topo)

    def forward(self, local: torch.Tensor, topo: torch.Tensor,
                diag_pts: torch.Tensor | None = None,
                diag_mask: torch.Tensor | None = None) -> torch.Tensor:
        """local: (B, T_tiles, n_local); topo: (B, d_topo) precomputed for
        scalar/pi_fixed; for perslay it is computed from diag_pts/mask.
        Returns logits (B, T_tiles)."""
        if self.variant == "perslay":
            topo = self.encoder(diag_pts, diag_mask)
        rep = topo.unsqueeze(1).expand(-1, local.shape[1], -1)
        return self.head(torch.cat([local, rep], dim=-1))
