"""Superlevel cubical persistence for 2D damage fields.

Backend: CubicalRipser (cripser). Sublevel persistence of (-field) is the
superlevel persistence of field, so births/deaths are negated back.

Conventions
-----------
- Input fields are 2D arrays of shape (ny, nx); index order is (row=y, col=x).
- Superlevel filtration: features are born at HIGH values and die at LOW
  values, so birth >= death and persistence = birth - death >= 0.
- The essential H0 class (the global component) gets death = field.min()
  and essential=True.

This module has no dependency on the (currently unpushed) crackle.data
package, so it imports cleanly in the public archive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cripser

    _BACKEND = "cripser"
except Exception:  # pragma: no cover
    cripser = None
    _BACKEND = None

try:
    import tcripser  # ships inside the cripser wheel; T-construction
except Exception:  # pragma: no cover
    tcripser = None

try:  # optional fallback
    import gudhi  # noqa: F401

    _HAVE_GUDHI = True
except Exception:  # pragma: no cover
    _HAVE_GUDHI = False


@dataclass
class Diagram:
    """Persistence diagram of one 2D frame (superlevel filtration)."""

    dim: np.ndarray  # (M,) int, 0 or 1
    birth: np.ndarray  # (M,) float, superlevel birth value
    death: np.ndarray  # (M,) float, superlevel death value (<= birth)
    birth_yx: np.ndarray  # (M, 2) int, (row, col) of the birth cell
    death_yx: np.ndarray  # (M, 2) int, (row, col) of the death cell
    essential: np.ndarray  # (M,) bool
    field_min: float
    field_max: float

    @property
    def persistence(self) -> np.ndarray:
        return self.birth - self.death

    def select(self, mask: np.ndarray) -> "Diagram":
        return Diagram(
            dim=self.dim[mask],
            birth=self.birth[mask],
            death=self.death[mask],
            birth_yx=self.birth_yx[mask],
            death_yx=self.death_yx[mask],
            essential=self.essential[mask],
            field_min=self.field_min,
            field_max=self.field_max,
        )

    def split(self) -> tuple["Diagram", "Diagram"]:
        return self.select(self.dim == 0), self.select(self.dim == 1)

    def significant(self, tau: float, include_essential: bool = True) -> "Diagram":
        mask = self.persistence >= float(tau)
        if include_essential:
            mask = mask | self.essential
        return self.select(mask)


def superlevel_persistence(
    field: np.ndarray, maxdim: int = 1, connectivity: str = "8"
) -> Diagram:
    """Superlevel cubical persistence diagram of a 2D field.

    connectivity="8" (default) uses the T-construction: superlevel sets are
    8-connected, which keeps diagonal crack segments connected and pairs with
    4-connected enclosed material islands (standard digital-topology duality).
    connectivity="4" uses the V-construction (4-connected superlevel sets).
    """
    if _BACKEND != "cripser":
        raise RuntimeError(
            "cripser is required (pip install cripser). gudhi fallback not wired "
            "in to keep one canonical convention."
        )
    f = np.ascontiguousarray(np.asarray(field, dtype=np.float64))
    if f.ndim != 2:
        raise ValueError(f"expected 2D field, got shape {f.shape}")
    if connectivity == "8":
        if tcripser is None:
            raise RuntimeError("tcripser module missing from cripser install")
        raw = tcripser.computePH(-f, maxdim=int(maxdim))
    elif connectivity == "4":
        raw = cripser.computePH(-f, maxdim=int(maxdim))
    else:
        raise ValueError("connectivity must be '4' or '8'")
    if raw.size == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float64)
        empty_yx = np.zeros((0, 2), dtype=np.int64)
        return Diagram(empty_i, empty_f, empty_f, empty_yx, empty_yx,
                       np.zeros((0,), dtype=bool), float(f.min()), float(f.max()))
    dim = raw[:, 0].astype(np.int64)
    birth = -raw[:, 1]
    death = -raw[:, 2]
    essential = ~np.isfinite(death) | (death < float(f.min()) - 1.0)
    death = np.where(essential, float(f.min()), death)
    # cripser reports cell coordinates in array index order (axis0, axis1).
    birth_yx = raw[:, 3:5].astype(np.int64)
    death_yx = raw[:, 6:8].astype(np.int64)
    ny, nx = f.shape
    birth_yx[:, 0] = np.clip(birth_yx[:, 0], 0, ny - 1)
    birth_yx[:, 1] = np.clip(birth_yx[:, 1], 0, nx - 1)
    death_yx[:, 0] = np.clip(death_yx[:, 0], 0, ny - 1)
    death_yx[:, 1] = np.clip(death_yx[:, 1], 0, nx - 1)
    order = np.argsort(-(birth - death))
    return Diagram(
        dim=dim[order],
        birth=birth[order],
        death=death[order],
        birth_yx=birth_yx[order],
        death_yx=death_yx[order],
        essential=essential[order],
        field_min=float(f.min()),
        field_max=float(f.max()),
    )


def betti_at_level(diag: Diagram, level: float) -> tuple[int, int]:
    """Betti numbers of the superlevel set {field >= level}."""
    alive = (diag.birth >= float(level)) & (
        (diag.death < float(level)) | diag.essential
    )
    return int(np.count_nonzero(alive & (diag.dim == 0))), int(
        np.count_nonzero(alive & (diag.dim == 1))
    )


def backend() -> str | None:
    return _BACKEND
