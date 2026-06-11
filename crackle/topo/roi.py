"""Region-of-interest masks for diagram-level boundary filtering.

The kinematic loading bands at the top and bottom edges of the
hetero_pinning world create damage ridges whose persistence features are
artifacts of the boundary condition, not of material failure. Phase 0
counted them as events.

The fix is diagram-level filtering: features whose BIRTH cell falls
outside the ROI are dropped from significance counts and event
extraction. The field itself is never modified (zeroing it would create
fake gradients and new spurious features along the cut).

Conventions
-----------
- A mask is a bool array of shape (ny, nx); True = inside the ROI.
- The essential H0 class (global component) is always kept, wherever its
  birth cell lies: it represents the whole superlevel set, not a
  boundary artifact, and dropping it would corrupt Betti curves.
"""
from __future__ import annotations

import numpy as np

from crackle.topo.cubical import Diagram


def boundary_margin_mask(
    ny: int,
    nx: int,
    *,
    margin_cells: int | tuple[int, int] = 0,
) -> np.ndarray:
    """ROI mask excluding a margin of cells along the field edges.

    margin_cells: int applies to all four edges; a (margin_y, margin_x)
    tuple applies margin_y to top/bottom rows and margin_x to left/right
    columns (the hetero_pinning default is loading bands only:
    ``(k_cells, 0)``).
    """
    if isinstance(margin_cells, tuple):
        my, mx = int(margin_cells[0]), int(margin_cells[1])
    else:
        my = mx = int(margin_cells)
    if my < 0 or mx < 0:
        raise ValueError(f"margins must be >= 0, got ({my}, {mx})")
    if 2 * my >= ny or 2 * mx >= nx:
        raise ValueError(f"margins ({my}, {mx}) erase the whole {ny}x{nx} grid")
    mask = np.zeros((ny, nx), dtype=bool)
    mask[my : ny - my if my else ny, mx : nx - mx if mx else nx] = True
    return mask


def horizon_margin_mask(
    ny: int,
    nx: int,
    *,
    height: float,
    horizon: float,
    k: float = 1.5,
    length: float | None = None,
    exclude_x: bool = False,
) -> np.ndarray:
    """Physical variant: exclude ``k * horizon`` from the top/bottom edges.

    Cell size is inferred from the node-grid geometry (make_grid places
    ny nodes across ``height``). With exclude_x=True the same physical
    margin is also applied to the left/right edges (requires ``length``).
    """
    dy = float(height) / max(ny - 1, 1)
    my = int(np.ceil(k * float(horizon) / max(dy, 1e-12)))
    mx = 0
    if exclude_x:
        if length is None:
            raise ValueError("exclude_x=True requires length")
        dx = float(length) / max(nx - 1, 1)
        mx = int(np.ceil(k * float(horizon) / max(dx, 1e-12)))
    return boundary_margin_mask(ny, nx, margin_cells=(my, mx))


def apply_roi(diag: Diagram, roi: np.ndarray | None) -> Diagram:
    """Drop non-essential features whose birth cell lies outside the ROI."""
    if roi is None:
        return diag
    mask = np.asarray(roi, dtype=bool)
    inside = mask[diag.birth_yx[:, 0], diag.birth_yx[:, 1]]
    return diag.select(inside | diag.essential)
