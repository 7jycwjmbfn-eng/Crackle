"""Per-frame topological summaries of a damage field sequence.

These are the candidate precursor signals for Phase 0: scalar curves over
load steps that may (or may not) move before macroscopic instability.
"""
from __future__ import annotations

import numpy as np

from crackle.topo.cubical import Diagram, betti_at_level, superlevel_persistence
from crackle.topo.roi import apply_roi


def persistence_entropy(persistence: np.ndarray) -> float:
    """Shannon entropy of the normalized finite lifetimes. 0 if < 2 bars."""
    p = np.asarray(persistence, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0.0)]
    if p.size < 2:
        return 0.0
    q = p / p.sum()
    return float(-np.sum(q * np.log(q)))


def frame_summary(
    diag: Diagram,
    *,
    sig_tau: float,
    betti_levels: tuple[float, ...] = (0.1, 0.2, 0.35),
) -> dict[str, float]:
    h0, h1 = diag.split()
    h0_sig = h0.significant(sig_tau)
    h1_sig = h1.significant(sig_tau, include_essential=False)
    out: dict[str, float] = {
        "n_h0_sig": float(h0_sig.dim.size),
        "n_h1_sig": float(h1_sig.dim.size),
        "total_pers_h0": float(np.sum(h0.persistence[~h0.essential])),
        "total_pers_h1": float(np.sum(h1.persistence)),
        "max_pers_h1": float(h1.persistence.max()) if h1.dim.size else 0.0,
        "entropy_h0": persistence_entropy(h0.persistence[~h0.essential]),
        "entropy_h1": persistence_entropy(h1.persistence),
        "field_max": diag.field_max,
    }
    for level in betti_levels:
        b0, b1 = betti_at_level(diag, level)
        key = f"{level:g}".replace(".", "p")
        out[f"betti0_at_{key}"] = float(b0)
        out[f"betti1_at_{key}"] = float(b1)
    return out


def sequence_summaries(
    fields: np.ndarray,
    *,
    sig_tau: float,
    betti_levels: tuple[float, ...] = (0.1, 0.2, 0.35),
    connectivity: str = "8",
    roi: np.ndarray | None = None,
) -> tuple[list[dict[str, float]], list[Diagram]]:
    """Compute diagrams and summaries for a (T, ny, nx) field sequence.

    roi (bool (ny, nx) mask, True = keep) drops boundary-artifact features
    from the summaries at the diagram level. The returned diagrams are the
    UNFILTERED ones, so callers can compare with/without ROI downstream;
    pass the same roi to extract_events for a consistent event stream.
    """
    f = np.asarray(fields, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError(f"expected (T, ny, nx), got {f.shape}")
    diagrams: list[Diagram] = []
    rows: list[dict[str, float]] = []
    for t in range(f.shape[0]):
        diag = superlevel_persistence(f[t], maxdim=1, connectivity=connectivity)
        diagrams.append(diag)
        row = frame_summary(
            apply_roi(diag, roi), sig_tau=sig_tau, betti_levels=betti_levels
        )
        row["step"] = float(t)
        rows.append(row)
    return rows, diagrams
