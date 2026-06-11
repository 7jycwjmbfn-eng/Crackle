"""Macroscopic instability reference time and precursor lead-time analysis.

Phase-0 question: do topological curves move BEFORE the macroscopic
instability? We define:

- instability_step: argmax of the bond-breaking rate (d/dt of total damage),
  i.e. the most violent load step of the reference simulation.
- precursor_step of a curve: first step where the curve's forward increment
  exceeds `frac` of its own maximum increment (a crude but honest onset
  detector; no future information beyond the max normalization, which is
  acceptable for a retrospective audit and must be replaced by a causal
  detector in Phase 2).
- lead = instability_step - precursor_step (positive = early warning).
"""
from __future__ import annotations

import numpy as np


def instability_step(total_damage: np.ndarray) -> int:
    """Most violent step: argmax of the damage increment."""
    d = np.asarray(total_damage, dtype=np.float64)
    if d.size < 2:
        return 0
    inc = np.diff(d)
    return int(np.argmax(inc) + 1)


def onset_step(curve: np.ndarray, *, frac: float = 0.25) -> int | None:
    """First step whose increment exceeds frac * max increment."""
    c = np.asarray(curve, dtype=np.float64)
    if c.size < 2:
        return None
    inc = np.diff(c)
    peak = float(np.max(np.abs(inc)))
    if peak <= 0.0:
        return None
    hits = np.flatnonzero(np.abs(inc) >= float(frac) * peak)
    return int(hits[0] + 1) if hits.size else None


def lead_time_table(
    curves: dict[str, np.ndarray],
    total_damage: np.ndarray,
    *,
    frac: float = 0.25,
) -> list[dict[str, float | str | None]]:
    t_star = instability_step(total_damage)
    rows: list[dict[str, float | str | None]] = []
    for name, curve in curves.items():
        t_on = onset_step(curve, frac=frac)
        rows.append(
            {
                "signal": name,
                "instability_step": int(t_star),
                "onset_step": None if t_on is None else int(t_on),
                "lead_steps": None if t_on is None else int(t_star - t_on),
            }
        )
    return rows
