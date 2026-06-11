"""Topological event extraction across load steps.

The point of this module is to turn a damage-field movie into a discrete
event stream that the existing Crackle survival/hazard machinery can consume:

- h0_born : a new significant damage hotspot nucleates (local max appears)
- h0_died : a hotspot merges into another component (coalescence)
- h1_born : a significant loop appears (crack/damage encircles material)
- h1_died : a loop is filled in or broken open

Matching between consecutive diagrams is greedy nearest-neighbor on birth
locations (plus a birth-value penalty), gated by a distance cutoff. This is a
Phase-0 heuristic, NOT optimal transport; Phase 1 should swap in a proper
Wasserstein matching (persim) if event identity over long horizons matters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackle.topo.cubical import Diagram
from crackle.topo.roi import apply_roi


@dataclass
class TopoEvent:
    step: int  # event attributed to transition (step-1) -> step
    kind: str  # h0_born | h0_died | h1_born | h1_died
    y: int
    x: int
    persistence: float
    birth_value: float

    def as_row(self) -> dict[str, float | int | str]:
        return {
            "step": int(self.step),
            "kind": self.kind,
            "y": int(self.y),
            "x": int(self.x),
            "persistence": float(self.persistence),
            "birth_value": float(self.birth_value),
        }


def _greedy_match(
    a_yx: np.ndarray,
    a_val: np.ndarray,
    b_yx: np.ndarray,
    b_val: np.ndarray,
    *,
    max_dist: float,
    value_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy 1-1 matching. Returns (matched_a_mask, matched_b_mask)."""
    na, nb = a_yx.shape[0], b_yx.shape[0]
    matched_a = np.zeros((na,), dtype=bool)
    matched_b = np.zeros((nb,), dtype=bool)
    if na == 0 or nb == 0:
        return matched_a, matched_b
    d_sp = np.linalg.norm(
        a_yx[:, None, :].astype(np.float64) - b_yx[None, :, :].astype(np.float64),
        axis=2,
    )
    d_val = np.abs(a_val[:, None] - b_val[None, :])
    cost = d_sp + float(value_weight) * d_val
    cost[d_sp > float(max_dist)] = np.inf
    flat = np.argsort(cost, axis=None)
    for idx in flat:
        i, j = np.unravel_index(idx, cost.shape)
        if not np.isfinite(cost[i, j]):
            break
        if matched_a[i] or matched_b[j]:
            continue
        matched_a[i] = True
        matched_b[j] = True
    return matched_a, matched_b


def _events_one_dim(
    prev: Diagram,
    curr: Diagram,
    *,
    step: int,
    dim: int,
    sig_tau: float,
    max_dist: float,
    value_weight: float,
    roi: np.ndarray | None,
) -> list[TopoEvent]:
    label = f"h{dim}"
    p = apply_roi(prev.select(prev.dim == dim), roi).significant(
        sig_tau, include_essential=False
    )
    c = apply_roi(curr.select(curr.dim == dim), roi).significant(
        sig_tau, include_essential=False
    )
    m_prev, m_curr = _greedy_match(
        p.birth_yx, p.birth, c.birth_yx, c.birth,
        max_dist=max_dist, value_weight=value_weight,
    )
    events: list[TopoEvent] = []
    for j in np.flatnonzero(~m_curr):  # in curr, unmatched -> born
        events.append(
            TopoEvent(
                step=step,
                kind=f"{label}_born",
                y=int(c.birth_yx[j, 0]),
                x=int(c.birth_yx[j, 1]),
                persistence=float(c.persistence[j]),
                birth_value=float(c.birth[j]),
            )
        )
    for i in np.flatnonzero(~m_prev):  # in prev, unmatched -> died
        events.append(
            TopoEvent(
                step=step,
                kind=f"{label}_died",
                y=int(p.death_yx[i, 0]),
                x=int(p.death_yx[i, 1]),
                persistence=float(p.persistence[i]),
                birth_value=float(p.birth[i]),
            )
        )
    return events


def extract_events(
    diagrams: list[Diagram],
    *,
    sig_tau: float,
    max_dist: float = 6.0,
    value_weight: float = 4.0,
    roi: np.ndarray | None = None,
) -> list[TopoEvent]:
    """Extract topological events from a diagram sequence.

    sig_tau filters noise bars BEFORE matching, so events are jumps in the
    *significant* topology only. max_dist is in grid cells. roi (bool
    (ny, nx) mask, True = keep) drops boundary-artifact features at the
    diagram level before matching; see crackle.topo.roi.
    """
    events: list[TopoEvent] = []
    for t in range(1, len(diagrams)):
        for dim in (0, 1):
            events.extend(
                _events_one_dim(
                    diagrams[t - 1],
                    diagrams[t],
                    step=t,
                    dim=dim,
                    sig_tau=sig_tau,
                    max_dist=max_dist,
                    value_weight=value_weight,
                    roi=roi,
                )
            )
    return events


def event_count_curves(
    events: list[TopoEvent], n_steps: int
) -> dict[str, np.ndarray]:
    kinds = ("h0_born", "h0_died", "h1_born", "h1_died")
    out = {k: np.zeros((n_steps,), dtype=np.int64) for k in kinds}
    for ev in events:
        if ev.kind in out and 0 <= ev.step < n_steps:
            out[ev.kind][ev.step] += 1
    out["all"] = sum(out[k] for k in kinds)
    return out
