"""Topological event extraction across load steps.

The point of this module is to turn a damage-field movie into a discrete
event stream that the existing Crackle survival/hazard machinery can consume:

- h0_born : a new significant damage hotspot nucleates (local max appears)
- h0_died : a hotspot merges into another component (coalescence)
- h1_born : a significant loop appears (crack/damage encircles material)
- h1_died : a loop is filled in or broken open

Matching between consecutive diagrams runs on birth locations plus a
birth-value penalty, gated by a distance cutoff (crackle.topo.matching).
Default method is "wasserstein" (optimal assignment, Phase 1); the Phase-0
greedy heuristic remains available as method="greedy".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackle.topo.cubical import Diagram
from crackle.topo.matching import match_features
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
    method: str,
) -> list[TopoEvent]:
    label = f"h{dim}"
    p = apply_roi(prev.select(prev.dim == dim), roi).significant(
        sig_tau, include_essential=False
    )
    c = apply_roi(curr.select(curr.dim == dim), roi).significant(
        sig_tau, include_essential=False
    )
    m_prev, m_curr = match_features(
        p.birth_yx, p.birth, c.birth_yx, c.birth,
        method=method, max_dist=max_dist, value_weight=value_weight,
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
    method: str = "wasserstein",
) -> list[TopoEvent]:
    """Extract topological events from a diagram sequence.

    sig_tau filters noise bars BEFORE matching, so events are jumps in the
    *significant* topology only. max_dist is in grid cells. roi (bool
    (ny, nx) mask, True = keep) drops boundary-artifact features at the
    diagram level before matching; see crackle.topo.roi. method selects
    the matcher ("wasserstein" = optimal assignment, default since Phase 1;
    "greedy" = Phase-0 heuristic); see crackle.topo.matching.
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
                    method=method,
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
