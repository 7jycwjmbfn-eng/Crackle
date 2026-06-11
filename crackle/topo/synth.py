"""Multi-notch randomized synthetic world for topological event mining.

Generalizes the hetero_pinning reference simulator from one edge notch to
N arbitrary notch segments (random position / angle / length), within the
same kinematic-loading approximation:

- global uniaxial tension in y with Poisson contraction in x,
- per-notch opening displacement decaying with distance to the segment,
  signed by the segment normal (symmetric opening for interior cracks),
- per-tip critical-stretch weakening so every crack tip can propagate,
- correlated lognormal-ish toughness field reused from hetero_pinning.

This stays a *kinematic proxy*, exactly like the original: positions are
prescribed, not solved. It is a data generator for topology mining, not a
mechanics solver, and must be described as such in any report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crackle.benchmarks.notched_plate import build_bond_graph
from crackle.experiments.hetero_pinning import (
    correlated_toughness_field,
    make_grid,
)


@dataclass
class Notch:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def p0(self) -> np.ndarray:
        return np.array([self.x0, self.y0], dtype=np.float64)

    @property
    def p1(self) -> np.ndarray:
        return np.array([self.x1, self.y1], dtype=np.float64)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def normal(self) -> np.ndarray:
        d = self.p1 - self.p0
        n = np.array([-d[1], d[0]], dtype=np.float64)
        return n / max(np.linalg.norm(n), 1e-12)


@dataclass
class MultiNotchCase:
    notches: list[Notch]
    contrast: float
    corr_length_mm: float
    seed: int
    nx: int = 48
    ny: int = 29
    length: float = 100.0
    height: float = 40.0
    horizon: float = 5.2
    steps: int = 80
    tension_strain: float = 0.075
    critical_stretch: float = 0.045
    opening_factor: float = 0.34
    poisson: float = 0.28
    meta: dict = field(default_factory=dict)


def _segments_cross(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
    def orient(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    d1, d2 = orient(b0, b1, a0), orient(b0, b1, a1)
    d3, d4 = orient(a0, a1, b0), orient(a0, a1, b1)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _point_segment_distance(p: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    """Vectorized distance from points (N,2) to one segment."""
    d = s1 - s0
    denom = float(d @ d)
    t = np.clip(((p - s0[None, :]) @ d) / max(denom, 1e-12), 0.0, 1.0)
    proj = s0[None, :] + t[:, None] * d[None, :]
    return np.linalg.norm(p - proj, axis=1)


def sample_notches(
    rng: np.random.Generator,
    *,
    n_notches: int,
    length: float,
    height: float,
    min_gap: float,
    include_edge_notch: bool,
    max_angle_deg: float = 35.0,
    len_range: tuple[float, float] = (8.0, 20.0),
    max_tries: int = 400,
) -> list[Notch]:
    """Random non-crossing notch set. Angles near horizontal (tension is in y)."""
    notches: list[Notch] = []
    if include_edge_notch:
        notch_len = float(rng.uniform(0.15, 0.30)) * length
        y = float(rng.uniform(-0.2, 0.2)) * height
        notches.append(Notch(0.0, y, notch_len, y))
    tries = 0
    while len(notches) < n_notches and tries < max_tries:
        tries += 1
        seg_len = float(rng.uniform(*len_range))
        ang = np.deg2rad(float(rng.uniform(-max_angle_deg, max_angle_deg)))
        cx = float(rng.uniform(0.15 * length, 0.9 * length))
        cy = float(rng.uniform(-0.38 * height, 0.38 * height))
        dx, dy = 0.5 * seg_len * np.cos(ang), 0.5 * seg_len * np.sin(ang)
        cand = Notch(cx - dx, cy - dy, cx + dx, cy + dy)
        ok = True
        for other in notches:
            if _segments_cross(cand.p0, cand.p1, other.p0, other.p1):
                ok = False
                break
            mids = 0.5 * (cand.p0 + cand.p1) - 0.5 * (other.p0 + other.p1)
            if np.linalg.norm(mids) < min_gap:
                ok = False
                break
        if ok:
            notches.append(cand)
    return notches


def initial_alive_multinotch(
    points: np.ndarray, bonds: np.ndarray, notches: list[Notch]
) -> np.ndarray:
    alive = np.ones((bonds.shape[0],), dtype=bool)
    for index, (i, j) in enumerate(np.asarray(bonds, dtype=np.int64)):
        for nt in notches:
            if _segments_cross(points[i], points[j], nt.p0, nt.p1):
                alive[index] = False
                break
    return alive


def _positions(case: MultiNotchCase, points: np.ndarray, load_fraction: float) -> np.ndarray:
    strain = case.tension_strain * float(load_fraction)
    out = points.copy()
    out[:, 0] += -case.poisson * strain * (points[:, 0] - 0.5 * case.length)
    out[:, 1] += strain * points[:, 1]
    for nt in notches_of(case):
        dist = _point_segment_distance(points, nt.p0, nt.p1)
        side = np.sign((points - nt.p0[None, :]) @ nt.normal)
        opening = (
            strain * case.height * case.opening_factor
            * np.exp(-dist / max(2.6 * case.horizon, 1e-9))
            * side
        )
        out += opening[:, None] * nt.normal[None, :]
    return out


def notches_of(case: MultiNotchCase) -> list[Notch]:
    return case.notches


def simulate_multinotch(case: MultiNotchCase) -> dict[str, np.ndarray | float]:
    """Reference rollout. Returns damage movie (T+1, ny, nx) and bond data."""
    points, _, _, _ = make_grid(case.nx, case.ny, case.length, case.height)
    gc_field = correlated_toughness_field(
        nx=case.nx, ny=case.ny, length=case.length, height=case.height,
        contrast=case.contrast, corr_length=case.corr_length_mm, seed=case.seed,
    )
    gc_nodes = gc_field.reshape(-1)
    bonds, rest = build_bond_graph(points, case.horizon)
    alive_now = initial_alive_multinotch(points, bonds, case.notches)

    gc_bond = 0.5 * (gc_nodes[bonds[:, 0]] + gc_nodes[bonds[:, 1]])
    critical = case.critical_stretch * np.sqrt(gc_bond / max(float(np.mean(gc_nodes)), 1e-12))
    mid = 0.5 * (points[bonds[:, 0]] + points[bonds[:, 1]])
    # tip weakening at BOTH endpoints of every notch
    weaken = np.zeros((bonds.shape[0],), dtype=np.float64)
    for nt in case.notches:
        for tip in (nt.p0, nt.p1):
            if tip[0] <= 1e-9:  # edge-notch root: no tip there
                continue
            dist_tip = np.linalg.norm(mid - tip[None, :], axis=1)
            weaken = np.maximum(weaken, 0.34 * np.exp(-dist_tip / max(2.8 * case.horizon, 1e-9)))
    critical = np.maximum(0.30 * case.critical_stretch, critical * (1.0 - weaken))

    n_nodes = points.shape[0]
    incident = np.zeros((n_nodes,), dtype=np.float64)
    np.add.at(incident, bonds[:, 0], 1.0)
    np.add.at(incident, bonds[:, 1], 1.0)
    incident = np.maximum(incident, 1.0)

    movie = np.zeros((case.steps + 1, case.ny, case.nx), dtype=np.float64)
    for step in range(case.steps + 1):
        pos = _positions(case, points, step / max(case.steps, 1))
        stretch = np.linalg.norm(pos[bonds[:, 1]] - pos[bonds[:, 0]], axis=1)
        stretch = (stretch - rest) / np.maximum(rest, 1e-12)
        if step > 0:
            alive_now = alive_now & ~(alive_now & (stretch > critical))
        broken = np.zeros((n_nodes,), dtype=np.float64)
        dead = ~alive_now
        np.add.at(broken, bonds[dead, 0], 1.0)
        np.add.at(broken, bonds[dead, 1], 1.0)
        movie[step] = (broken / incident).reshape(case.ny, case.nx)
    return {
        "movie": movie,
        "gc_field": gc_field,
        "n_bonds": float(bonds.shape[0]),
        "final_broken_frac": float(np.mean(~alive_now)),
    }


def sample_case(
    rng: np.random.Generator,
    *,
    case_seed: int,
    nx: int,
    ny: int,
    steps: int,
    contrast_range: tuple[float, float] = (2.0, 6.0),
    corr_choices: tuple[float, ...] = (2.5, 7.5, 15.0),
    n_notch_range: tuple[int, int] = (1, 4),
    edge_notch_prob: float = 0.5,
) -> MultiNotchCase:
    length, height, horizon = 100.0, 40.0, 5.2
    n_notches = int(rng.integers(n_notch_range[0], n_notch_range[1] + 1))
    include_edge = bool(rng.random() < edge_notch_prob)
    notches = sample_notches(
        rng, n_notches=n_notches, length=length, height=height,
        min_gap=2.0 * horizon, include_edge_notch=include_edge,
    )
    return MultiNotchCase(
        notches=notches,
        contrast=float(rng.uniform(*contrast_range)),
        corr_length_mm=float(rng.choice(corr_choices)),
        seed=case_seed,
        nx=nx, ny=ny, steps=steps,
        meta={
            "n_notches": len(notches),
            "include_edge": include_edge,
            "notch_segments": [[n.x0, n.y0, n.x1, n.y1] for n in notches],
        },
    )
