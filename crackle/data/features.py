from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = [
    "bias",
    "stretch_ratio",
    "bond_damage",
    "material_weakness",
    "neighbor_broken_norm",
    "local_energy",
    "frontier_incident",
    "boundary_near",
    "load_mag",
    "time_frac",
    "history_trigger",
]

HYBRID_FEATURE_NAMES = FEATURE_NAMES + [
    "stretch_x_history",
    "energy_x_history",
    "weakness_x_stretch",
    "frontier_x_history",
    "stretch_sq",
    "energy_x_frontier",
]

REGIME_NAMES = [
    "clean_single_notch",
    "heterogeneous_toughness",
    "mixed_mode",
    "arrest_candidate",
    "hole_plus_notch",
    "branching",
]

REGIME_MAP = {
    "single_notch_baseline": "clean_single_notch",
    "heterogeneous_toughness": "heterogeneous_toughness",
    "single_edge_notch_mixed_mode": "mixed_mode",
    "off_center_impact": "mixed_mode",
    "arrest_candidate": "arrest_candidate",
    "hole_plus_notch": "hole_plus_notch",
    "branching_candidate": "branching",
    "double_notch_competing": "branching",
}

RANKER_FEATURE_NAMES = HYBRID_FEATURE_NAMES + [
    "frontier_distance_proxy",
    "frontier_alignment_abs",
    "frontier_curvature_proxy",
    "global_recent_event_rate",
    "recent_neighbor_event_norm",
    "load_along_bond",
    "load_normal_abs",
    "stretch_x_frontier_near",
    "energy_x_frontier_near",
    "history_x_frontier_near",
    "neighbor_x_frontier_near",
    "boundary_x_frontier_near",
] + [f"regime_{name}" for name in REGIME_NAMES]


_STATIC_CACHE: dict[tuple[int, int, tuple[int, ...]], tuple["BondGeometry", np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class BondGeometry:
    centers: np.ndarray
    orientation: np.ndarray
    boundary_distance: np.ndarray
    node_degree: np.ndarray


def bond_geometry(points: np.ndarray, bonds: np.ndarray) -> BondGeometry:
    xy = np.asarray(points, dtype=np.float64)
    edges = np.asarray(bonds, dtype=np.int64)
    centers = 0.5 * (xy[edges[:, 0]] + xy[edges[:, 1]])
    delta = xy[edges[:, 1]] - xy[edges[:, 0]]
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    orientation = delta / np.maximum(norm, 1e-12)
    lo = np.min(xy, axis=0)
    hi = np.max(xy, axis=0)
    dist = np.minimum(centers - lo[None, :], hi[None, :] - centers)
    boundary_distance = np.min(dist, axis=1)
    node_degree = np.zeros((xy.shape[0],), dtype=np.float64)
    np.add.at(node_degree, edges[:, 0], 1.0)
    np.add.at(node_degree, edges[:, 1], 1.0)
    return BondGeometry(centers=centers, orientation=orientation, boundary_distance=boundary_distance, node_degree=node_degree)


def _static(labels: dict[str, np.ndarray]) -> tuple[BondGeometry, np.ndarray, np.ndarray]:
    points = labels["reference_x"]
    bonds = labels["bonds"].astype(np.int64)
    key = (
        int(points.__array_interface__["data"][0]),
        int(bonds.__array_interface__["data"][0]),
        tuple(bonds.shape),
    )
    cached = _STATIC_CACHE.get(key)
    if cached is not None:
        return cached
    geom = bond_geometry(points, bonds)
    toughness = material_toughness(labels, bonds)
    load_mag = local_load_magnitude(labels, bonds)
    cached = (geom, toughness, load_mag)
    _STATIC_CACHE[key] = cached
    return cached


def node_to_bond_mean(values: np.ndarray, bonds: np.ndarray, bond_indices: np.ndarray | None = None) -> np.ndarray:
    edges = np.asarray(bonds, dtype=np.int64)
    if bond_indices is not None:
        edges = edges[np.asarray(bond_indices, dtype=np.int64)]
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return 0.5 * (arr[edges[:, 0]] + arr[edges[:, 1]])


def frontier_bond_mask(crack_tip_mask_t: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    tip = np.asarray(crack_tip_mask_t, dtype=bool).reshape(-1)
    edges = np.asarray(bonds, dtype=np.int64)
    return tip[edges[:, 0]] | tip[edges[:, 1]]


def neighbor_broken_count(alive_t: np.ndarray, bonds: np.ndarray, node_degree: np.ndarray, bond_indices: np.ndarray | None = None) -> np.ndarray:
    alive = np.asarray(alive_t, dtype=bool)
    edges = np.asarray(bonds, dtype=np.int64)
    broken = ~alive
    node_broken = np.zeros((node_degree.shape[0],), dtype=np.float64)
    np.add.at(node_broken, edges[:, 0], broken.astype(np.float64))
    np.add.at(node_broken, edges[:, 1], broken.astype(np.float64))
    selected_edges = edges if bond_indices is None else edges[np.asarray(bond_indices, dtype=np.int64)]
    selected_broken = broken if bond_indices is None else broken[np.asarray(bond_indices, dtype=np.int64)]
    counts = node_broken[selected_edges[:, 0]] + node_broken[selected_edges[:, 1]]
    counts -= 2.0 * selected_broken.astype(np.float64)
    return np.maximum(counts, 0.0)


def local_load_magnitude(labels: dict[str, np.ndarray], bonds: np.ndarray) -> np.ndarray:
    loads = labels.get("loads")
    if loads is None:
        return np.zeros((bonds.shape[0],), dtype=np.float64)
    mag = np.linalg.norm(np.asarray(loads, dtype=np.float64), axis=1)
    return node_to_bond_mean(mag, bonds)


def material_toughness(labels: dict[str, np.ndarray], bonds: np.ndarray) -> np.ndarray:
    value = labels.get("material_toughness")
    if value is None:
        return np.ones((bonds.shape[0],), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != bonds.shape[0]:
        raise ValueError(f"material_toughness has {arr.size} entries, expected {bonds.shape[0]}")
    return np.maximum(arr, 1e-6)


def feature_matrix(
    labels: dict[str, np.ndarray],
    step: int,
    bond_indices: np.ndarray,
    *,
    history_trigger: np.ndarray | None = None,
    include_history: bool = True,
) -> np.ndarray:
    bonds = labels["bonds"].astype(np.int64)
    indices = np.asarray(bond_indices, dtype=np.int64)
    geom, toughness, load_mag = _static(labels)
    selected_edges = bonds[indices]
    alive_t = labels["bond_alive"][step].astype(bool)
    stretch_t = labels["bond_stretch"][step]
    damage_t = labels["damage"][step]
    energy_t = labels["strain_energy"][step]
    neighbor = neighbor_broken_count(alive_t, bonds, geom.node_degree, indices)
    tip = np.asarray(labels["crack_tip_mask"][step], dtype=bool)
    frontier = tip[selected_edges[:, 0]] | tip[selected_edges[:, 1]]
    degree_sum = geom.node_degree[selected_edges[:, 0]] + geom.node_degree[selected_edges[:, 1]]
    boundary_scale = max(float(np.max(geom.boundary_distance)), 1e-12)
    local_energy = node_to_bond_mean(energy_t, bonds, indices)
    hist = np.zeros((bonds.shape[0],), dtype=np.float64) if history_trigger is None else np.asarray(history_trigger, dtype=np.float64)
    if hist.size != bonds.shape[0]:
        raise ValueError(f"history_trigger has {hist.size} entries, expected {bonds.shape[0]}")
    features = np.stack(
        [
            np.ones((indices.size,), dtype=np.float64),
            stretch_t[indices] / toughness[indices],
            node_to_bond_mean(damage_t, bonds, indices),
            1.0 / toughness[indices],
            neighbor / np.maximum(degree_sum, 1.0),
            local_energy,
            frontier.astype(np.float64),
            1.0 - geom.boundary_distance[indices] / boundary_scale,
            load_mag[indices],
            np.full((indices.size,), float(step) / max(labels["bond_alive"].shape[0] - 1, 1), dtype=np.float64),
            hist[indices] if include_history else np.zeros((indices.size,), dtype=np.float64),
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def hybrid_feature_matrix(
    labels: dict[str, np.ndarray],
    step: int,
    bond_indices: np.ndarray,
    *,
    history_trigger: np.ndarray | None = None,
    include_history: bool = True,
) -> np.ndarray:
    base = feature_matrix(labels, step, bond_indices, history_trigger=history_trigger, include_history=include_history)
    stretch = base[:, 1]
    weakness = base[:, 3]
    energy = base[:, 5]
    frontier = base[:, 6]
    history = base[:, 10]
    extra = np.stack(
        [
            stretch * history,
            energy * history,
            weakness * stretch,
            frontier * history,
            stretch * stretch,
            energy * frontier,
        ],
        axis=1,
    )
    return np.nan_to_num(np.concatenate([base, extra], axis=1), nan=0.0, posinf=1e6, neginf=-1e6)


def regime_name(sample: dict[str, object] | None) -> str | None:
    if sample is None:
        return None
    raw = str(sample.get("regime") or sample.get("benchmark_case_id") or "")
    return REGIME_MAP.get(raw, raw if raw in REGIME_NAMES else None)


def _frontier_distance_norm(labels: dict[str, np.ndarray], step: int, indices: np.ndarray, geom: BondGeometry) -> np.ndarray:
    tip = np.asarray(labels["crack_tip_mask"][step], dtype=bool).reshape(-1)
    if not np.any(tip):
        return np.ones((indices.size,), dtype=np.float64)
    points = np.asarray(labels["reference_x"], dtype=np.float64)
    tip_points = points[tip]
    if tip_points.shape[0] > 512:
        stride = int(np.ceil(tip_points.shape[0] / 512.0))
        tip_points = tip_points[::stride]
    centers = geom.centers[indices]
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    diag = max(float(np.linalg.norm(hi - lo)), 1e-12)
    if centers.shape[0] * tip_points.shape[0] <= 4_000_000:
        d2 = np.sum((centers[:, None, :] - tip_points[None, :, :]) ** 2, axis=2)
        return np.clip(np.sqrt(np.min(d2, axis=1)) / diag, 0.0, 1.0)
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(tip_points).query(centers, k=1)
        return np.clip(np.asarray(dist, dtype=np.float64) / diag, 0.0, 1.0)
    except Exception:
        best = np.full((centers.shape[0],), np.inf, dtype=np.float64)
        for start in range(0, tip_points.shape[0], 2048):
            chunk = tip_points[start : start + 2048]
            d2 = np.sum((centers[:, None, :] - chunk[None, :, :]) ** 2, axis=2)
            best = np.minimum(best, np.min(d2, axis=1))
        return np.clip(np.sqrt(best) / diag, 0.0, 1.0)


def _frontier_alignment(labels: dict[str, np.ndarray], step: int, indices: np.ndarray, geom: BondGeometry) -> np.ndarray:
    tip = np.asarray(labels["crack_tip_mask"][step], dtype=bool).reshape(-1)
    edges = labels["bonds"].astype(np.int64)[indices]
    frontier = tip[edges[:, 0]] | tip[edges[:, 1]]
    if not np.any(frontier):
        return np.zeros((indices.size,), dtype=np.float64)
    mean_orientation = np.mean(geom.orientation[indices[frontier]], axis=0)
    norm = max(float(np.linalg.norm(mean_orientation)), 1e-12)
    mean_orientation = mean_orientation / norm
    return np.abs(geom.orientation[indices] @ mean_orientation)


def _recent_neighbor_event_count(labels: dict[str, np.ndarray], step: int, indices: np.ndarray, geom: BondGeometry) -> np.ndarray:
    if step <= 0:
        return np.zeros((indices.size,), dtype=np.float64)
    bonds = labels["bonds"].astype(np.int64)
    alive_prev = labels["bond_alive"][step - 1].astype(bool)
    alive_now = labels["bond_alive"][step].astype(bool)
    previous_events = alive_prev & ~alive_now
    event_idx = np.flatnonzero(previous_events)
    if event_idx.size == 0:
        return np.zeros((indices.size,), dtype=np.float64)
    edges = bonds
    node_events = np.zeros((geom.node_degree.shape[0],), dtype=np.float64)
    np.add.at(node_events, edges[event_idx, 0], 1.0)
    np.add.at(node_events, edges[event_idx, 1], 1.0)
    selected_edges = edges[indices]
    counts = node_events[selected_edges[:, 0]] + node_events[selected_edges[:, 1]]
    counts -= 2.0 * previous_events[indices].astype(np.float64)
    degree_sum = geom.node_degree[selected_edges[:, 0]] + geom.node_degree[selected_edges[:, 1]]
    return np.maximum(counts, 0.0) / np.maximum(degree_sum, 1.0)


def _local_load_vector(labels: dict[str, np.ndarray], bonds: np.ndarray, indices: np.ndarray) -> np.ndarray:
    loads = labels.get("loads")
    if loads is None:
        return np.zeros((indices.size, 2), dtype=np.float64)
    arr = np.asarray(loads, dtype=np.float64)
    selected_edges = bonds[indices]
    return 0.5 * (arr[selected_edges[:, 0]] + arr[selected_edges[:, 1]])


def ranker_feature_matrix(
    labels: dict[str, np.ndarray],
    sample: dict[str, object] | None,
    step: int,
    bond_indices: np.ndarray,
    *,
    history_trigger: np.ndarray | None = None,
    include_history: bool = True,
) -> np.ndarray:
    base = hybrid_feature_matrix(labels, step, bond_indices, history_trigger=history_trigger, include_history=include_history)
    bonds = labels["bonds"].astype(np.int64)
    indices = np.asarray(bond_indices, dtype=np.int64)
    geom, _, _ = _static(labels)
    frontier = base[:, 6]
    bond_damage = base[:, 2]
    frontier_near = np.clip(0.70 * frontier + 0.30 * bond_damage, 0.0, 1.0)
    frontier_distance = 1.0 - frontier_near
    frontier_alignment = _frontier_alignment(labels, step, indices, geom)
    neighbor = base[:, 4]
    history = base[:, 10]
    stretch = base[:, 1]
    energy = base[:, 5]
    boundary = base[:, 7]
    recent_neighbor = _recent_neighbor_event_count(labels, step, indices, geom)
    if step <= 0:
        global_recent = 0.0
    else:
        alive_prev = labels["bond_alive"][step - 1].astype(bool)
        alive_now = labels["bond_alive"][step].astype(bool)
        global_recent = float(np.count_nonzero(alive_prev & ~alive_now)) / max(float(np.count_nonzero(alive_prev)), 1.0)
    load_vec = _local_load_vector(labels, bonds, indices)
    orientation = geom.orientation[indices]
    load_along = np.sum(load_vec * orientation, axis=1)
    load_norm = np.linalg.norm(load_vec, axis=1)
    load_normal = np.sqrt(np.maximum(load_norm * load_norm - load_along * load_along, 0.0))
    # A local proxy, not geometric curvature: high when the current frontier is incident
    # and already has multiple broken/event neighbors nearby.
    frontier_curvature_proxy = frontier * np.clip(neighbor + recent_neighbor, 0.0, 1.0) * (1.0 - frontier_alignment)
    extra = np.stack(
        [
            frontier_distance,
            frontier_alignment,
            frontier_curvature_proxy,
            np.full((indices.size,), global_recent, dtype=np.float64),
            recent_neighbor,
            load_along,
            load_normal,
            stretch * frontier_near,
            energy * frontier_near,
            history * frontier_near,
            neighbor * frontier_near,
            boundary * frontier_near,
        ],
        axis=1,
    )
    regime = regime_name(sample)
    regime_one_hot = np.zeros((indices.size, len(REGIME_NAMES)), dtype=np.float64)
    if regime in REGIME_NAMES:
        regime_one_hot[:, REGIME_NAMES.index(regime)] = 1.0
    return np.nan_to_num(np.concatenate([base, extra, regime_one_hot], axis=1), nan=0.0, posinf=1e6, neginf=-1e6)


def update_history_trigger(history: np.ndarray, events: np.ndarray, bonds: np.ndarray, decay: float = 0.85) -> np.ndarray:
    out = np.asarray(history, dtype=np.float64) * float(decay)
    event_mask = np.asarray(events, dtype=bool)
    if not np.any(event_mask):
        return out
    edges = np.asarray(bonds, dtype=np.int64)
    event_nodes = np.zeros((int(np.max(edges)) + 1,), dtype=bool)
    event_nodes[edges[event_mask, 0]] = True
    event_nodes[edges[event_mask, 1]] = True
    touched = event_nodes[edges[:, 0]] | event_nodes[edges[:, 1]]
    out[touched] += 1.0
    out[event_mask] += 1.0
    return out
