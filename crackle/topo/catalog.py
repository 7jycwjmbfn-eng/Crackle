"""Topological event catalog + tile-based risk sets (Phase 1.3).

Catalog: one row per topological event with full provenance
(case_id, step, kind, y, x, persistence, birth_value, roi, matcher,
sig_tau, source_field_key).

Risk sets: one row per (case, step, tile). Tiles are a coarse spatial
partition (default 6x6 cells) restricted to tiles whose center row lies in
the ROI. The label for horizon H and kind K is "an event of kind K occurs
in this tile at a step in (t, t+H]". Features at time t are computed from
frames <= t and events with step <= t ONLY — causality is enforced by
construction (single forward pass) and asserted by a unit test that
perturbs future frames.

Feature groups (spec 1.3):
- local damage:   tile mean/max, mean gradient magnitude, 3-step delta
- global topology: per-step summary curves + 3-step deltas
- event history:  Hawkes-style decayed counts per kind, tile-local and
  global (decay per step, default 0.85, mirroring crackle.baselines.hawkes)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from crackle.topo.events import TopoEvent, extract_events
from crackle.topo.features import sequence_summaries
from crackle.topo.roi import horizon_margin_mask

EVENT_KINDS = ("h0_born", "h0_died", "h1_born", "h1_died")

CATALOG_COLUMNS = [
    "case_id", "step", "kind", "y", "x", "persistence", "birth_value",
    "roi", "matcher", "sig_tau", "source_field_key",
]

GLOBAL_CURVE_KEYS = (
    "n_h0_sig", "n_h1_sig", "total_pers_h0", "total_pers_h1",
    "entropy_h0", "entropy_h1",
)

LOCAL_FEATURES = ["dmg_mean", "dmg_max", "dmg_grad", "dmg_mean_d3"]
HISTORY_FEATURES = [f"hist_{k}" for k in EVENT_KINDS] + [
    f"hist_global_{k}" for k in EVENT_KINDS
]
GLOBAL_FEATURES = [c for k in GLOBAL_CURVE_KEYS for c in (k, f"{k}_d3")]
FEATURE_COLUMNS = LOCAL_FEATURES + GLOBAL_FEATURES + HISTORY_FEATURES


@dataclass
class RiskSetConfig:
    tile: int = 6
    horizons: tuple[int, ...] = (3, 5, 10)
    decay: float = 0.85
    delta_lag: int = 3
    roi_margin_k: float = 1.5
    height: float = 40.0
    horizon_mm: float = 5.2
    sig_tau: float = 0.08
    matcher: str = "wasserstein"
    connectivity: str = "8"
    kinds: tuple[str, ...] = field(default=EVENT_KINDS)


def split_of_case(
    case_id: str,
    n_notches: int,
    *,
    ood_notches: int = 4,
    train: float = 0.7,
    val: float = 0.15,
) -> str:
    """Deterministic case-level split; every ood_notches-notch case is OOD."""
    if n_notches == ood_notches:
        return "ood"
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    u = int(digest[:8], 16) / 0xFFFFFFFF
    if u < train:
        return "train"
    if u < train + val:
        return "val"
    return "test"


def case_events_and_curves(
    movie: np.ndarray, *, config: RiskSetConfig
) -> tuple[list[TopoEvent], dict[str, np.ndarray], np.ndarray]:
    """TDA pass over one movie. Returns (events, curves, roi_mask)."""
    ny, nx = movie.shape[1], movie.shape[2]
    roi = (
        horizon_margin_mask(
            ny, nx, height=config.height, horizon=config.horizon_mm,
            k=config.roi_margin_k,
        )
        if config.roi_margin_k > 0
        else np.ones((ny, nx), dtype=bool)
    )
    rows, diagrams = sequence_summaries(
        movie, sig_tau=config.sig_tau, connectivity=config.connectivity, roi=roi
    )
    events = extract_events(
        diagrams, sig_tau=config.sig_tau, roi=roi, method=config.matcher
    )
    curves = {k: np.array([r[k] for r in rows], dtype=np.float64)
              for k in GLOBAL_CURVE_KEYS}
    return events, curves, roi


def catalog_rows(
    events: list[TopoEvent],
    *,
    case_id: str,
    config: RiskSetConfig,
    source_field_key: str,
) -> list[dict]:
    return [
        {
            "case_id": case_id,
            **e.as_row(),
            "roi": config.roi_margin_k,
            "matcher": config.matcher,
            "sig_tau": config.sig_tau,
            "source_field_key": source_field_key,
        }
        for e in events
    ]


def _tile_grid(ny: int, nx: int, roi: np.ndarray, tile: int) -> list[tuple[int, int]]:
    """Tile indices (ty, tx) whose center row is inside the ROI."""
    out = []
    for ty in range(int(np.ceil(ny / tile))):
        cy = min(ty * tile + tile // 2, ny - 1)
        for tx in range(int(np.ceil(nx / tile))):
            cx = min(tx * tile + tile // 2, nx - 1)
            if roi[cy, cx]:
                out.append((ty, tx))
    return out


def riskset_rows(
    movie: np.ndarray,
    events: list[TopoEvent],
    curves: dict[str, np.ndarray],
    roi: np.ndarray,
    *,
    case_id: str,
    config: RiskSetConfig,
    label_events: list[TopoEvent] | None = None,
) -> list[dict]:
    """One forward pass; emits rows for every (step, in-ROI tile).

    Features at step t see frames <= t and events with step <= t.
    Labels for horizon H see events with step in (t, t+H]; rows where
    t + H exceeds the movie are right-censored and get label -1.

    label_events (optional): when given, LABELS are built from these
    ground-truth events while all FEATURES (damage stats, global curves,
    decayed event history) still come from `events`/`movie`/`curves`.
    This is the noisy-observation / clean-target setup: you forecast the
    true failure events from a noisy observation of the field. When None,
    labels and features share `events` (the standard clean build).
    """
    t_steps, ny, nx = movie.shape
    tile = config.tile
    tiles = _tile_grid(ny, nx, roi, tile)
    tile_of = {t: i for i, t in enumerate(tiles)}
    lag = config.delta_lag

    # events bucketed by step for history updates (observed features)
    ev_by_step: dict[int, list[TopoEvent]] = {}
    for e in events:
        ev_by_step.setdefault(e.step, []).append(e)

    def tile_idx(e: TopoEvent) -> int | None:
        return tile_of.get((e.y // tile, e.x // tile))

    # label lookup: kind -> (step, tile_index) occurrence matrix, built from
    # the ground-truth events (label_events if provided, else `events`)
    occ = {k: np.zeros((t_steps, len(tiles)), dtype=bool) for k in config.kinds}
    for e in (label_events if label_events is not None else events):
        ti = tile_idx(e)
        if ti is not None and e.kind in occ and 0 <= e.step < t_steps:
            occ[e.kind][e.step, ti] = True

    hist_tile = {k: np.zeros((len(tiles),), dtype=np.float64) for k in config.kinds}
    hist_global = {k: 0.0 for k in config.kinds}

    rows: list[dict] = []
    for t in range(t_steps):
        # update history with events observed AT t (transition (t-1)->t)
        if t > 0:
            for k in config.kinds:
                hist_tile[k] *= config.decay
                hist_global[k] *= config.decay
            for e in ev_by_step.get(t, []):
                if e.kind not in hist_global:
                    continue
                hist_global[e.kind] += 1.0
                ti = tile_idx(e)
                if ti is not None:
                    hist_tile[e.kind][ti] += 1.0

        frame = movie[t]
        gy, gx = np.gradient(frame)
        grad_mag = np.hypot(gy, gx)
        frame_lag = movie[max(t - lag, 0)]

        glob: dict[str, float] = {}
        for key in GLOBAL_CURVE_KEYS:
            glob[key] = float(curves[key][t])
            glob[f"{key}_d3"] = float(curves[key][t] - curves[key][max(t - lag, 0)])

        for ti, (ty, tx) in enumerate(tiles):
            ys, xs = ty * tile, tx * tile
            patch = frame[ys : ys + tile, xs : xs + tile]
            row: dict = {
                "case_id": case_id, "step": t, "tile_y": ty, "tile_x": tx,
                "dmg_mean": float(patch.mean()),
                "dmg_max": float(patch.max()),
                "dmg_grad": float(grad_mag[ys : ys + tile, xs : xs + tile].mean()),
                "dmg_mean_d3": float(
                    patch.mean() - frame_lag[ys : ys + tile, xs : xs + tile].mean()
                ),
                **glob,
            }
            for k in config.kinds:
                row[f"hist_{k}"] = float(hist_tile[k][ti])
                row[f"hist_global_{k}"] = float(hist_global[k])
            for h in config.horizons:
                for k in config.kinds:
                    if t + h >= t_steps:
                        label = -1  # right-censored
                    else:
                        label = int(occ[k][t + 1 : t + h + 1, ti].any())
                    row[f"label_{k}_H{h}"] = label
                row[f"label_any_H{h}"] = (
                    -1
                    if t + h >= t_steps
                    else int(
                        any(occ[k][t + 1 : t + h + 1, ti].any() for k in config.kinds)
                    )
                )
            rows.append(row)
    return rows
