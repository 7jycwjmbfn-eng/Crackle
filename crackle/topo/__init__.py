"""Topological data analysis branch for Crackle (Phase 0).

Turns damage-field movies into persistence diagrams, scalar precursor
curves, and a discrete topological event stream compatible with the
event-forecasting framing of the rest of the archive.
"""
from crackle.topo.cubical import Diagram, backend, betti_at_level, superlevel_persistence
from crackle.topo.events import TopoEvent, event_count_curves, extract_events
from crackle.topo.features import frame_summary, persistence_entropy, sequence_summaries
from crackle.topo.instability import instability_step, lead_time_table, onset_step
from crackle.topo.roi import apply_roi, boundary_margin_mask, horizon_margin_mask
from crackle.topo.io import flat_to_fields, infer_grid_shape, load_case_npz

__all__ = [
    "Diagram", "backend", "betti_at_level", "superlevel_persistence",
    "TopoEvent", "event_count_curves", "extract_events",
    "frame_summary", "persistence_entropy", "sequence_summaries",
    "instability_step", "lead_time_table", "onset_step",
    "apply_roi", "boundary_margin_mask", "horizon_margin_mask",
    "flat_to_fields", "infer_grid_shape", "load_case_npz",
]
