from __future__ import annotations

import numpy as np

from crackle.data.features import feature_matrix


DETERMINISTIC_MODELS = {
    "deterministic_threshold_geometry_v1a",
    "deterministic_threshold_energy_v1b",
    "deterministic_event_ranker_v1",
}


def deterministic_logits(model: str, labels: dict[str, np.ndarray], step: int, bond_indices: np.ndarray) -> np.ndarray:
    if model not in DETERMINISTIC_MODELS:
        raise ValueError(f"unknown deterministic model: {model}")
    x = feature_matrix(labels, step, bond_indices, include_history=False)
    stretch_ratio = x[:, 1]
    bond_damage = x[:, 2]
    weakness = x[:, 3]
    neighbor = x[:, 4]
    energy = x[:, 5]
    frontier = x[:, 6]
    boundary = x[:, 7]
    if model == "deterministic_threshold_geometry_v1a":
        return 8.0 * (stretch_ratio - 0.95) + 1.2 * bond_damage + 0.8 * neighbor + 0.7 * frontier - 0.15 * boundary
    if model == "deterministic_event_ranker_v1":
        # Ranker-only causal score. Its probabilities are not calibrated and
        # should be used for event localization gates, not probability gates.
        return 10.0 * stretch_ratio + 2.4 * energy + 1.2 * weakness + 1.0 * neighbor + 0.9 * frontier - 0.2 * boundary
    return 8.0 * (stretch_ratio - 0.92) + 1.0 * bond_damage + 1.0 * neighbor + 1.4 * energy + 0.4 * weakness + 0.6 * frontier


def deterministic_model_payload(model: str) -> dict[str, object]:
    if model not in DETERMINISTIC_MODELS:
        raise ValueError(f"unknown deterministic model: {model}")
    return {
        "schema": "crackle_model_v1",
        "model": model,
        "kind": "deterministic",
        "uses_future_labels": False,
        "oracle_row": False,
        "feature_names": ["handcoded_causal_mechanics"],
        "probability_calibrated": False,
    }
