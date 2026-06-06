import tempfile
import unittest
from pathlib import Path

import numpy as np

from crackle.calibration import fit_intensity_calibrators, weighted_binary_metrics
from crackle.baselines.deterministic import deterministic_logits
from crackle.data.event_catalog import bond_break_events
from crackle.data.features import REGIME_NAMES, RANKER_FEATURE_NAMES, feature_matrix, ranker_feature_matrix
from crackle.data.riskset import sample_riskset_summary


def tiny_labels():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    bonds = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    alive = np.array(
        [
            [True, True, True],
            [True, False, True],
            [True, False, False],
        ],
        dtype=bool,
    )
    stretch = np.array(
        [
            [0.1, 0.2, 0.1],
            [0.2, 1.2, 0.3],
            [0.3, 1.3, 1.4],
        ],
        dtype=np.float64,
    )
    damage = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.5, 1.0, 0.5],
        ],
        dtype=np.float64,
    )
    energy = np.array(
        [
            [0.0, 0.1, 0.1, 0.0],
            [0.0, 0.4, 0.5, 0.1],
            [0.0, 0.4, 0.9, 0.6],
        ],
        dtype=np.float64,
    )
    return {
        "reference_x": points,
        "bonds": bonds,
        "bond_alive": alive,
        "bond_stretch": stretch,
        "damage": damage,
        "strain_energy": energy,
        "crack_tip_mask": damage > 0.0,
        "material_toughness": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "loads": np.zeros((4, 2), dtype=np.float64),
    }


class CrackleContractTest(unittest.TestCase):
    def test_event_count_matches_bond_alive_drop(self):
        sample = {"id": "tiny", "benchmark_case_id": "unit", "split": "train", "time_step_dt": 0.5}
        labels = tiny_labels()
        events = bond_break_events(sample, labels)
        expected = int(np.count_nonzero(labels["bond_alive"][:-1] & ~labels["bond_alive"][1:]))
        self.assertEqual(len(events), expected)
        self.assertEqual([event["bond_id"] for event in events], [1, 2])

    def test_shuffle_future_labels_does_not_change_features(self):
        labels = tiny_labels()
        baseline = feature_matrix(labels, 0, np.array([0, 1, 2], dtype=np.int64))
        shuffled = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in labels.items()}
        shuffled["bond_alive"][1:] = shuffled["bond_alive"][1:][::-1]
        shuffled["damage"][1:] = shuffled["damage"][1:][::-1]
        changed_future = feature_matrix(shuffled, 0, np.array([0, 1, 2], dtype=np.int64))
        np.testing.assert_allclose(baseline, changed_future)

    def test_ranker_features_are_causal_and_include_regime(self):
        labels = tiny_labels()
        sample = {"benchmark_case_id": "hole_plus_notch"}
        indices = np.array([0, 1, 2], dtype=np.int64)
        baseline = ranker_feature_matrix(labels, sample, 0, indices)
        shuffled = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in labels.items()}
        shuffled["bond_alive"][1:] = shuffled["bond_alive"][1:][::-1]
        shuffled["bond_stretch"][1:] = shuffled["bond_stretch"][1:][::-1]
        shuffled["damage"][1:] = shuffled["damage"][1:][::-1]
        shuffled["strain_energy"][1:] = shuffled["strain_energy"][1:][::-1]
        changed_future = ranker_feature_matrix(shuffled, sample, 0, indices)
        np.testing.assert_allclose(baseline, changed_future)
        self.assertEqual(baseline.shape[1], len(RANKER_FEATURE_NAMES))
        regime_start = len(RANKER_FEATURE_NAMES) - len(REGIME_NAMES)
        one_hot = baseline[:, regime_start:]
        self.assertTrue(np.all(one_hot[:, REGIME_NAMES.index("hole_plus_notch")] == 1.0))
        self.assertTrue(np.all(np.sum(one_hot, axis=1) == 1.0))

    def test_deterministic_event_ranker_does_not_read_future_labels(self):
        labels = tiny_labels()
        baseline = deterministic_logits("deterministic_event_ranker_v1", labels, 0, np.array([0, 1, 2], dtype=np.int64))
        shuffled = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in labels.items()}
        shuffled["bond_alive"][1:] = shuffled["bond_alive"][1:][::-1]
        shuffled["bond_stretch"][1:] = shuffled["bond_stretch"][1:][::-1]
        changed_future = deterministic_logits("deterministic_event_ranker_v1", shuffled, 0, np.array([0, 1, 2], dtype=np.int64))
        np.testing.assert_allclose(baseline, changed_future)

    def test_riskset_contains_positive_and_censored_bonds(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = {"id": "tiny", "benchmark_case_id": "unit", "split": "train", "crack_npz": "unused"}
            row = sample_riskset_summary(sample, tiny_labels(), Path(tmp), partial_observation_rate=0.7)
            self.assertGreater(row["positive_event_count"], 0)
            self.assertGreater(row["censored_bond_count"], 0)
            self.assertTrue(Path(row["riskset_static_npz"]).exists())

    def test_intercept_calibration_improves_weighted_nll(self):
        prob = np.array([0.35, 0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)
        target = np.array([1, 0, 0, 0, 0, 0], dtype=np.float64)
        weight = np.array([1, 10, 10, 10, 10, 10], dtype=np.float64)
        before = weighted_binary_metrics(prob, target, weight)["NLL"]
        calibrator, _ = fit_intensity_calibrators(prob, target, weight=weight)
        after = weighted_binary_metrics(calibrator.apply(prob), target, weight)["NLL"]
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
