"""Phase 2.1 noise-model tests: determinism, identity, target std, clipping."""
from __future__ import annotations

import unittest

import numpy as np

from crackle.topo.noise import add_measurement_noise


class TestMeasurementNoise(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        # mid-range field so clipping does not dominate the std measurement
        self.movie = np.full((4, 20, 24), 0.5) + 0.05 * rng.standard_normal(
            (4, 20, 24))
        self.movie = np.clip(self.movie, 0.0, 1.0)

    def test_sigma_zero_is_identity(self) -> None:
        out = add_measurement_noise(self.movie, sigma=0.0, seed=1)
        np.testing.assert_array_equal(out, self.movie)

    def test_deterministic_given_seed(self) -> None:
        a = add_measurement_noise(self.movie, sigma=0.05, seed=7)
        b = add_measurement_noise(self.movie, sigma=0.05, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = add_measurement_noise(self.movie, sigma=0.05, seed=7)
        b = add_measurement_noise(self.movie, sigma=0.05, seed=8)
        self.assertGreater(np.abs(a - b).max(), 0.0)

    def test_does_not_mutate_input(self) -> None:
        ref = self.movie.copy()
        add_measurement_noise(self.movie, sigma=0.1, seed=1)
        np.testing.assert_array_equal(self.movie, ref)

    def test_output_in_range(self) -> None:
        out = add_measurement_noise(self.movie, sigma=0.2, seed=3)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_target_std_iid(self) -> None:
        # away from the clip boundaries, realized per-cell std ~ sigma
        base = np.full((6, 40, 40), 0.5)
        out = add_measurement_noise(base, sigma=0.05, corr_cells=0.0, seed=2)
        realized = float((out - base).std())
        self.assertAlmostEqual(realized, 0.05, delta=0.005)

    def test_correlated_has_target_std(self) -> None:
        base = np.full((6, 40, 40), 0.5)
        out = add_measurement_noise(base, sigma=0.05, corr_cells=1.5, seed=2)
        realized = float((out - base).std())
        self.assertAlmostEqual(realized, 0.05, delta=0.006)


if __name__ == "__main__":
    unittest.main()
