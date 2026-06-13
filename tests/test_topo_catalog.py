"""Phase 1.3 catalog/risk-set tests: causality, labels, history, splits."""
from __future__ import annotations

import unittest

import numpy as np

from crackle.topo.catalog import (
    FEATURE_COLUMNS,
    GLOBAL_CURVE_KEYS,
    RiskSetConfig,
    case_events_and_curves,
    riskset_rows,
    split_of_case,
)
from crackle.topo.events import TopoEvent


def _movie(rng: np.random.Generator, t_steps: int = 12, ny: int = 20,
           nx: int = 24) -> np.ndarray:
    """Deterministic growing-bump movie with enough topology to emit events."""
    movie = np.zeros((t_steps, ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    centers = rng.uniform([4, 4], [ny - 4, nx - 4], size=(4, 2))
    for t in range(t_steps):
        f = np.zeros((ny, nx))
        for k, (cy, cx) in enumerate(centers):
            onset = 2 + 2 * k
            if t >= onset:
                amp = min(0.15 * (t - onset + 1), 0.9)
                f += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0))
        movie[t] = np.clip(f, 0.0, 1.0)
    return movie


class TestCausality(unittest.TestCase):
    def test_features_ignore_future_frames(self) -> None:
        rng = np.random.default_rng(7)
        movie = _movie(rng)
        config = RiskSetConfig(horizons=(2, 3))
        t0 = 6

        def rows_until_t0(m: np.ndarray) -> list[dict]:
            events, curves, roi = case_events_and_curves(m, config=config)
            rows = riskset_rows(m, events, curves, roi, case_id="c", config=config)
            return [r for r in rows if r["step"] <= t0]

        base = rows_until_t0(movie)
        perturbed_movie = movie.copy()
        perturbed_movie[t0 + 1 :] = rng.uniform(0.0, 1.0,
                                                size=movie[t0 + 1 :].shape)
        perturbed = rows_until_t0(perturbed_movie)

        self.assertEqual(len(base), len(perturbed))
        self.assertGreater(len(base), 0)
        for a, b in zip(base, perturbed):
            for col in ("case_id", "step", "tile_y", "tile_x", *FEATURE_COLUMNS):
                self.assertEqual(a[col], b[col],
                                 f"feature {col} leaked future at step {a['step']}")


class TestLabelsAndHistory(unittest.TestCase):
    def _rows(self):
        t_steps, ny, nx = 8, 12, 12
        movie = np.zeros((t_steps, ny, nx))
        roi = np.ones((ny, nx), dtype=bool)
        curves = {k: np.zeros((t_steps,)) for k in GLOBAL_CURVE_KEYS}
        events = [TopoEvent(step=5, kind="h0_born", y=2, x=2,
                            persistence=0.5, birth_value=0.5)]
        config = RiskSetConfig(tile=6, horizons=(3,), decay=0.5)
        rows = riskset_rows(movie, events, curves, roi, case_id="c",
                            config=config)
        return {(r["step"], r["tile_y"], r["tile_x"]): r for r in rows}

    def test_label_window_is_t_plus_1_to_t_plus_h(self) -> None:
        rows = self._rows()
        # event at step 5, tile (0,0); H=3 -> positive for t in {2,3,4}
        self.assertEqual(rows[(1, 0, 0)]["label_h0_born_H3"], 0)
        for t in (2, 3, 4):
            self.assertEqual(rows[(t, 0, 0)]["label_h0_born_H3"], 1, t)
        # other tile never positive
        self.assertEqual(rows[(3, 0, 1)]["label_h0_born_H3"], 0)

    def test_censoring(self) -> None:
        rows = self._rows()
        for t in (5, 6, 7):  # t + 3 >= 8 steps
            self.assertEqual(rows[(t, 0, 0)]["label_h0_born_H3"], -1, t)

    def test_decayed_history(self) -> None:
        rows = self._rows()
        self.assertEqual(rows[(4, 0, 0)]["hist_h0_born"], 0.0)
        self.assertEqual(rows[(5, 0, 0)]["hist_h0_born"], 1.0)
        self.assertEqual(rows[(6, 0, 0)]["hist_h0_born"], 0.5)
        self.assertEqual(rows[(7, 0, 1)]["hist_h0_born"], 0.0)  # tile-local
        self.assertEqual(rows[(7, 0, 1)]["hist_global_h0_born"], 0.25)


class TestNoisyLabelsAreGroundTruth(unittest.TestCase):
    """Noisy-feature build must keep CLEAN labels (label_events override)."""

    def test_labels_clean_features_noisy(self) -> None:
        from crackle.topo.events import extract_events
        from crackle.topo.features import sequence_summaries
        from crackle.topo.noise import add_measurement_noise
        from crackle.topo.roi import horizon_margin_mask

        rng = np.random.default_rng(3)
        movie = _movie(rng, t_steps=12, ny=18, nx=24)
        config = RiskSetConfig(horizons=(3,), roi_margin_k=0.0)
        clean_events, clean_curves, roi = case_events_and_curves(
            movie, config=config)
        ny, nx = movie.shape[1], movie.shape[2]
        roi = np.ones((ny, nx), dtype=bool)

        clean_rows = riskset_rows(movie, clean_events, clean_curves, roi,
                                  case_id="c", config=config)
        noisy = add_measurement_noise(movie, sigma=0.05, seed=1)
        n_events, n_curves, _ = case_events_and_curves(noisy, config=config)
        noisy_rows = riskset_rows(noisy, n_events, n_curves, roi, case_id="c",
                                  config=config, label_events=clean_events)

        label_cols = [c for c in clean_rows[0] if c.startswith("label_")]
        feat_any_diff = False
        for a, b in zip(clean_rows, noisy_rows):
            for lc in label_cols:               # labels identical
                self.assertEqual(a[lc], b[lc], lc)
            if a["dmg_mean"] != b["dmg_mean"]:  # features differ
                feat_any_diff = True
        self.assertTrue(feat_any_diff, "noisy features should differ")


class TestSplits(unittest.TestCase):
    def test_deterministic_and_ood(self) -> None:
        self.assertEqual(split_of_case("case_000001", 2),
                         split_of_case("case_000001", 2))
        self.assertEqual(split_of_case("case_000001", 4), "ood")
        splits = {split_of_case(f"case_{i:06d}", 1) for i in range(200)}
        self.assertEqual(splits, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
