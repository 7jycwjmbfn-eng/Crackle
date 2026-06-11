"""Phase 1.1 ROI mask tests: mask geometry + diagram-level filtering."""
from __future__ import annotations

import unittest

import numpy as np

from crackle.topo.cubical import Diagram, superlevel_persistence
from crackle.topo.events import extract_events
from crackle.topo.features import frame_summary
from crackle.topo.roi import apply_roi, boundary_margin_mask, horizon_margin_mask


def _diagram(birth_yx: list[tuple[int, int]], essential: list[bool]) -> Diagram:
    n = len(birth_yx)
    yx = np.asarray(birth_yx, dtype=np.int64)
    return Diagram(
        dim=np.zeros((n,), dtype=np.int64),
        birth=np.full((n,), 0.5),
        death=np.zeros((n,)),
        birth_yx=yx,
        death_yx=yx.copy(),
        essential=np.asarray(essential, dtype=bool),
        field_min=0.0,
        field_max=0.5,
    )


class TestMaskGeometry(unittest.TestCase):
    def test_tuple_margin_rows_only(self) -> None:
        mask = boundary_margin_mask(29, 48, margin_cells=(6, 0))
        self.assertFalse(mask[:6].any())
        self.assertFalse(mask[23:].any())
        self.assertTrue(mask[6:23].all())

    def test_int_margin_all_edges(self) -> None:
        mask = boundary_margin_mask(10, 10, margin_cells=2)
        self.assertTrue(mask[2:8, 2:8].all())
        self.assertEqual(int(mask.sum()), 36)

    def test_margin_too_large_raises(self) -> None:
        with self.assertRaises(ValueError):
            boundary_margin_mask(10, 10, margin_cells=5)

    def test_horizon_variant_matches_cell_math(self) -> None:
        # ny=29 over height=40 -> dy=40/28; 1.5*5.2/dy = 5.46 -> 6 rows
        mask = horizon_margin_mask(29, 48, height=40.0, horizon=5.2, k=1.5)
        expected = boundary_margin_mask(29, 48, margin_cells=(6, 0))
        np.testing.assert_array_equal(mask, expected)


class TestApplyRoi(unittest.TestCase):
    def test_filters_by_birth_location_keeps_essential(self) -> None:
        roi = boundary_margin_mask(29, 48, margin_cells=(6, 0))
        diag = _diagram(
            [(2, 10), (15, 10), (27, 10)],  # boundary, interior, boundary
            [True, False, False],  # first is essential (and in the margin)
        )
        kept = apply_roi(diag, roi)
        self.assertEqual(kept.dim.size, 2)
        self.assertTrue(kept.essential[0])  # essential survives in the margin
        self.assertEqual(tuple(kept.birth_yx[1]), (15, 10))

    def test_none_roi_is_identity(self) -> None:
        diag = _diagram([(2, 10)], [False])
        self.assertIs(apply_roi(diag, None), diag)


class TestFieldLevelFiltering(unittest.TestCase):
    """A bump in the loading band vanishes from summaries and events."""

    def _field(self, with_margin_bump: bool) -> np.ndarray:
        f = np.zeros((29, 48))
        f[14, 24] = 0.9  # interior anchor (essential)
        f[15, 10] = 0.5  # interior feature
        if with_margin_bump:
            f[2, 30] = 0.5  # loading-band artifact
        return f

    def test_summary_counts_exclude_margin_features(self) -> None:
        roi = boundary_margin_mask(29, 48, margin_cells=(6, 0))
        diag = superlevel_persistence(self._field(True))
        no_roi = frame_summary(diag, sig_tau=0.08)
        with_roi = frame_summary(apply_roi(diag, roi), sig_tau=0.08)
        self.assertEqual(no_roi["n_h0_sig"], 3.0)  # essential + 2 bumps
        self.assertEqual(with_roi["n_h0_sig"], 2.0)

    def test_margin_birth_produces_no_event_with_roi(self) -> None:
        roi = boundary_margin_mask(29, 48, margin_cells=(6, 0))
        frames = np.stack([self._field(False), self._field(True)])
        diags = [superlevel_persistence(fr) for fr in frames]
        without = extract_events(diags, sig_tau=0.08)
        with_roi = extract_events(diags, sig_tau=0.08, roi=roi)
        self.assertEqual([e.kind for e in without], ["h0_born"])
        self.assertEqual(with_roi, [])


if __name__ == "__main__":
    unittest.main()
