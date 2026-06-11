"""Phase 1.2 matcher tests on hand-built fields (spec 1.2).

Required scenarios: merge event, loop birth, a feature that MOVES (must
match, not die+born), and a swap-prone pair where greedy is known to fail
and wasserstein must resolve it.
"""
from __future__ import annotations

import unittest

import numpy as np

from crackle.topo.cubical import superlevel_persistence
from crackle.topo.events import extract_events
from crackle.topo.matching import greedy_match, wasserstein_match

SIG_TAU = 0.08


def _events(frames: list[np.ndarray], method: str, max_dist: float = 6.0):
    diags = [superlevel_persistence(f) for f in frames]
    return extract_events(diags, sig_tau=SIG_TAU, max_dist=max_dist, method=method)


class TestMergeEvent(unittest.TestCase):
    def test_bridge_swallows_hotspot(self) -> None:
        prev = np.zeros((8, 16))
        prev[2, 2] = 0.9  # essential anchor
        prev[2, 8] = 0.5  # separate hotspot
        curr = prev.copy()
        curr[2, 3:8] = 0.6  # bridge above hotspot value: components coalesce
        for method in ("greedy", "wasserstein"):
            events = _events([prev, curr], method)
            self.assertEqual([e.kind for e in events], ["h0_died"], method)


class TestLoopBirth(unittest.TestCase):
    def test_ring_closes(self) -> None:
        open_c = np.zeros((10, 10))
        open_c[2:8, 2] = 0.6
        open_c[2:8, 7] = 0.6
        open_c[2, 2:8] = 0.6
        open_c[7, 2:8] = 0.6
        # 2-cell gap: a 1-cell gap stays 8-connected via the diagonal
        open_c[7, 2:4] = 0.0
        closed = open_c.copy()
        closed[7, 2:4] = 0.6
        for method in ("greedy", "wasserstein"):
            events = _events([open_c, closed], method)
            self.assertEqual([e.kind for e in events], ["h1_born"], method)


class TestMovingFeature(unittest.TestCase):
    def test_translation_within_gate_is_matched(self) -> None:
        prev = np.zeros((10, 12))
        prev[1, 1] = 0.9  # essential anchor
        prev[5, 5] = 0.5
        curr = np.zeros((10, 12))
        curr[1, 1] = 0.9
        curr[5, 7] = 0.5  # moved 2 cells
        for method in ("greedy", "wasserstein"):
            events = _events([prev, curr], method)
            self.assertEqual(events, [], method)


class TestSwapPronePair(unittest.TestCase):
    """A=(5,2), B=(5,8) -> A'=(5,7), B'=(5,13); gate max_dist=6.

    Pair costs: A-A'=5, B-B'=5, B-A'=1, A-B'=11 (gated out). Greedy grabs
    B-A' first, leaving A and B' unmatched (a spurious died+born pair).
    Optimal assignment pays 5+5=10 < 1+6+6=13 and matches both.
    """

    def _frames(self) -> list[np.ndarray]:
        prev = np.zeros((12, 18))
        prev[0, 0] = 0.9  # essential anchor
        prev[5, 2] = 0.5
        prev[5, 8] = 0.5
        curr = np.zeros((12, 18))
        curr[0, 0] = 0.9
        curr[5, 7] = 0.5
        curr[5, 13] = 0.5
        return [prev, curr]

    def test_greedy_fails(self) -> None:
        kinds = sorted(e.kind for e in _events(self._frames(), "greedy"))
        self.assertEqual(kinds, ["h0_born", "h0_died"])

    def test_wasserstein_resolves(self) -> None:
        self.assertEqual(_events(self._frames(), "wasserstein"), [])

    def test_raw_matchers_agree_with_field_level(self) -> None:
        a_yx = np.array([[5, 2], [5, 8]])
        b_yx = np.array([[5, 7], [5, 13]])
        vals = np.array([0.5, 0.5])
        ga, gb = greedy_match(a_yx, vals, b_yx, vals, max_dist=6.0, value_weight=4.0)
        self.assertEqual(int(ga.sum()), 1)
        self.assertEqual(int(gb.sum()), 1)
        wa, wb = wasserstein_match(
            a_yx, vals, b_yx, vals, max_dist=6.0, value_weight=4.0
        )
        self.assertTrue(wa.all())
        self.assertTrue(wb.all())


if __name__ == "__main__":
    unittest.main()
