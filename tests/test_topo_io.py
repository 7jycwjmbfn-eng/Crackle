"""Tests for the real-data mask-sequence loader (Phase 3)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from crackle.topo.io import load_mask_sequence, natural_key


class TestNaturalKey(unittest.TestCase):
    def test_numeric_order(self) -> None:
        files = ["f_10.png", "f_2.png", "f_1.png", "f_21.png"]
        ordered = sorted(files, key=natural_key)
        self.assertEqual(ordered, ["f_1.png", "f_2.png", "f_10.png", "f_21.png"])


class TestLoadMaskSequence(unittest.TestCase):
    def _write(self, d: Path, name: str, arr: np.ndarray) -> Path:
        from PIL import Image

        p = d / name
        Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(p)
        return p

    def test_load_binarize_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a1 = np.zeros((8, 12)); a1[2, 2] = 1.0
            a2 = np.zeros((8, 12)); a2[2, 2:5] = 1.0
            # write out of order to exercise natural_key sorting
            self._write(d, "frame_10.png", a2)
            self._write(d, "frame_2.png", a1)
            paths = sorted(d.glob("*.png"), key=natural_key)
            movie = load_mask_sequence(paths, threshold=0.5)
            self.assertEqual(movie.shape, (2, 8, 12))
            self.assertEqual(movie[0, 2, 2], 1.0)            # frame_2 first
            self.assertEqual(int(movie[1].sum()), 3)         # frame_10 has 3px
            self.assertTrue(set(np.unique(movie)).issubset({0.0, 1.0}))

    def test_downscale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            p = self._write(d, "m.png", np.ones((8, 8)))
            movie = load_mask_sequence([p], threshold=None, downscale=2)
            self.assertEqual(movie.shape, (1, 4, 4))

    def test_shape_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            p1 = self._write(d, "a.png", np.zeros((8, 8)))
            p2 = self._write(d, "b.png", np.zeros((8, 10)))
            with self.assertRaises(ValueError):
                load_mask_sequence([p1, p2])


if __name__ == "__main__":
    unittest.main()
