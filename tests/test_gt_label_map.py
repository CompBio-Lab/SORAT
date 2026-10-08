#!/usr/bin/env python3
"""Explicit ground-truth label maps override the anatomical inference."""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - local host may lack runtime deps
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from frame_manifest import parse_gt_label_map  # noqa: E402

if np is not None:
    from frame_manifest import _remap_cardiac_labels  # noqa: E402


class ParseGtLabelMapTests(unittest.TestCase):
    def test_auto_means_inference(self):
        for spec in (None, "", "auto", "AUTO"):
            self.assertIsNone(parse_gt_label_map(spec))

    def test_acdc_and_mms_maps(self):
        self.assertEqual(parse_gt_label_map("rv=1,myo=2,lv=3"), {1: 1, 2: 2, 3: 3})
        self.assertEqual(parse_gt_label_map("lv=1, myo=2, rv=3"), {1: 3, 2: 2, 3: 1})

    def test_invalid_maps_are_rejected(self):
        for spec in ("rv=1,myo=2", "rv=1,rv=2,lv=3", "ra=1,myo=2,lv=3", "rv:1,myo=2,lv=3"):
            with self.assertRaises(ValueError):
                parse_gt_label_map(spec)


@unittest.skipIf(np is None, "numpy is required")
class ExplicitRemapTests(unittest.TestCase):
    def _ring(self):
        # One slice: label 3 encloses label 1 (an M&Ms-style LV inside MYO),
        # label 2 sits outside. Anatomy alone would call 3 the myocardium.
        data = np.zeros((1, 12, 12), dtype=np.int32)
        data[0, 2:9, 2:9] = 3
        data[0, 4:7, 4:7] = 1
        data[0, 2:9, 9:11] = 2
        return data

    def test_explicit_map_is_applied_verbatim(self):
        data = self._ring()
        out = _remap_cardiac_labels(data, np, "nnformer", label_map={1: 3, 3: 2, 2: 1})
        self.assertTrue((out[data == 1] == 3).all())
        self.assertTrue((out[data == 3] == 2).all())
        self.assertTrue((out[data == 2] == 1).all())

    def test_unmapped_label_raises(self):
        data = self._ring()
        data[0, 0, 0] = 5
        with self.assertRaises(ValueError):
            _remap_cardiac_labels(data, np, "nnformer", label_map={1: 3, 2: 2, 3: 1})

    def test_atrial_labels_ignore_ventricular_map(self):
        data = self._ring()
        out = _remap_cardiac_labels(data, np, "atrial_nnunet", label_map={1: 3, 2: 2, 3: 1})
        self.assertTrue(np.array_equal(out, data))


if __name__ == "__main__":
    unittest.main()
