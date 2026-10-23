#!/usr/bin/env python3
"""Regression tests for ventricular ground-truth label canonicalization."""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - local host may lack runtime deps
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from frame_manifest import _architecture_family, _remap_cardiac_labels  # noqa: E402


def _synthetic_sax(labels):
    """Build a simple SAX-like slice stack with RV, MYO ring, and LV cavity."""
    rv_label, myo_label, lv_label = labels
    arr = np.zeros((3, 64, 64), dtype=np.int16)
    yy, xx = np.ogrid[:64, :64]

    lv = (yy - 32) ** 2 + (xx - 38) ** 2 <= 8**2
    myo_outer = (yy - 32) ** 2 + (xx - 38) ** 2 <= 12**2
    myo = myo_outer & ~lv
    rv = ((yy - 31) / 13) ** 2 + ((xx - 22) / 8) ** 2 <= 1

    arr[1][rv] = rv_label
    arr[1][myo] = myo_label
    arr[1][lv] = lv_label
    return arr


class FrameManifestLabelTests(unittest.TestCase):
    def test_architecture_family_treats_models_as_ventricular(self):
        for arch in ("vsa3l", "nnformer", "cinema", "vsa3l__model", ""):
            self.assertEqual(_architecture_family(arch), "ventricular")
        self.assertEqual(_architecture_family("atrial_nnunet"), "atrial")

    def test_acdc_labels_remain_canonical(self):
        if np is None:
            self.skipTest("numpy is not available in this Python runtime")
        raw = _synthetic_sax((1, 2, 3))
        canonical = _remap_cardiac_labels(raw, np, architecture="vsa3l")
        self.assertTrue(np.array_equal(canonical, raw))

    def test_mms_labels_are_swapped_to_sorat_convention(self):
        if np is None:
            self.skipTest("numpy is not available in this Python runtime")
        raw = _synthetic_sax((3, 2, 1))  # raw M&Ms convention: 1=LV, 2=MYO, 3=RV
        canonical = _remap_cardiac_labels(raw, np, architecture="nnformer")
        expected = _synthetic_sax((1, 2, 3))
        self.assertTrue(np.array_equal(canonical, expected))

    def test_nonstandard_label_values_are_canonicalized_by_anatomy(self):
        if np is None:
            self.skipTest("numpy is not available in this Python runtime")
        raw = _synthetic_sax((30, 20, 10))
        canonical = _remap_cardiac_labels(raw, np, architecture="cinema")
        expected = _synthetic_sax((1, 2, 3))
        self.assertTrue(np.array_equal(canonical, expected))

    def test_atrial_architecture_bypasses_ventricular_remap(self):
        if np is None:
            self.skipTest("numpy is not available in this Python runtime")
        raw = _synthetic_sax((3, 2, 1))
        canonical = _remap_cardiac_labels(raw, np, architecture="atrial_nnunet")
        self.assertTrue(np.array_equal(canonical, raw))


if __name__ == "__main__":
    unittest.main()
