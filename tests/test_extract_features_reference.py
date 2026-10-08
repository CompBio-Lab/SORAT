#!/usr/bin/env python3
"""Tests for the blood-pool intensity reference in extract_features.py."""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover - runtime container supplies these
    np = None
    sitk = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None and sitk is not None:
    from extract_features import (  # noqa: E402
        LABEL_LV,
        LABEL_MYO,
        build_radiomics_settings,
        normalize_to_reference,
    )


@unittest.skipIf(np is None or sitk is None, "numpy/SimpleITK not available")
class IntensityReferenceTests(unittest.TestCase):
    def _case(self, blood=400.0, myo=150.0, gain=1.0):
        mask = np.zeros((3, 20, 20), dtype=np.uint8)
        mask[:, 6:14, 6:14] = LABEL_MYO
        mask[:, 8:12, 8:12] = LABEL_LV
        image = np.full(mask.shape, 50.0, dtype=np.float32)
        image[mask == LABEL_MYO] = myo
        image[mask == LABEL_LV] = blood
        img = sitk.GetImageFromArray(image * gain)
        img.SetSpacing((1.5, 1.5, 8.0))
        return img, mask

    def test_blood_pool_maps_to_scale_and_keeps_geometry(self):
        img, mask = self._case()
        out = normalize_to_reference(img, mask, "lv_bloodpool", 100.0)
        arr = sitk.GetArrayFromImage(out)
        self.assertAlmostEqual(float(arr[mask == LABEL_LV].mean()), 100.0, places=4)
        self.assertAlmostEqual(float(arr[mask == LABEL_MYO].mean()), 37.5, places=4)
        self.assertEqual(out.GetSpacing(), img.GetSpacing())

    def test_scanner_gain_cancels_out(self):
        img_a, mask = self._case(gain=1.0)
        img_b, _ = self._case(gain=3.7)
        a = sitk.GetArrayFromImage(normalize_to_reference(img_a, mask))
        b = sitk.GetArrayFromImage(normalize_to_reference(img_b, mask))
        np.testing.assert_allclose(a, b, rtol=1e-5)

    def test_empty_blood_pool_raises(self):
        img, mask = self._case()
        mask[mask == LABEL_LV] = LABEL_MYO
        with self.assertRaises(ValueError):
            normalize_to_reference(img, mask)

    def test_settings_record_reference_and_reject_conflicts(self):
        settings = build_radiomics_settings(bin_count=32, intensity_reference="lv_bloodpool")
        self.assertEqual(settings["intensityReference"], "lv_bloodpool")
        self.assertEqual(settings["intensityReferenceScale"], 100.0)
        with self.assertRaises(ValueError):
            build_radiomics_settings(normalize=True, intensity_reference="lv_bloodpool")
        with self.assertRaises(ValueError):
            build_radiomics_settings(intensity_reference="whole_image")


if __name__ == "__main__":
    unittest.main()
