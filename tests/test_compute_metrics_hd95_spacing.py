#!/usr/bin/env python3
"""HD95 must use physical spacing in array axis order on anisotropic stacks."""

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
    from medpy.metric import binary  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - local host may lack runtime deps
    np = None
    sitk = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None and sitk is not None:
    from compute_metrics import (  # noqa: E402
        VENTRICULAR_LABELS,
        array_axis_spacing,
        compute_metrics_for_volume,
    )

SPACING_XYZ = (1.5, 1.5, 10.0)  # typical cine SAX: fine in-plane, 10 mm slices


def _write(arr, path):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))  # array is (z, y, x)
    img.SetSpacing(SPACING_XYZ)
    sitk.WriteImage(img, str(path))


@unittest.skipIf(np is None or sitk is None, "numpy, SimpleITK and medpy are required")
class Hd95SpacingTests(unittest.TestCase):
    def test_array_axis_spacing_is_reversed(self):
        img = sitk.Image(4, 5, 6, sitk.sitkUInt8)
        img.SetSpacing(SPACING_XYZ)
        self.assertEqual(array_axis_spacing(img), (10.0, 1.5, 1.5))

    def _hd95_lv(self, pred, gt):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write(pred, tmp / "pred.nii.gz")
            _write(gt, tmp / "gt.nii.gz")
            metrics = compute_metrics_for_volume(
                tmp / "pred.nii.gz", tmp / "gt.nii.gz", VENTRICULAR_LABELS
            )
        return metrics["hd95_lv"]

    def test_missed_slice_costs_the_slice_gap(self):
        gt = np.zeros((6, 20, 20), dtype=np.uint8)
        gt[1:5, 5:15, 5:15] = 3
        pred = gt.copy()
        pred[4] = 0  # prediction misses the last slice of the cavity
        self.assertAlmostEqual(self._hd95_lv(pred, gt), 10.0, places=5)

    def test_in_plane_shift_costs_in_plane_spacing(self):
        gt = np.zeros((6, 20, 20), dtype=np.uint8)
        gt[1:5, 5:15, 5:15] = 3
        pred = np.zeros_like(gt)
        pred[1:5, 5:15, 6:16] = 3  # one voxel shift along x
        self.assertAlmostEqual(self._hd95_lv(pred, gt), 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
