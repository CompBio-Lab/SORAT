#!/usr/bin/env python3
"""Regression tests for metric computation on non-orthonormal NIfTI geometry."""

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover - local host may lack runtime deps
    nib = None
    np = None
    sitk = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None and sitk is not None:
    from compute_metrics import VENTRICULAR_LABELS, compute_metrics_for_volume  # noqa: E402


@unittest.skipIf(
    nib is None or np is None or sitk is None,
    "nibabel, numpy, and SimpleITK are required for this regression test",
)
class ComputeMetricsGeometryTests(unittest.TestCase):
    def test_nonorthonormal_ground_truth_uses_nibabel_fallback(self):
        pred = np.zeros((6, 7, 5), dtype=np.uint8)
        gt = np.zeros_like(pred)
        pred[2:4, 2:5, 1:4] = 3
        gt[2:4, 2:5, 1:4] = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_path = tmp / "pred.nii.gz"
            gt_path = tmp / "gt_sheared.nii.gz"

            nib.save(nib.Nifti1Image(pred, np.eye(4)), pred_path)

            affine = np.array(
                [
                    [1.0, 0.25, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            gt_img = nib.Nifti1Image(gt, affine)
            gt_img.set_qform(np.eye(4), code=0)
            gt_img.set_sform(affine, code=1)
            nib.save(gt_img, gt_path)

            with self.assertRaises(RuntimeError):
                sitk.ReadImage(str(gt_path))

            metrics = compute_metrics_for_volume(pred_path, gt_path, VENTRICULAR_LABELS)

        self.assertEqual(metrics["dice_lv"], 1.0)
        self.assertEqual(metrics["hd95_lv"], 0.0)


if __name__ == "__main__":
    unittest.main()
