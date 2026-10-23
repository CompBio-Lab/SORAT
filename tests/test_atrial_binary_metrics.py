#!/usr/bin/env python3
"""Regression tests for binary AIM-ASD-style atrial references."""

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
    from medpy.metric import binary  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - runtime container supplies these
    np = None
    sitk = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None and sitk is not None:
    from compute_metrics import (  # noqa: E402
        compute_metrics_for_volume,
        get_label_spec,
    )


@unittest.skipIf(
    np is None or sitk is None,
    "numpy, SimpleITK, and medpy are required for this regression test",
)
class AtrialBinaryMetricTests(unittest.TestCase):
    def test_union_schema_resamples_and_collapses_multiclass_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred = np.zeros((4, 5, 6), dtype=np.uint8)
            pred[1:3, 1:4, 1:5] = 2
            pred[0:2, 3:5, 0:2] = 3
            gt = np.zeros((5, 6, 7), dtype=np.uint8)
            gt[1:3, 1:4, 1:5] = 1
            gt[0:2, 3:5, 0:2] = 1

            pred_path = tmp / "prediction.nii.gz"
            gt_path = tmp / "binary_biatrial.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(pred), str(pred_path))
            sitk.WriteImage(sitk.GetImageFromArray(gt), str(gt_path))

            metrics = compute_metrics_for_volume(
                pred_path,
                gt_path,
                get_label_spec("atrial_nnunet", "atrial_binary_union"),
                label_schema="atrial_binary_union",
            )

        self.assertEqual(metrics["dice_biatrial"], 1.0)
        self.assertEqual(metrics["hd95_biatrial"], 0.0)
        self.assertEqual(metrics["dice_mean"], 1.0)

    def test_binary_schema_is_restricted_to_atrial_architecture(self):
        with self.assertRaises(ValueError):
            get_label_spec("vsa3l", "atrial_binary_union")


if __name__ == "__main__":
    unittest.main()
