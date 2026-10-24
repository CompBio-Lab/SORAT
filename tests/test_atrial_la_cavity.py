#!/usr/bin/env python3
"""The atrial_la_cavity schema scores predicted LA (label 3) against a binary reference."""

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
    from medpy.metric import binary  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None:
    from compute_metrics import compute_metrics_for_volume, get_label_spec  # noqa: E402


@unittest.skipIf(np is None, "numpy, SimpleITK and medpy are required")
class AtrialLaCavityTests(unittest.TestCase):
    def test_label_spec(self):
        self.assertEqual(get_label_spec("atrial_nnunet", "atrial_la_cavity"), [(3, "la")])
        with self.assertRaises(ValueError):
            get_label_spec("nnformer", "atrial_la_cavity")

    def test_only_la_label_counts(self):
        pred = np.zeros((8, 16, 16), dtype=np.uint8)
        pred[2:6, 4:10, 4:10] = 3   # LA cavity, matches the reference
        pred[2:6, 10:14, 4:10] = 2  # RA must be ignored
        gt = np.zeros_like(pred)
        gt[2:6, 4:10, 4:10] = 255   # binary reference stored as 0/255
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, arr in (("pred", pred), ("gt", gt)):
                img = sitk.GetImageFromArray(arr)
                img.SetSpacing((0.625, 0.625, 1.25))
                p = Path(tmp) / f"{name}.nii.gz"
                sitk.WriteImage(img, str(p))
                paths.append(p)
            m = compute_metrics_for_volume(paths[0], paths[1], [(3, "la")], label_schema="atrial_la_cavity")
        self.assertAlmostEqual(m["dice_la"], 1.0)


if __name__ == "__main__":
    unittest.main()
