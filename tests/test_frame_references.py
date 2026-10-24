#!/usr/bin/env python3
"""Selected 4-D frames must equal the per-frame images a dataset provides."""

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover - local host may lack runtime deps
    np = None
    sitk = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None and sitk is not None:
    from frame_manifest import verify_frame_references  # noqa: E402


def _write(arr, path):
    sitk.WriteImage(sitk.GetImageFromArray(arr.astype(np.float32)), str(path))


@unittest.skipIf(np is None or sitk is None, "numpy and SimpleITK are required")
class VerifyFrameReferencesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.cine = rng.random((5, 3, 8, 8)).astype(np.float32)  # (t, z, y, x)
        self.cine_path = self.dir / "patient001_4d.nii.gz"
        img = sitk.JoinSeries([sitk.GetImageFromArray(f) for f in self.cine])
        sitk.WriteImage(img, str(self.cine_path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_acdc_frame_files_are_one_based(self):
        _write(self.cine[0], self.dir / "patient001_frame01.nii.gz")
        _write(self.cine[3], self.dir / "patient001_frame04.nii.gz")
        manifest = {"frames": [{"tag": "ED", "idx": 0}, {"tag": "ES", "idx": 3}]}
        self.assertEqual(verify_frame_references(self.cine_path, manifest, "patient001"), 2)

    def test_off_by_one_index_is_rejected(self):
        # Info.cfg "ED: 1" read as a 0-based index selects 4-D frame 1, but the
        # dataset's phase image is frame01 = 4-D frame 0.
        _write(self.cine[0], self.dir / "patient001_frame01.nii.gz")
        _write(self.cine[3], self.dir / "patient001_frame04.nii.gz")
        manifest = {"frames": [{"tag": "ED", "idx": 1}, {"tag": "ES", "idx": 4}]}
        with self.assertRaisesRegex(ValueError, "not one of the provided phase images"):
            verify_frame_references(self.cine_path, manifest, "patient001")

    def test_misnumbered_phase_image_is_rejected(self):
        _write(self.cine[0], self.dir / "patient001_frame02.nii.gz")
        manifest = {"frames": [{"tag": "ED", "idx": 1}]}
        with self.assertRaisesRegex(ValueError, "should equal 4-D frame 1"):
            verify_frame_references(self.cine_path, manifest, "patient001")

    def test_mms2_tagged_reference(self):
        _write(self.cine[4], self.dir / "patient001_SA_ES.nii.gz")
        ok = {"frames": [{"tag": "ES", "idx": 4}]}
        self.assertEqual(verify_frame_references(self.cine_path, ok, "patient001"), 1)
        with self.assertRaises(ValueError):
            verify_frame_references(self.cine_path, {"frames": [{"tag": "ES", "idx": 2}]}, "patient001")

    def test_symlinked_input_finds_references_next_to_the_real_file(self):
        _write(self.cine[0], self.dir / "patient001_frame01.nii.gz")
        with tempfile.TemporaryDirectory() as work:
            link = Path(work) / "patient001_4d.nii.gz"
            link.symlink_to(self.cine_path)
            manifest = {"frames": [{"tag": "ED", "idx": 0}]}
            self.assertEqual(verify_frame_references(link, manifest, "patient001"), 1)

    def test_no_references_verifies_nothing(self):
        manifest = {"frames": [{"tag": "ED", "idx": 0}]}
        self.assertEqual(verify_frame_references(self.cine_path, manifest, "patient001"), 0)


if __name__ == "__main__":
    unittest.main()
