#!/usr/bin/env python3
"""prepare_input.py converts NRRD and DICOM inputs to NIfTI with provenance."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover
    np = None

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))

if np is not None:
    import prepare_input as P  # noqa: E402


def write_dicom_series(directory: Path, volumes, series_uid: str, spacing=(0.8, 0.8, 8.0), trigger_step=40.0):
    """Write one series: volumes is a list of (z, y, x) arrays, one per cine frame."""
    directory.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    study_uid = "1.2.826.0.1.3680043.2.1125.1"
    n = 0
    for t, vol in enumerate(volumes):
        for z in range(vol.shape[0]):
            sl = sitk.GetImageFromArray(vol[z].astype(np.int16))
            sl.SetSpacing(spacing[:2])
            tags = {
                "0008|0060": "MR", "0020|000d": study_uid, "0020|000e": series_uid,
                "0008|0018": f"{series_uid}.{t}.{z}", "0020|0013": str(n + 1),
                "0020|0032": f"0\\0\\{z * spacing[2]}", "0020|0037": "1\\0\\0\\0\\1\\0",
                "0028|0030": f"{spacing[1]}\\{spacing[0]}", "0018|0050": str(spacing[2]),
                "0018|1060": str(t * trigger_step),
            }
            for k, v in tags.items():
                sl.SetMetaData(k, v)
            writer.SetFileName(str(directory / f"{series_uid.split('.')[-1]}_{n:04d}.dcm"))
            writer.Execute(sl)
            n += 1


@unittest.skipIf(np is None, "numpy and SimpleITK are required")
class PrepareInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        out = self.dir / "out"
        subprocess.run([sys.executable, str(BIN / "prepare_input.py"), "--out_dir", str(out), *args],
                       check=True, capture_output=True, text=True)
        return out

    def test_nifti_passes_through(self):
        img = self.dir / "a.nii.gz"
        sitk.WriteImage(sitk.Image(4, 4, 4, sitk.sitkUInt8), str(img))
        rec = P.prepare(img, self.dir / "x.nii.gz", None, None, "image")
        self.assertFalse(rec["converted"])
        self.assertFalse((self.dir / "x.nii.gz").exists())

    def test_nrrd_with_spacing_override(self):
        arr = self.rng.integers(0, 255, (6, 8, 8)).astype(np.uint8)
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, str(self.dir / "lge.nrrd"))
        sitk.WriteImage(sitk.GetImageFromArray((arr > 128).astype(np.uint8) * 255), str(self.dir / "la.nrrd"))
        out = self.run_cli("--patient_id", "p1", "--image", str(self.dir / "lge.nrrd"),
                           "--ground_truth", str(self.dir / "la.nrrd"), "--image_spacing", "0.625,0.625,1.25")
        conv = sitk.ReadImage(str(out / "p1_image.nii.gz"))
        self.assertEqual(conv.GetSpacing(), (0.625, 0.625, 1.25))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(conv), arr))
        self.assertEqual(sitk.ReadImage(str(out / "p1_gt.nii.gz")).GetSpacing(), (0.625, 0.625, 1.25))
        prov = json.loads((out / "p1_input_provenance.json").read_text())
        self.assertEqual(prov["files"][0]["spacing_override_mm"], [0.625, 0.625, 1.25])
        self.assertEqual(len(prov["files"][0]["source_sha256"]), 64)

    def test_unit_spacing_without_override_warns(self):
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((3, 4, 4), np.uint8)), str(self.dir / "u.nrrd"))
        res = subprocess.run([sys.executable, str(BIN / "prepare_input.py"), "--patient_id", "u", "--image",
                              str(self.dir / "u.nrrd"), "--out_dir", str(self.dir / "o")],
                             capture_output=True, text=True, check=True)
        self.assertIn("spacing 1 x 1 x 1", res.stderr)

    def test_bad_spacing_is_rejected(self):
        for text in ("1,2", "1,0,1", "a,b,c"):
            with self.assertRaises(ValueError):
                P.parse_spacing(text)

    def test_dicom_3d_series(self):
        vol = self.rng.integers(0, 1000, (5, 6, 7))
        write_dicom_series(self.dir / "dcm", [vol], "1.2.3.4.100")
        image, info = P.read_dicom_series(self.dir / "dcm")
        self.assertEqual(info["n_frames"], 1)
        self.assertEqual(image.GetSize(), (7, 6, 5))
        self.assertTrue(np.array_equal(sitk.GetArrayFromImage(image), vol))
        np.testing.assert_allclose(image.GetSpacing(), (0.8, 0.8, 8.0), atol=1e-6)

    def test_dicom_cine_becomes_4d_in_trigger_order(self):
        frames = [self.rng.integers(0, 1000, (4, 6, 6)) for _ in range(3)]
        # Trigger times run opposite to file/instance order, so only trigger-time
        # sorting gives the expected frame order.
        write_dicom_series(self.dir / "cine", frames, "1.2.3.4.200", trigger_step=-40.0)
        image, info = P.read_dicom_series(self.dir / "cine")
        self.assertEqual((info["n_slices"], info["n_frames"]), (4, 3))
        arr = sitk.GetArrayFromImage(image)  # (t, z, y, x)
        self.assertEqual(arr.shape, (3, 4, 6, 6))
        for t in range(3):
            self.assertTrue(np.array_equal(arr[t], frames[2 - t]))

    def test_multiple_series_need_a_choice(self):
        write_dicom_series(self.dir / "two", [self.rng.integers(0, 9, (3, 4, 4))], "1.2.3.4.301")
        write_dicom_series(self.dir / "two", [self.rng.integers(0, 9, (3, 4, 4))], "1.2.3.4.302")
        with self.assertRaisesRegex(ValueError, "2 DICOM series"):
            P.read_dicom_series(self.dir / "two")
        image, info = P.read_dicom_series(self.dir / "two", "1.2.3.4.302")
        self.assertEqual(info["series_uid"], "1.2.3.4.302")


if __name__ == "__main__":
    unittest.main()
