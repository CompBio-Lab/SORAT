#!/usr/bin/env python3
"""pick_ed_es.py takes ED/ES from all-frame LV volumes and writes a 1-based Info.cfg."""

import csv
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
    import frame_manifest  # noqa: E402
    import pick_ed_es as P  # noqa: E402


def write_frames(folder: Path, pid: str, tag: str, lv_radii, spacing=(1.5, 1.5, 8.0)):
    """One mask per frame: an LV disc of the given radius (negative: no LV) on 4 slices, with a fixed RV block."""
    frames = []
    for idx, r in enumerate(lv_radii):
        arr = np.zeros((4, 32, 32), np.uint8)
        yy, xx = np.mgrid[:32, :32]
        if r >= 0:
            arr[:, (yy - 16) ** 2 + (xx - 16) ** 2 <= r * r] = 3
        arr[:, 2:6, 2:6] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        sitk.WriteImage(img, str(folder / f"{pid}_frame{idx:02d}_{tag}.nii.gz"))
        frames.append({"tag": f"frame{idx:02d}", "idx": idx})
    (folder / f"{pid}_{tag}_manifest.json").write_text(
        json.dumps({"patient_id": pid, "has_info_cfg": False, "num_frames": len(lv_radii), "frames": frames}))


@unittest.skipIf(np is None, "numpy and SimpleITK are required")
class PickEdEsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.seg = self.dir / "seg"
        self.seg.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self):
        out = self.dir / "info"
        subprocess.run([sys.executable, str(BIN / "pick_ed_es.py"), "--segmentations", str(self.seg),
                        "--model_tag", "nnformer__fold0", "--out_dir", str(out)], check=True, capture_output=True)
        with open(out / "ed_es.csv") as h:
            return out, {r["patient_id"]: r for r in csv.DictReader(h)}

    def test_largest_and_smallest_lv_become_ed_and_es(self):
        write_frames(self.seg, "p1", "nnformer__fold0", [10, 9, 7, 5, 6, 8, 9, 10])
        out, rows = self.run_cli()
        # Largest radius first appears at frame 0; smallest at frame 3.
        self.assertEqual((rows["p1"]["ed_1based"], rows["p1"]["es_1based"]), ("1", "4"))
        info = frame_manifest.parse_info_cfg(out / "p1" / "Info.cfg")
        self.assertEqual((info["ed_frame"], info["es_frame"]), (0, 3))
        self.assertEqual(rows["p1"]["ed_far_from_start"], "False")
        with open(out / "volume_curves.csv") as h:
            curve = list(csv.DictReader(h))
        self.assertEqual(len(curve), 8)
        self.assertGreater(float(curve[0]["lv_ml"]), float(curve[3]["lv_ml"]))

    def test_empty_frames_are_left_out_and_flagged(self):
        write_frames(self.seg, "p2", "nnformer__fold0", [10, 8, -1, 6, 9, 10, 10, 10])
        _, rows = self.run_cli()
        self.assertEqual(rows["p2"]["es_1based"], "4")
        self.assertEqual(rows["p2"]["empty_lv_frames"], "1")

    def test_late_ed_is_flagged(self):
        write_frames(self.seg, "p3", "nnformer__fold0", [7, 6, 5, 6, 7, 8, 9, 10, 9, 8])
        _, rows = self.run_cli()
        self.assertEqual(rows["p3"]["ed_far_from_start"], "True")

    def test_partial_frames_are_an_error(self):
        write_frames(self.seg, "p4", "nnformer__fold0", [10, 8, 6])
        manifest = self.seg / "p4_nnformer__fold0_manifest.json"
        data = json.loads(manifest.read_text())
        data["num_frames"] = 20
        manifest.write_text(json.dumps(data))
        res = subprocess.run([sys.executable, str(BIN / "pick_ed_es.py"), "--segmentations", str(self.seg),
                              "--model_tag", "nnformer__fold0", "--out_dir", str(self.dir / "o")],
                             capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("frames_mode all", res.stderr)

    def test_local_minima_are_cyclic(self):
        self.assertEqual(P.local_minima(np.array([1.0, 3.0, 2.0, 4.0])), [0, 2])


if __name__ == "__main__":
    unittest.main()
