#!/usr/bin/env python3
"""Info.cfg frame numbers count from 1 (ACDC) and map to 0-based indices."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from frame_manifest import build_frame_manifest, parse_info_cfg  # noqa: E402


class InfoCfgTests(unittest.TestCase):
    def _cfg(self, text):
        tmp = tempfile.NamedTemporaryFile("w", suffix="Info.cfg", delete=False)
        tmp.write(text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return tmp.name

    def test_acdc_values_become_zero_based(self):
        cfg = self._cfg("ED: 1\nES: 14\nGroup: NOR\nNbFrame: 30\n")
        self.assertEqual(parse_info_cfg(cfg), {"ed_frame": 0, "es_frame": 13})

    def test_manifest_uses_zero_based_indices(self):
        cfg = self._cfg("ED: 1\nES: 30\nNbFrame: 30\n")
        manifest = build_frame_manifest(info_cfg_path=cfg, num_frames=30, patient_id="p")
        self.assertEqual(manifest["frames"], [{"tag": "ED", "idx": 0}, {"tag": "ES", "idx": 29}])

    def test_zero_is_rejected(self):
        cfg = self._cfg("ED: 0\nES: 9\n")
        with self.assertRaisesRegex(ValueError, "counts frames from 1"):
            parse_info_cfg(cfg)

    def test_missing_file_gives_defaults(self):
        self.assertEqual(parse_info_cfg(None), {"ed_frame": 0, "es_frame": None})
        self.assertEqual(parse_info_cfg("/nonexistent/Info.cfg"), {"ed_frame": 0, "es_frame": None})


if __name__ == "__main__":
    unittest.main()
