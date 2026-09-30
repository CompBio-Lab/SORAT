#!/usr/bin/env python3
"""Tests for segmentation discovery used by POSTPROCESS_ONLY / FEATURES_ONLY."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from discover_postprocess_inputs import collect_frames  # noqa: E402


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class CollectFramesTests(unittest.TestCase):
    def test_only_segmentation_masks_are_collected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seg_dir = root / "nnformer" / "segmentations"
            _touch(seg_dir / "071_ED_nnformer__fold0.nii.gz")
            _touch(seg_dir / "071_ES_nnformer__fold0.nii.gz")
            _touch(
                seg_dir / "071_nnformer__fold0_manifest.json",
                json.dumps({"frames": [{"tag": "ED", "idx": 3}, {"tag": "ES", "idx": 11}]}),
            )
            # Bulky trees that must be ignored even when names match the pattern.
            _touch(root / "postprocess" / "nnformer__fold0" / "segmentations" / "071_ED_nnformer__fold0.nii.gz")
            _touch(root / "nnformer" / "preprocessed" / "071_ED_0000.nii.gz")
            _touch(root / "previews" / "nnformer__fold0" / "071_ED_nnformer__fold0.nii.gz")
            _touch(root / "features" / "raw" / "071_ED_nnformer__fold0.nii.gz")
            _touch(root / "cinema" / "segmentations" / "071_ED_cinema__acdc_ensemble_pp.nii.gz")

            rows = collect_frames(root, {"all"})

        found = {(r["patient_id"], r["frame_tag"], r["model"], r["frame_idx"]) for r in rows}
        self.assertEqual(found, {("071", "ED", "nnformer__fold0", 3), ("071", "ES", "nnformer__fold0", 11)})
        self.assertTrue(all("/nnformer/segmentations/" in r["seg"] for r in rows))

    def test_architecture_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "nnformer" / "segmentations" / "001_ED_nnformer__fold0.nii.gz")
            _touch(root / "vsa3l" / "segmentations" / "001_ED_vsa3l__model.nii.gz")

            rows = collect_frames(root, {"vsa3l"})

        self.assertEqual([r["model"] for r in rows], ["vsa3l__model"])

    def test_unselected_architecture_folders_are_not_walked(self):
        from discover_postprocess_inputs import iter_segmentation_files

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "nnformer" / "segmentations" / "001_ED_nnformer__fold0.nii.gz")
            _touch(root / "cinema" / "segmentations" / "001_ED_cinema__acdc_ensemble.nii.gz")

            selected = [p.name for p in iter_segmentation_files(root, {"nnformer"})]
            everything = [p.name for p in iter_segmentation_files(root, {"all"})]

        self.assertEqual(selected, ["001_ED_nnformer__fold0.nii.gz"])
        self.assertEqual(len(everything), 2)


if __name__ == "__main__":
    unittest.main()
