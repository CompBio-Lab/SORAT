#!/usr/bin/env python3
"""Synthetic-phantom tests for in-plane myocardial wall thickness."""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK  # noqa: F401
    import scipy  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - runtime container supplies these
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None:
    from extract_features import (  # noqa: E402
        LABEL_LV,
        LABEL_MYO,
        LABEL_RV,
        compute_wall_thickness,
    )

# SimpleITK spacing order is (x, y, z); arrays are [z, y, x].
SPACING_XYZ = (1.25, 1.25, 10.0)
SHAPE_ZYX = (6, 96, 96)
TOLERANCE_MM = 1.25  # one in-plane pixel


def _radius_grid(shape_yx, spacing_yx, center_yx):
    yy, xx = np.indices(shape_yx, dtype=float)
    return np.hypot((yy - center_yx[0]) * spacing_yx[0], (xx - center_yx[1]) * spacing_yx[1])


def _annulus_slice(inner_mm, outer_mm, center_yx=(48, 48)):
    """One short-axis slice: LV disc inside a MYO ring, background elsewhere."""
    radius = _radius_grid(SHAPE_ZYX[1:], (SPACING_XYZ[1], SPACING_XYZ[0]), center_yx)
    slice_ = np.zeros(SHAPE_ZYX[1:], dtype=np.uint8)
    slice_[radius <= outer_mm] = LABEL_MYO
    slice_[radius <= inner_mm] = LABEL_LV
    return slice_


def _stack(slices):
    mask = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    for z, slice_ in slices.items():
        mask[z] = slice_
    return mask


@unittest.skipIf(np is None, "numpy, scipy, and SimpleITK are required for this test")
class WallThicknessTests(unittest.TestCase):
    def test_concentric_annulus_matches_ring_width(self):
        mask = _stack({z: _annulus_slice(20.0, 28.0) for z in range(1, 5)})

        result = compute_wall_thickness(mask, SPACING_XYZ)

        self.assertAlmostEqual(result["wall_thickness_mean_mm"], 8.0, delta=TOLERANCE_MM)
        self.assertAlmostEqual(result["wall_thickness_max_mm"], 8.0, delta=2 * TOLERANCE_MM)
        self.assertLessEqual(result["wall_thickness_p95_mm"], result["wall_thickness_max_mm"])

    def test_septum_bordering_rv_is_not_inflated(self):
        plain = _stack({z: _annulus_slice(20.0, 28.0) for z in range(1, 5)})
        with_rv = plain.copy()
        # RV crescent directly against the septal (left) side of the ring.
        radius_rv = _radius_grid(SHAPE_ZYX[1:], (SPACING_XYZ[1], SPACING_XYZ[0]), (48, 10))
        for z in range(1, 5):
            rv_region = (radius_rv <= 30.0) & (with_rv[z] == 0)
            with_rv[z][rv_region] = LABEL_RV

        plain_result = compute_wall_thickness(plain, SPACING_XYZ)
        rv_result = compute_wall_thickness(with_rv, SPACING_XYZ)

        self.assertTrue(np.any(with_rv == LABEL_RV))
        self.assertAlmostEqual(
            rv_result["wall_thickness_max_mm"], plain_result["wall_thickness_max_mm"], delta=TOLERANCE_MM
        )
        self.assertAlmostEqual(
            rv_result["wall_thickness_mean_mm"], plain_result["wall_thickness_mean_mm"], delta=TOLERANCE_MM
        )

    def test_max_comes_from_thickest_slice(self):
        slices = {z: _annulus_slice(20.0, 28.0) for z in (1, 2, 4)}
        slices[3] = _annulus_slice(15.0, 31.0)  # 16 mm wall in one slice
        mask = _stack(slices)

        result = compute_wall_thickness(mask, SPACING_XYZ)

        self.assertAlmostEqual(result["wall_thickness_max_mm"], 16.0, delta=2 * TOLERANCE_MM)
        self.assertLess(result["wall_thickness_mean_mm"], 12.0)

    def test_no_through_plane_leakage_between_slices(self):
        # A thick basal slice above a slice whose MYO has no LV neighbour must not
        # contribute cross-slice distances.
        slices = {1: _annulus_slice(20.0, 28.0), 2: _annulus_slice(20.0, 28.0)}
        cap = np.zeros(SHAPE_ZYX[1:], dtype=np.uint8)
        cap[_radius_grid(SHAPE_ZYX[1:], (1.25, 1.25), (48, 48)) <= 28.0] = LABEL_MYO
        slices[3] = cap  # apical cap: MYO only
        mask = _stack(slices)

        result = compute_wall_thickness(mask, SPACING_XYZ)

        self.assertAlmostEqual(result["wall_thickness_max_mm"], 8.0, delta=2 * TOLERANCE_MM)

    def test_missing_lv_returns_nan(self):
        mask = np.zeros(SHAPE_ZYX, dtype=np.uint8)
        mask[2, 40:50, 40:50] = LABEL_MYO

        result = compute_wall_thickness(mask, SPACING_XYZ)

        for key in ("wall_thickness_mean_mm", "wall_thickness_max_mm", "wall_thickness_p95_mm"):
            self.assertTrue(np.isnan(result[key]), key)

    def test_slice_axis_follows_largest_spacing(self):
        # Same phantom, stored with slices along x instead of z.
        mask_z = _stack({z: _annulus_slice(20.0, 28.0) for z in range(1, 5)})
        mask_x = np.transpose(mask_z, (1, 2, 0))  # [y, x, z] -> slice axis is last
        spacing_x_is_slice = (10.0, 1.25, 1.25)  # (x, y, z) spacing for array [z, y, x]

        result_z = compute_wall_thickness(mask_z, SPACING_XYZ)
        result_x = compute_wall_thickness(mask_x, spacing_x_is_slice)

        self.assertAlmostEqual(
            result_x["wall_thickness_mean_mm"], result_z["wall_thickness_mean_mm"], delta=1e-6
        )


@unittest.skipIf(np is None, "numpy, scipy, and SimpleITK are required for this test")
class LoadMaskGeometryTests(unittest.TestCase):
    def test_load_mask_accepts_non_orthonormal_sform(self):
        try:
            import nibabel as nib
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("nibabel is required")
        import tempfile

        from extract_features import load_mask

        mask = np.zeros((6, 7, 5), dtype=np.uint8)
        mask[2:4, 2:5, 1:4] = LABEL_MYO
        affine = np.array(
            [[1.0, 0.25, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sheared_mask.nii.gz"
            nib.save(nib.Nifti1Image(mask, affine), path)

            _, loaded = load_mask(path)

        self.assertEqual(int((loaded == LABEL_MYO).sum()), int((mask == LABEL_MYO).sum()))


if __name__ == "__main__":
    unittest.main()
