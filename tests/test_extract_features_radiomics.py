#!/usr/bin/env python3
"""Tests for configurable PyRadiomics settings in extract_features.py."""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover - runtime container supplies these
    np = None
    sitk = None

try:
    import radiomics  # noqa: F401

    HAVE_RADIOMICS = True
except ModuleNotFoundError:  # pragma: no cover - PyRadiomics comes from the feature venv
    HAVE_RADIOMICS = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

if np is not None and sitk is not None:
    from extract_features import (  # noqa: E402
        LABEL_LV,
        LABEL_MYO,
        build_radiomics_settings,
        extract_radiomics_features,
        parse_spacing,
    )

NORM_SETTINGS = {
    "normalize": True,
    "normalize_scale": 100,
    "bin_count": 32,
    "resample_spacing": [1.25, 1.25, 0],
    "force2d": True,
    "force2d_dimension": 0,
}


def _phantom(scale=1.0, offset=0.0, seed=0):
    """Short-axis-like image (x,y,z spacing 1.25,1.25,10) with a textured MYO ring."""
    rng = np.random.default_rng(seed)
    shape_zyx = (6, 64, 64)
    yy, xx = np.indices(shape_zyx[1:], dtype=float)
    radius = np.hypot((yy - 32) * 1.25, (xx - 32) * 1.25)

    mask = np.zeros(shape_zyx, dtype=np.uint8)
    image = rng.normal(40.0, 5.0, size=shape_zyx)
    for z in range(1, 5):
        mask[z][radius <= 26.0] = LABEL_MYO
        mask[z][radius <= 18.0] = LABEL_LV
        image[z][radius <= 18.0] += 150.0  # bright blood pool
        ring = (radius > 18.0) & (radius <= 26.0)
        image[z][ring] += 60.0 + 25.0 * np.sin(xx[ring] / 3.0) + rng.normal(0, 8.0, ring.sum())
    image = image * scale + offset

    def to_sitk(array, pixel_type):
        img = sitk.GetImageFromArray(array.astype(pixel_type))
        img.SetSpacing((1.25, 1.25, 10.0))
        return img

    return to_sitk(image, np.float32), to_sitk(mask, np.uint8)


def _texture(features):
    return {
        k: v for k, v in features.items()
        if "_firstorder_" in k or "_glcm_" in k
    }


@unittest.skipIf(np is None or sitk is None, "numpy and SimpleITK are required")
class RadiomicsSettingsTests(unittest.TestCase):
    def test_defaults_produce_empty_settings(self):
        self.assertEqual(build_radiomics_settings(), {})

    def test_norm_config_maps_to_pyradiomics_keys(self):
        settings = build_radiomics_settings(**NORM_SETTINGS)
        self.assertEqual(
            settings,
            {
                "normalize": True,
                "normalizeScale": 100.0,
                "binCount": 32,
                "resampledPixelSpacing": [1.25, 1.25, 0.0],
                "force2D": True,
                "force2Ddimension": 0,
            },
        )

    def test_bin_count_and_width_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            build_radiomics_settings(bin_count=32, bin_width=25)

    def test_parse_spacing(self):
        self.assertEqual(parse_spacing("1.25,1.25,0"), [1.25, 1.25, 0.0])
        self.assertIsNone(parse_spacing(None))
        self.assertIsNone(parse_spacing(""))
        with self.assertRaises(ValueError):
            parse_spacing("1.25,1.25")


@unittest.skipIf(np is None or sitk is None or not HAVE_RADIOMICS, "PyRadiomics is required")
class RadiomicsExtractionTests(unittest.TestCase):
    def test_norm_texture_is_invariant_to_linear_intensity_rescaling(self):
        settings = build_radiomics_settings(**NORM_SETTINGS)
        base = _texture(extract_radiomics_features(*_phantom(), settings))
        for scale, offset in ((0.2, 0.0), (5.0, 30.0)):
            rescaled = _texture(extract_radiomics_features(*_phantom(scale, offset), settings))
            self.assertEqual(base.keys(), rescaled.keys())
            for key, value in base.items():
                self.assertTrue(
                    np.isclose(value, rescaled[key], rtol=1e-3, atol=1e-6),
                    f"{key} changed under scale={scale} offset={offset}: {value} vs {rescaled[key]}",
                )

    def test_default_texture_depends_on_intensity_scale(self):
        base = _texture(extract_radiomics_features(*_phantom(), {}))
        rescaled = _texture(extract_radiomics_features(*_phantom(0.2, 0.0), {}))
        key = "radiomics_original_glcm_Contrast"
        self.assertFalse(np.isclose(base[key], rescaled[key], rtol=1e-2))

    def test_empty_settings_match_plain_extractor(self):
        from radiomics import featureextractor

        image, mask = _phantom()
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        for name in ("shape", "firstorder", "glcm"):
            extractor.enableFeatureClassByName(name)
        reference = {
            f"radiomics_{k}": float(v)
            for k, v in extractor.execute(image, mask, label=LABEL_MYO).items()
            if str(k).startswith("original_")
        }
        self.assertEqual(extract_radiomics_features(image, mask, {}), reference)


if __name__ == "__main__":
    unittest.main()
