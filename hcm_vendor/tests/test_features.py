import numpy as np
import pandas as pd
import pytest

from hcmv.features import (
    add_derived,
    build_feature_table,
    feature_columns,
    feature_family,
    parse_feature_filename,
    prune_columns,
    read_phase_csvs,
    to_wide,
)

SOURCE = "nnformer__fold0"


def _phase_row(pid, phase, lv, rv, myo, texture, extra=None):
    p = phase.lower()
    row = {
        "patient_id": f"{pid}_{SOURCE}_{phase}",
        f"{p}_mask_file": f"/masks/{pid}_{phase}.nii.gz",
        f"{p}_mask_source": "auto",
        f"{p}_voxel_volume_ml": 0.0139,
        f"{p}_radiomics_settings": "{}",
        f"{p}_lv_volume_ml": lv,
        f"{p}_rv_volume_ml": rv,
        f"{p}_myo_volume_ml": myo,
        f"{p}_myocardial_mass_g": myo * 1.05,
        f"{p}_wall_thickness_mean_mm": 6.0,
        f"{p}_wall_thickness_max_mm": 11.0,
        f"{p}_wall_thickness_p95_mm": 9.0,
        f"{p}_radiomics_original_shape_Sphericity": 0.4 + lv / 1000,
        f"{p}_radiomics_original_shape_MeshVolume": myo * 1000,
        f"{p}_radiomics_original_firstorder_Mean": texture,
        f"{p}_radiomics_original_glcm_Contrast": texture / 10,
        "myocardial_mass_g": myo * 1.05,
    }
    row.update(extra or {})
    return row


@pytest.fixture
def feature_dir(tmp_path):
    values = {
        "001": (150, 160, 100, 40.0, 0.4),
        "071": (120, 130, 140, 55.0, 0.3),
        "230": (140, 150, 90, 47.0, 0.45),
    }
    for pid, (lv, rv, myo, tex, es_fraction) in values.items():
        pd.DataFrame([_phase_row(pid, "ED", lv, rv, myo, tex)]).to_csv(
            tmp_path / f"{pid}_{SOURCE}_ED_features.csv", index=False)
        pd.DataFrame([_phase_row(pid, "ES", lv * es_fraction, rv * 0.5, myo, tex + 3)]).to_csv(
            tmp_path / f"{pid}_{SOURCE}_ES_features.csv", index=False)
    return tmp_path


@pytest.fixture
def cohort():
    return pd.DataFrame({
        "subject_id": ["mms2_001", "mms2_071", "mms2_230", "acdc_patient101"],
        "dataset": ["mms2", "mms2", "mms2", "acdc"],
        "source_id": ["001", "071", "230", "patient101"],
        "disease": ["NOR", "HCM", "NOR", "HCM"],
        "y": [0, 1, 0, 1],
        "vendor": ["GE", "Siemens", "Philips", "Siemens"],
    })


def test_parse_feature_filename():
    assert parse_feature_filename("071_nnformer__fold0_ED_features.csv") == ("071", SOURCE, "ED")
    assert parse_feature_filename("patient101_gt_ES_features.csv") == ("patient101", "gt", "ES")
    with pytest.raises(ValueError):
        parse_feature_filename("071_nnformer__fold0_FRAME_features.csv")


def test_read_phase_csvs_uses_filename_ids(feature_dir):
    long = read_phase_csvs(feature_dir, SOURCE)
    assert sorted(long["source_id"].unique()) == ["001", "071", "230"]
    assert set(long["phase"]) == {"ED", "ES"}
    assert "lv_volume_ml" in long.columns and "ed_lv_volume_ml" not in long.columns


def test_empty_feature_file_is_rejected(feature_dir):
    (feature_dir / f"282_{SOURCE}_ED_features.csv").write_text("")
    with pytest.raises(ValueError, match="Empty"):
        read_phase_csvs(feature_dir, SOURCE)


def test_missing_phase_is_rejected(feature_dir):
    (feature_dir / f"071_{SOURCE}_ES_features.csv").unlink()
    with pytest.raises(ValueError, match="missing a phase"):
        to_wide(read_phase_csvs(feature_dir, SOURCE))


def test_derived_features():
    wide = pd.DataFrame({
        "ed_lv_volume_ml": [150.0, 0.0], "es_lv_volume_ml": [60.0, 0.0],
        "ed_rv_volume_ml": [160.0, 100.0], "es_rv_volume_ml": [80.0, 50.0],
        "ed_myocardial_mass_g": [105.0, 90.0],
    })
    out = add_derived(wide)
    assert out.loc[0, "lv_sv_ml"] == 90.0
    assert out.loc[0, "lvef_pct"] == pytest.approx(60.0)
    assert out.loc[0, "rvef_pct"] == pytest.approx(50.0)
    assert out.loc[0, "mass_to_volume_g_per_ml"] == pytest.approx(0.7)
    assert np.isnan(out.loc[1, "lvef_pct"]) and np.isnan(out.loc[1, "mass_to_volume_g_per_ml"])


def test_feature_families():
    assert feature_family("ed_wall_thickness_max_mm") == "clinical"
    assert feature_family("lvef_pct") == "clinical"
    assert feature_family("ed_wall_thickness_p95_mm") == "sensitivity"
    assert feature_family("es_radiomics_original_shape_Sphericity") == "shape"
    assert feature_family("ed_radiomics_original_firstorder_Mean") == "texture"
    assert feature_family("ed_radiomics_original_glcm_Contrast") == "texture"
    assert feature_family("vendor") is None


def test_prune_drops_meta_constant_and_duplicates():
    wide = pd.DataFrame({
        "ed_mask_file": ["a", "b", "c"],
        "ed_wall_thickness_mean_mm": [6.0, 6.0, 6.0],        # constant
        "ed_myo_volume_ml": [100.0, 140.0, 90.0],
        "es_myo_volume_ml": [100.0, 140.0, 90.0],            # duplicate of ED
        "lvef_pct": [60.0, 55.0, 58.0],
    })
    table, log = prune_columns(wide)
    assert list(table.columns) == ["ed_myo_volume_ml", "lvef_pct"]
    reasons = dict(zip(log["column"], log["reason"]))
    assert reasons["ed_mask_file"] == "meta"
    assert reasons["ed_wall_thickness_mean_mm"] == "constant"
    assert reasons["es_myo_volume_ml"] == "duplicate_of:ed_myo_volume_ml"


def test_build_feature_table_joins_cohort(feature_dir, cohort):
    table, log = build_feature_table(feature_dir, SOURCE, cohort, "mms2")
    assert list(table.index) == ["mms2_001", "mms2_071", "mms2_230"]
    assert table.loc["mms2_071", "y"] == 1 and table.loc["mms2_071", "vendor"] == "Siemens"
    assert "myocardial_mass_g" not in table.columns
    assert not any(c.endswith("radiomics_settings") for c in table.columns)
    assert table.loc["mms2_001", "lvef_pct"] == pytest.approx(60.0)
    assert set(feature_columns(table, ["texture"])) == {
        "ed_radiomics_original_firstorder_Mean", "es_radiomics_original_firstorder_Mean",
        "ed_radiomics_original_glcm_Contrast", "es_radiomics_original_glcm_Contrast",
    }


def test_cohort_subject_without_features_raises(feature_dir, cohort):
    (feature_dir / f"230_{SOURCE}_ED_features.csv").unlink()
    (feature_dir / f"230_{SOURCE}_ES_features.csv").unlink()
    with pytest.raises(ValueError, match="lack features"):
        build_feature_table(feature_dir, SOURCE, cohort, "mms2")


def test_radiomics_error_raises(feature_dir, cohort):
    row = _phase_row("071", "ED", 120, 130, 140, 55.0, {"ed_radiomics_error": "boom"})
    pd.DataFrame([row]).to_csv(feature_dir / f"071_{SOURCE}_ED_features.csv", index=False)
    with pytest.raises(ValueError, match="radiomics_error"):
        build_feature_table(feature_dir, SOURCE, cohort, "mms2")
