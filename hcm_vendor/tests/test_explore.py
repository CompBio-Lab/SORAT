import numpy as np
import pandas as pd
import pytest

from hcmv.explore import apply_d6, auc_agreement, load_dice, univariate_auc


def _table():
    rows = []
    for i in range(6):
        rows.append({
            "subject_id": f"mms2_{i:03d}", "vendor": "Siemens" if i < 3 else "Philips", "y": i % 2,
            "ed_lv_volume_ml": 150.0, "es_lv_volume_ml": 60.0, "ed_myo_volume_ml": 100.0,
            "es_myo_volume_ml": 100.0, "lvef_pct": 60.0, "rvef_pct": 55.0,
            "ed_radiomics_original_glcm_Contrast": float(i),
        })
    return pd.DataFrame(rows).set_index("subject_id")


def test_apply_d6_passes_clean_table():
    assert apply_d6(_table()).empty


def test_apply_d6_flags_each_criterion_and_joins_reasons():
    t = _table()
    t.loc["mms2_000", "ed_radiomics_original_glcm_Contrast"] = np.nan
    t.loc["mms2_001", "es_myo_volume_ml"] = 0.0
    t.loc["mms2_002", "ed_lv_volume_ml"] = 15.0
    t.loc["mms2_003", "lvef_pct"] = 120.0
    t.loc["mms2_003", "rvef_pct"] = -5.0
    out = apply_d6(t).set_index("subject_id")["reason"]
    assert out["mms2_000"] == "missing or NaN features"
    assert out["mms2_001"] == "empty MYO at ES"
    assert out["mms2_002"] == "LV EDV < 20 ml"
    assert out["mms2_003"] == "lvef_pct outside 0-100%; rvef_pct outside 0-100%"
    assert len(out) == 4


def test_load_dice_pads_mms2_ids_and_filters_model(tmp_path):
    path = tmp_path / "aggregated_metrics.csv"
    pd.DataFrame({
        "patient_id": ["9", "9", "71", "patient101"],
        "model": ["nnformer__fold0", "vsa3l__model", "nnformer__fold0", "nnformer__fold0"],
        "frame_tag": ["ED"] * 4, "dice_lv": [0.9, 0.5, 0.8, 0.7],
    }).to_csv(path, index=False)
    cohort = pd.DataFrame({
        "subject_id": ["mms2_009", "mms2_071", "acdc_patient101"], "dataset": ["mms2", "mms2", "acdc"],
        "source_id": ["009", "071", "patient101"], "vendor": ["GE", "Siemens", "Siemens"],
        "disease": ["NOR", "HCM", "HCM"],
    })
    mms2 = load_dice(path, cohort, "mms2", "nnformer__fold0")
    assert sorted(mms2["subject_id"]) == ["mms2_009", "mms2_071"]
    assert mms2.set_index("subject_id").loc["mms2_009", "dice_lv"] == 0.9
    acdc = load_dice(path, cohort, "acdc", "nnformer__fold0")
    assert acdc["subject_id"].tolist() == ["acdc_patient101"]


def test_univariate_auc_direction_and_agreement():
    rows = []
    for vendor, sign in (("Siemens", 1), ("Philips", -1)):
        for y in (0, 1):
            for i in range(5):
                rows.append({"vendor": vendor, "y": y,
                             "ed_lv_volume_ml": 100 + 10 * y + i,  # higher in HCM on both
                             "ed_radiomics_original_glcm_Contrast": sign * (10 * y + i)})  # flips
    t = pd.DataFrame(rows)
    aucs = univariate_auc(t, ["ed_lv_volume_ml", "ed_radiomics_original_glcm_Contrast"])
    by = aucs.set_index(["vendor", "feature"])["auc"]
    assert by[("Siemens", "ed_lv_volume_ml")] == pytest.approx(1.0)
    assert by[("Philips", "ed_radiomics_original_glcm_Contrast")] == pytest.approx(0.0)
    assert (aucs["separation"] == 1.0).all()
    agree = auc_agreement(aucs)
    assert agree.loc["texture", "strong_flips"] == 1
    assert agree.loc["clinical", "direction_flips"] == 0
