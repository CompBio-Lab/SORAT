import numpy as np
import pandas as pd
import pytest

from hcmv.qc import agreement, md_table, vendor_effect, wall_thickness_summary


def _table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for vendor, shift in (("Siemens", 0.0), ("Philips", 5.0), ("GE", 10.0)):
        for disease in ("NOR", "HCM"):
            for i in range(8):
                rows.append({
                    "subject_id": f"mms2_{vendor}_{disease}_{i}", "dataset": "mms2",
                    "vendor": vendor, "disease": disease,
                    "vendor_driven": shift + rng.normal(0, 0.5),
                    "noise": rng.normal(0, 1),
                    "ed_wall_thickness_max_mm": 15.0 if disease == "HCM" else 10.0,
                    "ed_wall_thickness_p95_mm": 12.0, "ed_wall_thickness_mean_mm": 7.0,
                })
    return pd.DataFrame(rows).set_index("subject_id")


def test_vendor_effect_separates_vendor_driven_from_noise():
    effects = vendor_effect(_table(), ["vendor_driven", "noise"]).set_index("feature")
    assert effects.loc["vendor_driven", "eta2"] > 0.8
    assert effects.loc["vendor_driven", "p"] < 1e-3
    assert effects.loc["noise", "eta2"] < 0.3


def test_wall_thickness_summary_groups():
    summary = wall_thickness_summary(_table())
    assert summary.loc["M&Ms-2 Philips HCM", "ed_wall_thickness_max_mm_median"] == 15.0
    assert summary.loc["M&Ms-2 GE NOR", "n"] == 8


def test_agreement_reports_bias_and_correlation():
    gt = _table()
    pred = gt.copy()
    pred["ed_wall_thickness_mean_mm"] = gt["ed_wall_thickness_mean_mm"] + 1.0
    pred["noise"] = gt["noise"] * 2.0
    out = agreement(pred, gt, ["noise", "ed_wall_thickness_mean_mm"])
    noise = out[(out.feature == "noise") & (out.vendor == "GE")].iloc[0]
    assert noise["pearson_r"] == pytest.approx(1.0)
    shifted = out[(out.feature == "ed_wall_thickness_mean_mm") & (out.vendor == "GE")]
    assert shifted.empty or shifted.iloc[0]["bias"] == pytest.approx(1.0)


def test_md_table_renders_header_and_nan():
    text = md_table(pd.DataFrame({"a": [1.0, np.nan]}, index=pd.Index(["x", "y"], name="g")))
    lines = text.splitlines()
    assert lines[0] == "| g | a |"
    assert lines[3] == "| y |  |"
