import numpy as np
import pandas as pd

from hcmv.shap_analysis import cluster_importance, correlation_clusters, family_share, stability
from hcmv.texture_check import _cohens_d, _shift


def _pool(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    return pd.DataFrame({
        "ed_wall_thickness_max_mm": base,
        "ed_myocardial_mass_g": base * 2 + rng.normal(0, 0.01, n),  # |r| > 0.95 with wall thickness
        "ed_radiomics_original_glcm_Contrast": rng.normal(size=n),
        "ed_radiomics_original_shape_Sphericity": rng.normal(size=n),
    })


def test_correlated_features_share_a_cluster_named_by_priority():
    clusters = correlation_clusters(_pool())
    assert clusters["ed_wall_thickness_max_mm"] == clusters["ed_myocardial_mass_g"] == "ed_myocardial_mass_g"
    assert clusters.nunique() == 3


def test_cluster_importance_counts_dropped_features_as_zero():
    clusters = correlation_clusters(_pool())
    values = pd.DataFrame({"ed_wall_thickness_max_mm": [1.0, -1.0], "ed_radiomics_original_glcm_Contrast": [0.5, 0.5]})
    imp = cluster_importance(values, clusters)
    assert imp["ed_myocardial_mass_g"] == 1.0 and imp["ed_radiomics_original_glcm_Contrast"] == 0.5
    assert imp["ed_radiomics_original_shape_Sphericity"] == 0.0


def test_identical_attributions_are_perfectly_stable():
    clusters = correlation_clusters(_pool())
    rng = np.random.default_rng(1)
    values = pd.DataFrame(rng.normal(size=(30, 3)) * [3, 2, 1], columns=["ed_myocardial_mass_g",
                          "ed_radiomics_original_glcm_Contrast", "ed_radiomics_original_shape_Sphericity"])
    result = stability(values, values, clusters, n_boot=50, seed=0, top=2)
    assert result["spearman_rho"] == 1.0 and result["top2_jaccard"] == 1.0
    share = family_share(values)
    assert np.isclose(share.sum(), 1.0) and share["clinical"] > share["texture"] > share["shape"]


def test_effect_size_and_shift_helpers():
    assert _cohens_d(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0])) == 2.0
    assert np.isclose(_shift(np.array([0.0, 2.0]), np.array([3.0, 5.0])), 3 / np.std([0.0, 2.0], ddof=1))
