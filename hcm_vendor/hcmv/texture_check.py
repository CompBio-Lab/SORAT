"""Why does normalized texture transfer across vendors worse than raw texture? (follow-up to T43)

For the texture features of the M&Ms-2 train pool (Siemens + Philips) under each
radiomics config (``norm`` and ``raw``), this measures:

1. **Vendor shift:** for every texture feature, how far the other vendor's NOR mean
   lies from the training vendor's NOR mean, in SDs of the training vendor's NOR
   (the scale the pipeline's StandardScaler uses).
2. **HCM effect agreement:** Cohen's d (HCM vs NOR) within Siemens and within
   Philips, and the share of features whose effect has the same sign on both vendors.
   A model that learns a texture-HCM relationship on one vendor can only transfer it
   if the sign agrees on the other.
3. **Texture survival:** texture features kept by the preprocessing (impute, variance,
   |r| > 0.95 filter) when fitted on each vendor.
4. **Reliance:** for the E2 LR-EN and MLP pipelines on ``clinical+texture``, the drop
   in AUC when the texture block is permuted across subjects (10 permutations, first
   two seeds), on the training vendor and on the test vendor. A negative drop on the
   test vendor means the learned texture relationship hurts there.

Writes ``results_hcm_vendor/qc/texture_check/`` (``per_feature.csv``, ``summary.csv``,
``reliance.csv``, ``summary.md``, ``texture_effect_agreement.png``).
"""

import glob
import json

import numpy as np
import pandas as pd

from . import figures
from .config import output_dir, repo_path
from .features import feature_family
from .preprocessing import build_preprocessor, select_features
from .qc import md_table
from .stats import fast_auc

CONFIGS = ("norm", "raw")
VENDORS = ("Siemens", "Philips")


def _cohens_d(a, b) -> float:
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / sd) if sd > 0 else np.nan


def _shift(train_nor, test_nor) -> float:
    sd = np.std(train_nor, ddof=1)
    return float((np.mean(test_nor) - np.mean(train_nor)) / sd) if sd > 0 else np.nan


def per_feature(table: pd.DataFrame, cfg: str) -> pd.DataFrame:
    pool = table[table["role"] == "train_pool"]
    columns = [c for c in select_features(pool, "all") if feature_family(c) == "texture"]
    by = {v: pool[pool["vendor"] == v] for v in VENDORS}
    rows = []
    for c in columns:
        s, p = by["Siemens"], by["Philips"]
        rows.append({
            "feature_config": cfg, "feature": c,
            "shift_siemens_to_philips": _shift(s.loc[s.y == 0, c], p.loc[p.y == 0, c]),
            "shift_philips_to_siemens": _shift(p.loc[p.y == 0, c], s.loc[s.y == 0, c]),
            "d_hcm_siemens": _cohens_d(s.loc[s.y == 1, c], s.loc[s.y == 0, c]),
            "d_hcm_philips": _cohens_d(p.loc[p.y == 1, c], p.loc[p.y == 0, c]),
        })
    out = pd.DataFrame(rows)
    out["same_sign"] = np.sign(out["d_hcm_siemens"]) == np.sign(out["d_hcm_philips"])
    return out


def survival(table: pd.DataFrame, cfg: str) -> dict:
    pool = table[table["role"] == "train_pool"]
    out = {}
    for v in VENDORS:
        data = pool[pool["vendor"] == v]
        kept = build_preprocessor().fit(data[select_features(data, "all")]).get_feature_names_out()
        out[f"texture_kept_{v.lower()}"] = int(sum(feature_family(k) == "texture" for k in kept))
    return out


def reliance(config: dict, tables: dict, family_set: str = "clinical+texture", n_perm: int = 10,
             n_seeds: int = 2) -> pd.DataFrame:
    import joblib

    rng = np.random.default_rng(config["seed"])
    root = repo_path(config, config["paths"]["output_root"]) / "runs" / "E2"
    rows = []
    for cfg, table in tables.items():
        pool = table[table["role"] == "train_pool"]
        for direction in ("siemens_to_philips", "philips_to_siemens"):
            source, target = (v.title() for v in direction.split("_to_"))
            for model in ("lr_en", "mlp"):
                run = root / direction / family_set / model / cfg
                if not (run / "manifest.json").exists():
                    continue
                features = json.loads((run / "hyperparams.json").read_text())["features"]
                texture = [f for f in features if feature_family(f) == "texture"]
                for path in sorted(glob.glob(str(run / "models" / "seed_*.joblib")))[:n_seeds]:
                    pipe = joblib.load(path)
                    for scope, vendor in (("train vendor", source), ("test vendor", target)):
                        data = pool[pool["vendor"] == vendor]
                        X, y = data[features], data["y"].to_numpy()
                        base = fast_auc(y, pipe.predict_proba(X)[:, 1])
                        drops = []
                        for _ in range(n_perm):
                            permuted = X.copy()
                            permuted[texture] = X[texture].to_numpy()[rng.permutation(len(X))]
                            drops.append(base - fast_auc(y, pipe.predict_proba(permuted)[:, 1]))
                        rows.append({"feature_config": cfg, "direction": direction, "model": model,
                                     "scope": scope, "auc": base, "texture_permutation_drop": float(np.mean(drops))})
    return (pd.DataFrame(rows).groupby(["feature_config", "direction", "model", "scope"], sort=False)
            .mean().reset_index())


def texture_check(config: dict) -> str:
    out = output_dir(config, "qc", "texture_check")
    source = config["experiments"].get("source") or config["segmentation_model_tag"]
    tables = {cfg: pd.read_parquet(repo_path(config, config["paths"]["output_root"]) / "tables" /
                                   f"features_mms2_{source.replace('__', '-')}_{cfg}.parquet") for cfg in CONFIGS}
    features = pd.concat([per_feature(tables[cfg], cfg) for cfg in CONFIGS], ignore_index=True)
    features.to_csv(out / "per_feature.csv", index=False)

    summary = []
    for cfg in CONFIGS:
        f = features[features.feature_config == cfg]
        summary.append({
            "feature_config": cfg, "n_texture": len(f),
            "median_abs_shift_S_to_P": f["shift_siemens_to_philips"].abs().median(),
            "median_abs_shift_P_to_S": f["shift_philips_to_siemens"].abs().median(),
            "median_abs_d_siemens": f["d_hcm_siemens"].abs().median(),
            "median_abs_d_philips": f["d_hcm_philips"].abs().median(),
            "same_sign_share": f["same_sign"].mean(),
            "same_sign_share_effect_ge_0.5": f.loc[(f.d_hcm_siemens.abs() >= 0.5) & (f.d_hcm_philips.abs() >= 0.5),
                                                   "same_sign"].mean(),
            **survival(tables[cfg], cfg),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    rel = reliance(config, tables)
    rel.to_csv(out / "reliance.csv", index=False)
    figures.plot_texture_effect_agreement(features, out / "texture_effect_agreement.png")

    lines = ["# Raw vs normalized texture: why the cross-vendor gap differs", "",
             "M&Ms-2 train pool (Siemens 30 HCM / 22 NOR, Philips 27 / 35), 84 texture features per config. "
             "Shift = other vendor's NOR mean minus the training vendor's NOR mean, in training-vendor NOR SDs. "
             "d = Cohen's d, HCM vs NOR, within a vendor.", "",
             md_table(summary.round(3).set_index("feature_config")), "",
             "## Texture reliance of the E2 models (clinical + texture)", "",
             "AUC drop when the texture block is permuted (mean of 10 permutations × 2 seeds). Negative on the "
             "test vendor = the learned texture relationship hurts there.", "",
             md_table(rel.round(3).set_index(["feature_config", "direction", "model", "scope"])), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
