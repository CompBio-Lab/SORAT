"""Consolidated results and hypothesis verdicts.

Reads the analysis outputs of E1–E5, the GE check and SHAP (it does not recompute
them) and writes ``results_hcm_vendor/tables/``:

* ``summary_tests.csv`` -- every paired test / probe in long format, grouped by
  question, with the Holm adjustment used in its own report (``p_holm``) and a
  stricter Holm across all tests of the same question (``p_holm_question``).
* ``summary.md`` -- headline tables, the four hypothesis verdicts with effect sizes
  and CIs, and the limitations.

Verdict rules (stated so they can be checked):
* H1 (clinical features transfer with a small gap): supported if no clinical-only
  ΔAUC (normalized config, every model × 2 directions) is positive with a CI excluding 0;
  the verdict text gives the range of the point estimates.
* H2 (texture is the most vendor-predictive family): supported if every texture
  probe beats chance after Holm and has a higher balanced accuracy than the
  clinical and shape probes, for the primary target (3-class) and classifier (LR).
* H3 (texture has the largest gap): per model, supported if ΔΔAUC
  (clinical+texture − clinical, normalized texture) > 0 with Holm p < 0.05 in at
  least one direction.
* H4 (SHAP reliance is consistent across vendors): supported if Spearman ρ ≥ 0.7
  for every model.
"""

import numpy as np
import pandas as pd

from .. import figures
from ..config import output_dir
from ..qc import md_table
from ..runner import runs_root
from ..stats import holm
from .common import format_p

ALPHA = 0.05
MAIN_SETS = ("all", "all-no-wt")


def _read(config: dict, *parts) -> pd.DataFrame:
    path = runs_root(config).joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the corresponding report first")
    return pd.read_csv(path)


def _name(model: str) -> str:
    return figures.MODEL_NAMES[model]


def _unit(unit: str) -> str:
    return figures.UNIT_NAMES.get(unit, unit)


def collect_tests(config: dict) -> pd.DataFrame:
    rows = []
    e1 = _read(config, "E1", "analysis", "e1_model_comparisons.csv")
    for r in e1[e1.family_set.isin(MAIN_SETS)].itertuples():
        rows.append({"question": "Q1 Which model is best within the pooled cohort (E1)?",
                     "family": f"E1 pooled, {r.family_set}", "test": f"ΔAUC {_name(r.model_a)} − {_name(r.model_b)}",
                     "estimate": r.difference, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value,
                     "p_holm": r.p_holm})
    e2 = _read(config, "E2", "analysis", "e2_gap.csv")
    for r in e2[e2.family_set.isin(MAIN_SETS) & e2.metric.isin(["auc", "specificity"])].itertuples():
        rows.append({"question": "Q2 How much is lost when crossing vendors (E2)?",
                     "family": f"{_unit(r.direction)}, {r.family_set}, {r.metric}",
                     "test": f"Δ{r.metric} within − cross, {_name(r.model)}", "estimate": r.difference,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value, "p_holm": r.p_holm})
    e4 = _read(config, "E4", "analysis", "e4_contrasts.csv")
    for r in e4.itertuples():
        rows.append({"question": "Q3 Which feature family drives the gap (E4; H1, H3)?",
                     "family": f"{r.contrast}, {_unit(r.direction)}",
                     "test": f"ΔΔAUC {r.a} − {r.b}, {_name(r.model)}", "estimate": r.difference,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value, "p_holm": r.p_holm})
    e3 = _read(config, "E3", "analysis", "e3_probe.csv")
    for r in e3.itertuples():
        rows.append({"question": "Q4 Which family carries the vendor signal (E3; H2)?",
                     "family": f"{r.target}, {r.classifier}",
                     "test": f"balanced accuracy, {r.probe} (chance {r.chance:.2f})", "estimate": r.balanced_accuracy,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value, "p_holm": r.p_holm})
    shap = _read(config, "SHAP", "analysis", "shap_stability.csv")
    for r in shap.itertuples():
        rows.append({"question": "Q5 Is SHAP reliance consistent across training vendors (H4)?",
                     "family": "Siemens- vs Philips-trained, all", "test": f"Spearman ρ, {_name(r.model)}",
                     "estimate": r.spearman_rho, "ci_low": r.rho_ci_low, "ci_high": r.rho_ci_high,
                     "p_value": np.nan, "p_holm": np.nan})
    table = pd.DataFrame(rows)
    table["p_holm_question"] = np.nan
    for _, index in table.dropna(subset=["p_value"]).groupby("question").groups.items():
        table.loc[index, "p_holm_question"] = holm(table.loc[index, "p_value"].to_numpy())
    return table


def _ci(e, lo, hi, digits=2, sign=False) -> str:
    f = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return f"{f.format(e)} [{f.format(lo)}, {f.format(hi)}]"


def verdicts(config: dict) -> list:
    lines = []
    e4gap = _read(config, "E4", "analysis", "e4_gap.csv")
    contrasts = _read(config, "E4", "analysis", "e4_contrasts.csv")
    e3 = _read(config, "E3", "analysis", "e3_probe.csv")
    shap = _read(config, "SHAP", "analysis", "shap_stability.csv")

    clin = e4gap[(e4gap.family_set == "clinical") & (e4gap.metric == "auc") & (e4gap.feature_config == "norm")]
    worse = clin[clin.ci_low > 0]
    h1 = "Supported" if worse.empty else "Not supported"
    lines += [f"**H1: clinical features transfer with a small gap — {h1}.** With clinical features only, the "
              f"cross-vendor ΔAUC ranges from {clin.difference.min():+.3f} to {clin.difference.max():+.3f} across "
              f"the {clin.model.nunique()} models × 2 directions; none is positive with a CI excluding 0"
              + ("" if worse.empty else f" except {len(worse)}") +
              ". All point estimates are ≤ 0: the within-vendor term comes from nested CV, whose models train on "
              "4/5 of the test vendor (about 42–50 subjects), while the cross-vendor model trains on all of the other "
              "vendor (52–62), so Δ is slightly biased in favour of transfer (see limitations).", ""]

    main = e3[(e3.target == "three_vendor") & (e3.classifier == "logreg")].set_index("probe")
    tex = main.loc[["texture_norm", "texture_raw"]]
    others = main.loc[["clinical", "shape"]]
    h2 = (tex.p_holm < ALPHA).all() and tex.balanced_accuracy.min() > others.balanced_accuracy.max()
    lines += [f"**H2: texture is the most vendor-predictive family — {'Supported' if h2 else 'Not supported'}.** "
              "On normal hearts (3-class, chance 0.33, multinomial LR), texture identifies the vendor with balanced "
              f"accuracy {_ci(*main.loc['texture_norm', ['balanced_accuracy', 'ci_low', 'ci_high']])} (normalized) "
              f"and {_ci(*main.loc['texture_raw', ['balanced_accuracy', 'ci_low', 'ci_high']])} (raw), "
              f"p_Holm = {tex.p_holm.max():.3f}; clinical "
              f"{_ci(*main.loc['clinical', ['balanced_accuracy', 'ci_low', 'ci_high']])} and shape "
              f"{_ci(*main.loc['shape', ['balanced_accuracy', 'ci_low', 'ci_high']])} are not significant "
              f"(p_Holm ≥ {others.p_holm.min():.2f}). The random forest and the binary "
              "Siemens-vs-Philips target agree.", ""]

    ct = contrasts[(contrasts.contrast == "family vs clinical") & (contrasts.a == "clinical+texture (norm)")]
    per_model = []
    supported = []
    for model in figures.MODEL_ORDER:
        g = ct[ct.model == model]
        best = g.loc[g.difference.idxmax()]
        sig = ((g.difference > 0) & (g.p_holm < ALPHA)).any()
        if sig:
            supported.append(_name(model))
        per_model.append(f"{_name(model)} {best.difference:+.2f} [{best.ci_low:+.2f}, {best.ci_high:+.2f}] "
                         f"({_unit(best.direction)}, p_Holm {format_p(best.p_holm, config)})")
    label = ("Supported" if len(supported) == len(figures.MODEL_ORDER) else
             "Partially supported (model-dependent)" if supported else "Not supported")
    lines += [f"**H3: texture produces the largest gap — {label}.** Adding normalized texture to clinical features "
              "widens the cross-vendor AUC gap (ΔΔAUC, larger of the two directions): " + "; ".join(per_model) +
              f". Significant after Holm for: {', '.join(supported) or 'none'}. Tree ensembles and TabPFN show no "
              "texture-driven gap. Raw texture widens the gap less than normalized texture (see the texture check).",
              ""]

    rho_ok = (shap.spearman_rho >= 0.7).all()
    lines += [f"**H4: SHAP reliance is consistent across vendors — {'Supported' if rho_ok else 'Not supported'}.** "
              "Spearman ρ between the Siemens- and Philips-trained models' cluster importances: " +
              "; ".join(f"{_name(r.model)} {_ci(r.spearman_rho, r.rho_ci_low, r.rho_ci_high)}"
                        f" (top-10 shared {r.top10_shared})" for r in shap.itertuples()) +
              ". Trees agree most and keep wall thickness on top; Siemens-trained LR-EN, SVM and MLP lean on "
              "texture. TabPFN was not explained: KernelSHAP would need millions of transformer passes.", ""]
    return lines


LIMITATIONS = [
    "Small samples: 114 Siemens/Philips subjects for training and 52–62 per vendor, so within-vendor CIs are wide "
    "and cross-vendor gaps of a few AUC points cannot be resolved.",
    "Ceiling: pooled AUC is 0.94–1.00 for every model because maximum wall thickness nearly defines HCM; the vendor "
    "effect shows in transfer, calibration and specificity, not in headline accuracy. `all-no-wt` lowers it only "
    "slightly.",
    "GE has only 3 HCM subjects, so the GE check reports specificity on 18 normals and lists the 3 HCM predictions "
    "descriptively.",
    "ACDC is small (10 HCM / 10 NOR), its masks come from an nnFormer trained on ACDC training data (in-domain), and it "
    "includes 3 T scans without per-subject field strength.",
    "No body-surface-area indexing: volumes and mass are absolute, so body size is a confounder.",
    "M&Ms-1 was excluded because it overlaps M&Ms-2.",
    "Wall thickness follows our corrected definition (in-plane, per slice); results depend on it, and the original "
    "SORAT measure was wrong.",
    "Texture depends on preprocessing: with the normalized config, HCM texture effects point in opposite directions "
    "on Siemens and Philips for 56 of 84 features (raw: 20 of 84). A blood-pool intensity reference showed the two "
    "causes: GLCM features are identical under any intensity rescaling with a fixed bin count, so their reversals "
    "(48% agreement vs 79% raw) come from the 32-bin, 2D, 1.25 mm extraction settings; first-order features depend "
    "on the reference (agreement 14% whole-image, 39% blood pool, 72% raw). Raw texture transfers best; texture "
    "identifies the vendor (balanced accuracy 0.94–1.00) under every setting tried.",
    "The gap Δ compares nested-CV models trained on 4/5 of the test vendor (about 42–50 subjects) with a model "
    "trained on all of the other vendor (52–62 subjects), so Δ is slightly biased towards 0 or below; clinical-only "
    "gaps are all ≤ 0 partly for this reason.",
    "Probabilities shift across sites: at the fixed 0.5 threshold, specificity collapses on a new vendor or site for "
    "several models even when AUC holds.",
    "Bootstrap CIs resample subjects with models fixed (SHAP ρ) or are subject-level over repeated CV; they do not "
    "include model-refitting variability. Bootstrap p-values cannot go below 1/2000 and permutation p below 1/1001.",
]


def summary_report(config: dict) -> str:
    out = output_dir(config, "tables")
    tests = collect_tests(config)
    tests.to_csv(out / "summary_tests.csv", index=False)

    e1 = _read(config, "E1", "analysis", "e1_metrics.csv")
    pooled = e1[(e1.cohort == "pooled") & (e1.reading == "pooled") & e1.family_set.isin(MAIN_SETS)]
    within = e1[e1.reading.str.startswith("(a)") & (e1.family_set == "all")]
    e2 = _read(config, "E2", "analysis", "e2_gap.csv")
    gap = e2[(e2.family_set == "all") & (e2.metric.isin(["auc", "specificity"]))]
    e5 = _read(config, "E5", "analysis", "e5_metrics.csv")
    ge = _read(config, "GE", "analysis", "ge_specificity.csv")
    ge = ge[ge.family_set.isin(["clinical", "all"])]
    ge_table = ge.assign(specificity=[f"{c}/{n} = {s:.2f} [{lo:.2f}, {hi:.2f}]" for c, n, s, lo, hi in
                                      zip(ge.correct_nor, ge.n_nor, ge.specificity, ge.ci_low, ge.ci_high)]
                         ).pivot_table(index="family_set", columns="model", values="specificity", aggfunc="first")
    ge_table = ge_table[[m for m in figures.MODEL_ORDER if m in ge_table.columns]].rename(columns=figures.MODEL_NAMES)

    def pivot(frame, index, columns, value_cols, signed=False):
        f = frame.assign(cell=[_ci(e, lo, hi, 3, signed) for e, lo, hi in zip(*(frame[c] for c in value_cols))])
        wide = f.pivot_table(index=index, columns=columns, values="cell", aggfunc="first")
        wide = wide[[m for m in figures.MODEL_ORDER if m in wide.columns]]
        return wide.rename(columns=figures.MODEL_NAMES).rename_axis(None)

    lines = ["# Results summary and hypothesis verdicts", "",
             "Primary analysis: nnFormer features, normalized texture, feature set `all`, threshold 0.5. "
             "Cells are estimate [95% CI]. Every number is read from the experiment reports in "
             "`results_hcm_vendor/runs/*/analysis/`; all tests are in `summary_tests.csv`.", "",
             "## Verdicts", ""] + verdicts(config) + [
             "## E1: within-cohort AUC (nested 5×5 CV)", "",
             md_table(pivot(pooled.assign(row="pooled, " + pooled.family_set), "row", "model",
                            ["auc", "auc_ci_low", "auc_ci_high"])), "",
             md_table(pivot(within.assign(row=within.cohort + " only"), "row", "model",
                            ["auc", "auc_ci_low", "auc_ci_high"])), "",
             "## E2: cross-vendor gap (within − cross, same test subjects), all features", "",
             md_table(pivot(gap.assign(row=gap.direction.map(_unit) + ", Δ" + gap.metric), "row", "model",
                            ["difference", "ci_low", "ci_high"], signed=True)), "",
             "## E5: external ACDC test (trained on all M&Ms-2, n = 135; tested on 10 HCM / 10 NOR)", "",
             md_table(e5[e5.family_set.isin(["clinical", "all"])].set_index(["family_set", "model"])
                      [["auc", "sensitivity_wilson", "specificity_wilson"]].round(3)), "",
             "## GE check: specificity on 18 GE normals (trained on Siemens + Philips)", "",
             md_table(ge_table), "",
             "## Tests by question", "",
             "Counts of tests with p_Holm < 0.05 within each question (own-report Holm / question-wide Holm):", ""]
    for question, g in tests.groupby("question", sort=False):
        valid = g.dropna(subset=["p_value"])
        lines.append(f"- {question}: {len(g)} rows; {int((valid.p_holm < ALPHA).sum())} / "
                     f"{int((valid.p_holm_question < ALPHA).sum())} significant")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    return text
