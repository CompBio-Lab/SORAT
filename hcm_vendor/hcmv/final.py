"""Final tables and figures for the report: one command each, read from the result store.

``python -m hcmv tables`` writes ``results_hcm_vendor/tables/final/`` (CSV + Markdown per
table) and the model-pair tests that figure 9 marks; ``python -m hcmv figures`` writes
the numbered figures to ``results_hcm_vendor/figures/`` as PNG and PDF. Both read the
experiment reports (``runs/*/analysis``), the run stores and the cached SHAP values;
run the experiment reports first (``e1-report`` … ``e5-report``, ``e3-probe``,
``shap``, ``texture-check``). ``tables`` must run before ``figures``.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd

from . import figures
from .config import output_dir, repo_path
from .experiments.common import load_runs
from .experiments.e1 import model_comparisons
from .experiments.e2 import _aligned
from .qc import md_table
from .runner import runs_root
from .stats import holm, paired_bootstrap

MAIN_SET = "all"


def _analysis(config: dict, experiment: str, name: str) -> pd.DataFrame:
    path = runs_root(config) / experiment / "analysis" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the {experiment} report first")
    return pd.read_csv(path)


def _ci(e, lo, hi, digits=3, signed=False) -> str:
    f = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return f"{f.format(e)} [{f.format(lo)}, {f.format(hi)}]"


def _models(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [m for m in figures.MODEL_ORDER if m in frame.columns]
    return frame[cols].rename(columns=figures.MODEL_NAMES)


# ----------------------------------------------------------------------------- model pairs
def transfer_model_comparisons(e1: dict, e2: dict, config: dict, family_set: str = MAIN_SET) -> pd.DataFrame:
    """Paired bootstrap ΔAUC between models on the same cross-vendor test subjects, Holm per direction."""
    rows = []
    for direction in ("siemens_to_philips", "philips_to_siemens"):
        target = direction.split("_to_")[1]
        preds = {}
        for model in figures.MODEL_ORDER:
            if (direction, family_set, model) in e2 and (target, family_set, model) in e1:
                y, _, p = _aligned(e1[(target, family_set, model)], e2[(direction, family_set, model)])
                preds[model] = (y, p)
        block = []
        for a, b in combinations(preds, 2):
            y = preds[a][0]
            block.append({"setting": direction, "family_set": family_set, "model_a": a, "model_b": b,
                          **paired_bootstrap(y, preds[a][1], preds[b][1], "auc", n_boot=config["bootstrap_resamples"],
                                             seed=config["seed"])})
        for r, p in zip(block, holm([r["p_value"] for r in block])):
            r["p_holm"] = float(p)
        rows += block
    return pd.DataFrame(rows)


def model_pairs(config: dict) -> pd.DataFrame:
    e1 = load_runs(config, "E1")
    e2 = load_runs(config, "E2")
    within = [model_comparisons({k: v for k, v in e1.items() if k[1] == MAIN_SET}, config, unit)
              .rename(columns={"cohort": "setting"}) for unit in ("pooled", "siemens", "philips")]
    return pd.concat(within + [transfer_model_comparisons(e1, e2, config)], ignore_index=True)


# ----------------------------------------------------------------------------- tables
def make_tables(config: dict) -> list:
    out = output_dir(config, "tables", "final")
    written = []

    def save(name, frame, title, note=""):
        frame.to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(f"# {title}\n\n{note}\n\n{md_table(frame)}\n" if note else
                                        f"# {title}\n\n{md_table(frame)}\n")
        written.append(name)

    cohort = pd.read_parquet(repo_path(config, config["paths"]["output_root"]) / "tables" / "cohort.parquet")
    counts = cohort.pivot_table(index=["dataset", "vendor"], columns="disease", values="source_id",
                                aggfunc="count", fill_value=0)
    save("table1_cohort", counts[["NOR", "HCM"]], "Cohort",
         "Subjects per dataset, vendor and diagnosis. M&Ms-2 Siemens + Philips form the training pool; "
         "GE is the specificity check; ACDC is the external test.")

    e1 = _analysis(config, "E1", "e1_metrics.csv")
    e1 = e1[(e1.family_set.isin([MAIN_SET, "all-no-wt"])) & e1.reading.isin(["pooled"]) |
            ((e1.family_set == MAIN_SET) & e1.reading.str.startswith("(a)"))]
    e1 = e1.assign(row=e1.family_set + " | " + e1.cohort,
                   cell=[_ci(*v) for v in zip(e1.auc, e1.auc_ci_low, e1.auc_ci_high)])
    save("table2_e1_auc", _models(e1.pivot_table(index="row", columns="model", values="cell", aggfunc="first")),
         "E1: AUC [95% CI], nested 5×5 cross-validation", "Rows: feature set | cohort.")

    gap = _analysis(config, "E2", "e2_gap.csv")
    gap = gap[(gap.family_set == MAIN_SET) & gap.metric.isin(["auc", "balanced_accuracy", "specificity", "brier"])]
    gap = gap.assign(row=gap.direction.map(figures.UNIT_NAMES) + " | Δ" + gap.metric,
                     cell=[_ci(d, lo, hi, signed=True) + ("*" if p < 0.05 else "")
                           for d, lo, hi, p in zip(gap.difference, gap.ci_low, gap.ci_high, gap.p_holm)])
    save("table3_e2_gap", _models(gap.pivot_table(index="row", columns="model", values="cell", aggfunc="first")),
         "E2: cross-vendor gap Δ = within − cross (all features)",
         "Same test subjects; paired bootstrap 95% CI; * = Holm p < 0.05 across models.")

    e3 = _analysis(config, "E3", "e3_probe.csv")
    e3 = e3.assign(cell=[_ci(e, lo, hi) + f", p_Holm {p:.3f}" for e, lo, hi, p in
                         zip(e3.balanced_accuracy, e3.ci_low, e3.ci_high, e3.p_holm)])
    save("table4_e3_probe", e3.pivot_table(index="probe", columns=["target", "classifier"], values="cell",
                                           aggfunc="first"),
         "E3: vendor prediction from normal hearts (balanced accuracy [95% CI])",
         "Chance: 0.333 (3-class), 0.5 (Siemens vs Philips). Permutation p (1000 shuffles), Holm across probes.")

    ct = _analysis(config, "E4", "e4_contrasts.csv")
    ct = ct.assign(row=ct.direction.map(figures.UNIT_NAMES) + " | " + ct.a + " − " + ct.b,
                   cell=[_ci(d, lo, hi, signed=True) + ("*" if p < 0.05 else "")
                         for d, lo, hi, p in zip(ct.difference, ct.ci_low, ct.ci_high, ct.p_holm)])
    save("table5_e4_contrasts", _models(ct.pivot_table(index="row", columns="model", values="cell", aggfunc="first")),
         "E4: change in the cross-vendor AUC gap (ΔΔAUC)",
         "Positive = the first feature set transfers worse. * = Holm p < 0.05 within contrast type and direction.")

    e5 = _analysis(config, "E5", "e5_metrics.csv")
    e5 = e5.assign(cell=[f"AUC {_ci(a, lo, hi, 2)}; sens {s.split(' [')[0]}; spec {p.split(' [')[0]}"
                         for a, lo, hi, s, p in zip(e5.auc, e5.auc_ci_low, e5.auc_ci_high, e5.sensitivity_wilson,
                                                    e5.specificity_wilson)])
    save("table6_e5_acdc", _models(e5.pivot_table(index="family_set", columns="model", values="cell",
                                                  aggfunc="first")),
         "E5: external test on ACDC (10 HCM / 10 NOR)", "Trained on all M&Ms-2 NOR/HCM (n = 135).")

    ge = _analysis(config, "GE", "ge_specificity.csv")
    ge = ge.assign(cell=[f"{c}/{n} = {_ci(s, lo, hi, 2)}" for c, n, s, lo, hi in
                         zip(ge.correct_nor, ge.n_nor, ge.specificity, ge.ci_low, ge.ci_high)])
    save("table7_ge_specificity", _models(ge.pivot_table(index="family_set", columns="model", values="cell",
                                                         aggfunc="first")),
         "GE check: specificity on 18 GE normal subjects (Wilson 95% CI)", "Trained on Siemens + Philips (n = 114).")

    shap = _analysis(config, "SHAP", "shap_stability.csv").set_index("model")
    shap = shap.assign(rho=[_ci(*v, 2) for v in zip(shap.spearman_rho, shap.rho_ci_low, shap.rho_ci_high)])
    save("table8_shap_stability", shap[["rho", "top10_shared", "top10_jaccard"]].round(2)
         .rename(index=figures.MODEL_NAMES),
         "SHAP stability between Siemens- and Philips-trained models",
         "Spearman ρ of correlation-cluster importance (95% CI over explained subjects); top-10 clusters shared.")

    pairs = model_pairs(config)
    pairs.to_csv(out / "model_pairs.csv", index=False)
    written.append("model_pairs")
    return written


# ----------------------------------------------------------------------------- figures
def _best_marks(frame: pd.DataFrame, pairs: pd.DataFrame, settings: dict) -> set:
    """(setting label, model) pairs whose AUC is below the setting's best model with Holm p < 0.05."""
    marks = set()
    for key, label in settings.items():
        rows = frame[frame.setting == label].reset_index(drop=True)
        if rows.empty:
            continue
        best = rows.loc[rows.estimate.idxmax(), "model"]
        tests = pairs[pairs.setting == key]
        for model in rows.model:
            if model == best:
                continue
            t = tests[((tests.model_a == best) & (tests.model_b == model)) |
                      ((tests.model_a == model) & (tests.model_b == best))]
            if len(t) and t.p_holm.iloc[0] < 0.05:
                marks.add((label, model))
    return marks


def make_figures(config: dict) -> list:
    from .experiments import e1 as e1_mod
    from .experiments import e2 as e2_mod
    from .experiments.e3 import CLASSIFIER_NAMES, PROBE_NAMES, TARGET_NAMES
    from .experiments.e5 import SET_ORDER

    figures.EXPORT_PDF = True
    out = output_dir(config, "figures")
    root = repo_path(config, config["paths"]["output_root"])
    written = []

    cohort = pd.read_parquet(root / "tables" / "cohort.parquet")
    counts = {v: int(((cohort.dataset == "mms2") & (cohort.vendor == v)).sum()) for v in figures.VENDOR_ORDER}
    counts["ACDC"] = int((cohort.dataset == "acdc").sum())
    figures.plot_workflow(out / "fig01_workflow.png", counts)
    written.append("fig01_workflow")
    source = config["experiments"].get("source") or config["segmentation_model_tag"]
    table = pd.read_parquet(root / "tables" / f"features_mms2_{source.replace('__', '-')}_norm.parquet")
    figures.plot_cohort_panel(cohort, table, out / "fig02_cohort.png")
    written.append("fig02_cohort")

    e1 = load_runs(config, "E1")
    figures.plot_e1_roc(e1_mod.roc_curves(e1, MAIN_SET), out / "fig03_e1_roc.png", MAIN_SET,
                        cv_text=f"{config['cv']['outer_repeats']}×{config['cv']['outer_splits']}")
    written.append("fig03_e1_roc")

    gap = _analysis(config, "E4", "e4_gap.csv")
    norm = gap[(gap.metric == "auc") & (gap.feature_config == "norm") &
               gap.family_set.isin(["clinical", "clinical+shape", "clinical+texture", "all"])]
    norm = norm.assign(_o=norm.family_set.map({s: i for i, s in enumerate(SET_ORDER)})).sort_values("_o")
    figures.plot_gap_forest(norm, out / "fig04_gap_by_family.png",
                            title="Cross-vendor AUC gap by feature family")
    written.append("fig04_gap_by_family")

    probe = _analysis(config, "E3", "e3_probe.csv")
    main = probe[(probe.target == "three_vendor") & (probe.classifier == "logreg")]
    results = []
    for r in main.itertuples():
        stored = json.loads((runs_root(config) / "E3" / r.target / r.classifier / f"{r.probe}.json").read_text())
        results.append({"family": r.probe, "label": PROBE_NAMES[r.probe], "observed": r.observed_first_repeat,
                        "null": np.asarray(stored["null"]), "chance": r.chance, "p": r.p_value, "p_holm": r.p_holm})
    figures.plot_probe_null(results, out / "fig05_vendor_probe.png",
                            f"Vendor prediction from normal hearts: {TARGET_NAMES['three_vendor']}, "
                            f"{CLASSIFIER_NAMES['logreg']} (n = {int(main.n.iloc[0])})")
    written.append("fig05_vendor_probe")

    e5 = load_runs(config, "E5")
    curves = {}
    for (unit, family_set, model), run in e5.items():
        if family_set in ("clinical", MAIN_SET):
            p = run["predictions"]
            curves.setdefault(family_set, {})[model] = (figures.mean_roc(p.y.to_numpy(), p.prob.to_numpy()),
                                                         run["metrics"]["overall"]["auc"])
    figures.plot_external_roc({s: curves[s] for s in ("clinical", MAIN_SET) if s in curves},
                              out / "fig06_acdc_roc.png",
                              "External test on ACDC (10 HCM / 10 NOR), trained on M&Ms-2 (n = 135)")
    written.append("fig06_acdc_roc")

    importances = _analysis(config, "SHAP", "shap_cluster_importance.csv")
    stability = _analysis(config, "SHAP", "shap_stability.csv")
    shares = _analysis(config, "SHAP", "shap_family_share.csv")
    figures.plot_shap_rank_scatter(importances, stability, out / "fig07a_shap_ranks.png")
    figures.plot_shap_family_share(shares, out / "fig07b_shap_family_share.png")
    written += ["fig07a_shap_ranks", "fig07b_shap_family_share"]

    e2 = load_runs(config, "E2")
    _, calibration = e2_mod.curves(e1, e2, MAIN_SET)
    figures.plot_transfer_calibration(calibration, out / "fig08_calibration_shift.png", MAIN_SET)
    written.append("fig08_calibration_shift")

    frame = e1_mod.comparison_frame(e1, MAIN_SET)
    for direction in ("siemens_to_philips", "philips_to_siemens"):
        for model in figures.MODEL_ORDER:
            run = e2.get((direction, MAIN_SET, model))
            if run is not None:
                frame = pd.concat([frame, pd.DataFrame([{"setting": f"E2 {figures.UNIT_NAMES[direction]}",
                                                         "model": model, **run["metrics"]["overall"]["auc"]}])])
    pairs = pd.read_csv(output_dir(config, "tables", "final") / "model_pairs.csv")
    settings = {"pooled": "E1 pooled", "siemens": "E1 Siemens only", "philips": "E1 Philips only",
                "siemens_to_philips": "E2 Siemens → Philips", "philips_to_siemens": "E2 Philips → Siemens"}
    figures.plot_model_comparison(frame, out / "fig09_model_comparison.png", MAIN_SET,
                                  marks=_best_marks(frame, pairs, settings),
                                  mark_note="* lower AUC than the best model in that setting "
                                            "(paired bootstrap, Holm p < 0.05)",
                                  title="Model-family comparison within and across vendors (all features)")
    written.append("fig09_model_comparison")
    figures.EXPORT_PDF = False
    return written


# ----------------------------------------------------------------------------- reproducibility
def compare_stores(config: dict, other: str) -> pd.DataFrame:
    """Overall metric estimates of every run in the primary store vs the same run in ``other``."""
    root = repo_path(config, config["paths"]["output_root"])
    primary, rerun = root / config["experiments"].get("runs_dir", "runs"), root / other
    rows = []
    for path in sorted(primary.rglob("metrics.json")):
        twin = rerun / path.relative_to(primary)
        if "analysis" in path.parts or not twin.exists():
            continue
        a, b = json.loads(path.read_text())["overall"], json.loads(twin.read_text())["overall"]
        for metric in a:
            rows.append({"run": str(path.parent.relative_to(primary)), "metric": metric,
                         "stored": a[metric]["estimate"], "rerun": b[metric]["estimate"],
                         "abs_diff": abs(a[metric]["estimate"] - b[metric]["estimate"])})
    for path in sorted((primary / "E3").rglob("*.json")):
        twin = rerun / path.relative_to(primary)
        if twin.exists() and "analysis" not in path.parts and path.name != "manifest.json":
            a, b = json.loads(path.read_text()), json.loads(twin.read_text())
            rows.append({"run": str(path.relative_to(primary)), "metric": "balanced_accuracy",
                         "stored": a["balanced_accuracy"], "rerun": b["balanced_accuracy"],
                         "abs_diff": abs(a["balanced_accuracy"] - b["balanced_accuracy"])})
    return pd.DataFrame(rows)
