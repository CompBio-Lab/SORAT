"""Follow-up analyses: ground-truth-mask features (S1) and ComBat harmonization (S3).

S1 asks how much of the cross-vendor gap comes from the *segmentation* rather than the
*images*: E1, E2 and E3 are rerun on features from the manual (ground-truth) masks
(``experiments.source=gt``, result store ``runs-gt``) and compared with the nnFormer
runs on the same subjects. ΔΔ = Δ(nnFormer) − Δ(ground truth); positive means the
automatic segmentation adds to the gap.

S3 asks whether ComBat harmonization closes the gap: E2C is E2 after ComBat (fitted on
both vendors' features, no labels). ΔΔ = Δ(E2) − Δ(E2C); positive means ComBat
narrows the gap. The within-vendor term (E1) is the same for both.

Both contrasts are paired subject bootstraps on the test vendor, with Holm
adjustment across models within each direction and family set. Outputs go to
``results_hcm_vendor/followups/{s1_ground_truth,s3_combat}/``.
"""

import copy

import numpy as np
import pandas as pd

from . import figures
from .config import output_dir
from .experiments.common import format_p, load_runs
from .experiments.e2 import DIRECTIONS, _aligned
from .experiments.e4 import gap_difference
from .qc import md_table
from .runner import runs_root
from .stats import holm, paired_bootstrap

SETS = ("clinical", "clinical+shape", "clinical+texture", "all", "all-no-wt")


def gt_config(config: dict) -> dict:
    out = copy.deepcopy(config)
    out["experiments"]["source"] = "gt"
    out["experiments"]["runs_dir"] = "runs-gt"
    return out


def _inputs(e1: dict, e2: dict, direction: str, family_set: str, model: str):
    target = direction.split("_to_")[1]
    within, cross = e1.get((target, family_set, model)), e2.get((direction, family_set, model))
    return None if within is None or cross is None else _aligned(within, cross)


def _gap(inputs, metric: str, config: dict, label: str) -> dict:
    """Gap estimate and paired-bootstrap CI for one analysis, as ``gap_<label>[_low|_high]``."""
    y, Pw, pc = inputs
    r = paired_bootstrap(y, Pw, pc, metric, n_boot=config["bootstrap_resamples"], seed=config["seed"])
    return {f"gap_{label}": r["difference"], f"gap_{label}_low": r["ci_low"], f"gap_{label}_high": r["ci_high"]}


def compare_gaps(a: tuple, b: tuple, config: dict, metrics=("auc", "specificity"), label_a="a",
                 label_b="b") -> pd.DataFrame:
    """ΔΔ = Δ(a) − Δ(b) for every direction x family set x model present in both (e1, e2) pairs."""
    rows = []
    for metric in metrics:
        for direction in DIRECTIONS:
            for family_set in SETS:
                for model in figures.MODEL_ORDER:
                    ia = _inputs(*a, direction, family_set, model)
                    ib = _inputs(*b, direction, family_set, model)
                    if ia is None or ib is None:
                        continue
                    if not np.array_equal(ia[0], ib[0]):
                        raise ValueError("the two analyses must cover the same test subjects")
                    rows.append({"metric": metric, "direction": direction, "family_set": family_set,
                                 "model": model, **_gap(ia, metric, config, label_a),
                                 **_gap(ib, metric, config, label_b),
                                 **gap_difference(ia, ib, config, metric)})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = np.nan
        for _, index in table.groupby(["metric", "direction", "family_set"]).groups.items():
            table.loc[index, "p_holm"] = holm(table.loc[index, "p_value"].to_numpy())
    return table


def _gap_rows(table: pd.DataFrame, label_a: str, label_b: str, names: dict, family_set: str) -> pd.DataFrame:
    """Long table for plot_gap_forest: one 'family_set' per analysis, with that analysis's gap CI."""
    rows = []
    t = table[(table.metric == "auc") & (table.family_set == family_set)]
    for label in (label_a, label_b):
        for r in t.itertuples():
            rows.append({"direction": r.direction, "model": r.model, "family_set": names[label],
                         "difference": getattr(r, f"gap_{label}"), "ci_low": getattr(r, f"gap_{label}_low"),
                         "ci_high": getattr(r, f"gap_{label}_high")})
    return pd.DataFrame(rows)


def _summary_lines(table: pd.DataFrame, config: dict, label_a: str, label_b: str, sets=("clinical", "all")) -> list:
    lines = []
    for metric in ("auc", "specificity"):
        t = table[(table.metric == metric) & table.family_set.isin(sets)]
        show = t.assign(
            cell=[f"{a:+.3f} vs {b:+.3f}; ΔΔ {d:+.3f} [{lo:+.3f}, {hi:+.3f}], p_Holm {format_p(p, config)}"
                  for a, b, d, lo, hi, p in zip(t[f"gap_{label_a}"], t[f"gap_{label_b}"], t.difference, t.ci_low,
                                                t.ci_high, t.p_holm)])
        wide = show.pivot_table(index=["direction", "family_set"], columns="model", values="cell", aggfunc="first")
        wide = wide[[m for m in figures.MODEL_ORDER if m in wide.columns]].rename(columns=figures.MODEL_NAMES)
        lines += [f"## Gap in {metric}: {label_a} vs {label_b}", "", md_table(wide), ""]
    return lines


def s1_report(config: dict) -> str:
    gt = gt_config(config)
    nn_runs = (load_runs(config, "E1"), load_runs(config, "E2"))
    gt_runs = (load_runs(gt, "E1"), load_runs(gt, "E2"))
    if not gt_runs[1]:
        raise FileNotFoundError("No ground-truth E2 runs in runs-gt; run E1/E2 with experiments.source=gt")
    out = output_dir(config, "followups", "s1_ground_truth")
    table = compare_gaps(nn_runs, gt_runs, config, label_a="nnformer", label_b="gt")
    table.to_csv(out / "s1_gap_nnformer_vs_gt.csv", index=False)

    e1 = []
    for name, runs in (("nnFormer", nn_runs[0]), ("ground truth", gt_runs[0])):
        for (unit, fs, model), run in runs.items():
            if fs in SETS:
                a = run["metrics"]["overall"]["auc"]
                e1.append({"masks": name, "cohort": unit, "family_set": fs, "model": model,
                           "auc": f"{a['estimate']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]"})
    e1 = pd.DataFrame(e1)
    e1.to_csv(out / "s1_e1_auc.csv", index=False)

    probe = []
    for name, cfg in (("nnFormer", config), ("ground truth", gt)):
        path = runs_root(cfg) / "E3" / "analysis" / "e3_probe.csv"
        if path.exists():
            probe.append(pd.read_csv(path).assign(masks=name))
    probe = pd.concat(probe) if probe else pd.DataFrame()
    if len(probe):
        probe.to_csv(out / "s1_e3_probe.csv", index=False)

    names = {"nnformer": "nnFormer masks", "gt": "ground-truth masks"}
    for fs in ("all", "clinical"):
        rows = _gap_rows(table, "nnformer", "gt", names, fs)
        if len(rows):
            figures.plot_gap_forest(rows, out / f"s1_gap_{fs}.png", labels={v: v for v in names.values()},
                                    title=f"Cross-vendor AUC gap, nnFormer vs ground-truth masks "
                                          f"({figures.FAMILY_SET_NAMES[fs]})")

    lines = ["# S1: cross-vendor gap with ground-truth vs nnFormer masks", "",
             "Gap Δ = within-vendor AUC (E1 nested CV) − cross-vendor AUC (E2), on the same test subjects. "
             "ΔΔ = Δ(nnFormer) − Δ(ground truth); positive = automatic segmentation adds to the gap. Paired "
             f"{config['bootstrap_resamples']}-resample bootstrap; Holm across models within direction and family "
             "set. Cells: Δ nnFormer vs Δ ground truth; ΔΔ [95% CI], p_Holm.", ""]
    lines += _summary_lines(table, config, "nnformer", "gt")
    pooled = e1[(e1.cohort == "pooled") & e1.family_set.isin(["clinical", "all"])]
    if len(pooled):
        wide = pooled.pivot_table(index=["masks", "family_set"], columns="model", values="auc", aggfunc="first")
        wide = wide[[m for m in figures.MODEL_ORDER if m in wide.columns]].rename(columns=figures.MODEL_NAMES)
        lines += ["## E1 pooled AUC", "", md_table(wide), ""]
    if len(probe):
        p = probe[(probe.target == "three_vendor") & (probe.classifier == "logreg")]
        p = p.assign(cell=[f"{e:.2f} [{lo:.2f}, {hi:.2f}]" for e, lo, hi in zip(p.balanced_accuracy, p.ci_low,
                                                                                 p.ci_high)])
        lines += ["## E3 vendor probe (3-class balanced accuracy, multinomial LR; chance 0.33)", "",
                  md_table(p.pivot_table(index="probe", columns="masks", values="cell", aggfunc="first")), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)


def s3_report(config: dict) -> str:
    e1 = load_runs(config, "E1")
    e2, e2c = load_runs(config, "E2"), load_runs(config, "E2C")
    if not e2c:
        raise FileNotFoundError("No E2C runs; run `run-experiment --experiment E2C` first")
    out = output_dir(config, "followups", "s3_combat")
    table = compare_gaps((e1, e2), (e1, e2c), config, label_a="raw", label_b="combat")
    table.to_csv(out / "s3_gap_without_vs_with_combat.csv", index=False)
    names = {"raw": "no harmonization", "combat": "ComBat"}
    for fs in ("all", "clinical"):
        rows = _gap_rows(table, "raw", "combat", names, fs)
        if len(rows):
            figures.plot_gap_forest(rows, out / f"s3_gap_{fs}.png", labels={v: v for v in names.values()},
                                    title=f"Cross-vendor AUC gap without vs with ComBat "
                                          f"({figures.FAMILY_SET_NAMES[fs]})")
    lines = ["# S3: does ComBat harmonization close the cross-vendor gap?", "",
             "Gap Δ = within-vendor (E1) − cross-vendor (E2 without, E2C with ComBat) on the same test subjects. "
             "ΔΔ = Δ(without) − Δ(with ComBat); positive = ComBat narrows the gap. ComBat is fitted on both "
             "vendors' features without labels (transductive), and without a disease covariate it also removes "
             "part of each vendor's disease mix. Paired bootstrap, Holm across models within direction and set. "
             "Cells: Δ without vs Δ with ComBat; ΔΔ [95% CI], p_Holm.", ""]
    lines += _summary_lines(table, config, "raw", "combat")
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
