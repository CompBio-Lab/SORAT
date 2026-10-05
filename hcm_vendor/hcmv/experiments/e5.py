"""External checks: E5 on ACDC and GE specificity.

E5: models tuned and refitted on all M&Ms-2 NOR/HCM subjects (n = 135, all three
vendors) predict the 20 ACDC test subjects (10 NOR, 10 HCM). n = 20 makes every CI
wide. Caveats: the ACDC masks come from an nnFormer trained on ACDC training data
(in-domain segmentations), and ACDC includes 3 T scans.

GE: models tuned and refitted on the Siemens + Philips pool (n = 114) predict the
21 GE subjects. Specificity on the 18 GE NOR is reported with a Wilson 95% CI; the
3 GE HCM predictions are listed only (no sensitivity claim).

Writes ``runs/E5/analysis/`` (``e5_metrics.csv``, ``e5_predictions.csv``,
``e5_roc.png``, ``summary.md``) and ``runs/GE/analysis/`` (``ge_specificity.csv``,
``ge_hcm_predictions.csv``, ``summary.md``).
"""

import numpy as np
import pandas as pd

from .. import figures
from ..qc import md_table
from ..stats import THRESHOLD, wilson_ci
from .common import analysis_dir, formatted, load_runs, metric_rows

SET_ORDER = ("clinical", "clinical+shape", "clinical+texture", "all", "all-no-wt")


def _order(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(_s=table.family_set.map({s: i for i, s in enumerate(SET_ORDER)}),
                        _m=table.model.map({m: i for i, m in enumerate(figures.MODEL_ORDER)})
                        ).sort_values(["_s", "_m"]).drop(columns=["_s", "_m"])


def _wilson_cell(k: int, n: int) -> str:
    low, high = wilson_ci(k, n)
    return f"{k}/{n} = {k / n:.2f} [{low:.2f}, {high:.2f}]"


def e5_report(config: dict) -> str:
    runs = load_runs(config, "E5")
    if not runs:
        raise FileNotFoundError("No completed E5 runs; run `run-experiment --experiment E5` first")
    out = analysis_dir(config, "E5")
    rows, predictions, curves = [], [], {}
    for (unit, family_set, model), run in runs.items():
        p = run["predictions"]
        y, prob = p["y"].to_numpy(), p["prob"].to_numpy()
        called = prob >= THRESHOLD
        tp, tn = int((called & (y == 1)).sum()), int((~called & (y == 0)).sum())
        rows.append(metric_rows(run["metrics"]["overall"], family_set=family_set, model=model,
                                n_train=run["metrics"]["n_train"], n_test=len(y),
                                sensitivity_wilson=_wilson_cell(tp, int((y == 1).sum())),
                                specificity_wilson=_wilson_cell(tn, int((y == 0).sum()))))
        predictions.append(p.assign(family_set=family_set, model=model))
        curves.setdefault(family_set, {})[model] = (figures.mean_roc(y, prob), run["metrics"]["overall"]["auc"])
    metrics = _order(pd.DataFrame(rows))
    metrics.to_csv(out / "e5_metrics.csv", index=False)
    wide = (pd.concat(predictions).pivot_table(index=["subject_id", "y"], columns=["family_set", "model"],
                                               values="prob").round(3))
    wide.to_csv(out / "e5_predictions.csv")
    sets = [s for s in SET_ORDER if s in curves]
    n_hcm = int(predictions[0]["y"].sum())
    figures.plot_external_roc({s: curves[s] for s in sets}, out / "e5_roc.png",
                              f"External test on ACDC ({n_hcm} HCM / {len(predictions[0]) - n_hcm} NOR), "
                              "trained on M&Ms-2 (n = 135)")
    keys = ["family_set", "model"]
    lines = ["# E5: external test on ACDC", "",
             "Trained (inner-CV tuned, refitted) on all M&Ms-2 NOR/HCM subjects, all vendors (n = 135); tested on "
             "the 20 ACDC test subjects. Bootstrap CIs; sensitivity and specificity also with Wilson CIs. With n = 20 "
             "every interval is wide.", "",
             "Caveats: ACDC masks come from an nnFormer trained on ACDC training data, so segmentations are "
             "in-domain and likely better than on M&Ms-2; ACDC includes 3 T scans (field strength per subject is "
             "not recorded).", "",
             md_table(formatted(metrics, keys).set_index(keys)), "",
             "## Sensitivity and specificity at 0.5 (Wilson 95% CI)", "",
             md_table(metrics[keys + ["sensitivity_wilson", "specificity_wilson"]].set_index(keys)), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)


def ge_report(config: dict) -> str:
    runs = load_runs(config, "GE")
    if not runs:
        raise FileNotFoundError("No completed GE runs; run `run-experiment --experiment GE` first")
    out = analysis_dir(config, "GE")
    rows, hcm = [], []
    for (unit, family_set, model), run in runs.items():
        p = run["predictions"]
        nor = p[p["y"] == 0]
        k = int((nor["prob"] < THRESHOLD).sum())
        low, high = wilson_ci(k, len(nor))
        rows.append({"family_set": family_set, "model": model, "n_nor": len(nor), "correct_nor": k,
                     "specificity": k / len(nor), "ci_low": low, "ci_high": high,
                     "median_prob_nor": float(nor["prob"].median())})
        for r in p[p["y"] == 1].itertuples():
            hcm.append({"family_set": family_set, "model": model, "subject_id": r.subject_id,
                        "prob": round(float(r.prob), 3), "called_hcm": bool(r.prob >= THRESHOLD)})
    table = _order(pd.DataFrame(rows))
    table.to_csv(out / "ge_specificity.csv", index=False)
    hcm = pd.DataFrame(hcm)
    hcm.to_csv(out / "ge_hcm_predictions.csv", index=False)
    show = table.assign(specificity=[_wilson_cell(k, n) for k, n in zip(table.correct_nor, table.n_nor)],
                        median_prob_nor=table.median_prob_nor.round(3))
    hcm_wide = _order(hcm).pivot_table(index=["family_set", "model"], columns="subject_id", values="prob",
                                       aggfunc="first", sort=False)
    lines = ["# GE specificity check", "",
             "Trained (inner-CV tuned, refitted) on the Siemens + Philips pool (n = 114), applied to the 21 GE "
             "subjects (never seen in training). Specificity on the 18 GE NOR at threshold 0.5 with a Wilson 95% CI.",
             "",
             md_table(show[["family_set", "model", "specificity", "median_prob_nor"]]
                      .set_index(["family_set", "model"])), "",
             "## Predicted P(HCM) for the 3 GE HCM subjects (descriptive only)", "",
             md_table(hcm_wide), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
