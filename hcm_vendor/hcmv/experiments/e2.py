"""E2 analysis (T41): cross-vendor transfer and the generalization gap Δ.

For a direction A→B, Δ_{A→B} = metric_{B→B} − metric_{A→B}, both on the same B subjects:

* metric_{B→B}: E1 nested CV inside vendor B (reading (a)), averaged over repeats;
* metric_{A→B}: the E2 model trained on all of A, predicting B (seed-averaged).

The CI and two-sided p come from a paired subject bootstrap over B
(:func:`hcmv.stats.paired_bootstrap`, which recomputes both terms on each resample).
For the Brier score, a positive Δ means the cross-vendor model is *better*
calibrated; for the other metrics a positive Δ is a loss from crossing vendors.

Writes ``runs/E2/analysis/``: ``e2_transfer_metrics.csv``, ``e2_gap.csv``,
``summary.md``, ``e2_gap_auc.png``, ``e2_roc_<family_set>.png``,
``e2_calibration_<family_set>.png`` and ``model_comparison_<family_set>.png``
(E1 and E2 together).
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .. import figures
from ..qc import md_table
from ..stats import METRICS, holm, paired_bootstrap
from .common import METRIC_NAMES, analysis_dir, format_p, formatted, load_runs, metric_rows, oof
from .e1 import comparison_frame

DIRECTIONS = ("siemens_to_philips", "philips_to_siemens")


def _aligned(e1_run: dict, e2_run: dict):
    """(y, P_within, p_cross) on the B subjects, rows in the same order."""
    ids, y, _, P_within = oof(e1_run)
    cross = e2_run["predictions"].set_index("subject_id").loc[ids]
    if not np.array_equal(cross["y"].to_numpy(), y):
        raise ValueError("E1 and E2 labels disagree for the same subjects")
    return y, P_within, cross["prob"].to_numpy()


def gap_table(e1: dict, e2: dict, config: dict, metrics=METRICS) -> pd.DataFrame:
    rows = []
    for (direction, family_set, model), run in e2.items():
        target = direction.split("_to_")[1]
        within = e1.get((target, family_set, model))
        if within is None:
            continue
        y, P_within, p_cross = _aligned(within, run)
        for metric in metrics:
            result = paired_bootstrap(y, P_within, p_cross, metric, n_boot=config["bootstrap_resamples"],
                                      seed=config["seed"])
            rows.append({"direction": direction, "family_set": family_set, "model": model, "n": len(y),
                         "within": within["metrics"]["overall"][metric]["estimate"],
                         "cross": run["metrics"]["overall"][metric]["estimate"], **result})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = np.nan
        for _, index in table.groupby(["direction", "family_set", "metric"]).groups.items():
            table.loc[index, "p_holm"] = holm(table.loc[index, "p_value"].to_numpy())
    return table


def transfer_metrics(e2: dict) -> pd.DataFrame:
    rows = [metric_rows(run["metrics"]["overall"], direction=d, family_set=fs, model=m, n_train=run["metrics"]["n_train"],
                        n_test=run["metrics"]["n"], n_seeds=run["metrics"]["n_seeds"])
            for (d, fs, m), run in e2.items()]
    return pd.DataFrame(rows)


def _calibration(y, prob, n_bins: int = 5):
    frac, mean = calibration_curve(y, prob, n_bins=n_bins, strategy="quantile")
    return mean, frac


def curves(e1: dict, e2: dict, family_set: str):
    roc, calibration = {}, {}
    for (direction, fs, model), run in e2.items():
        target = direction.split("_to_")[1]
        within = e1.get((target, fs, model))
        if fs != family_set or within is None:
            continue
        y, P_within, p_cross = _aligned(within, run)
        roc.setdefault(direction, {})[model] = {
            "within": (figures.mean_roc(y, P_within), within["metrics"]["overall"]["auc"]),
            "cross": (figures.mean_roc(y, p_cross), run["metrics"]["overall"]["auc"]),
            "n": (int(y.sum()), int((y == 0).sum())),
        }
        calibration.setdefault(direction, {})[model] = {
            "within": _calibration(y, P_within.mean(axis=1)),
            "cross": _calibration(y, p_cross),
            "brier": (within["metrics"]["overall"]["brier"]["estimate"], run["metrics"]["overall"]["brier"]["estimate"]),
        }
    return roc, calibration


def e2_report(config: dict) -> str:
    e2 = load_runs(config, "E2")
    if not e2:
        raise FileNotFoundError("No completed E2 runs; run `run-experiment --experiment E2` first")
    e1 = load_runs(config, "E1")
    out = analysis_dir(config, "E2")
    metrics = transfer_metrics(e2)
    metrics.to_csv(out / "e2_transfer_metrics.csv", index=False)
    gap = gap_table(e1, e2, config)
    gap.to_csv(out / "e2_gap.csv", index=False)

    family_sets = sorted({fs for (_, fs, _) in e2})
    for family_set in family_sets:
        roc, calibration = curves(e1, e2, family_set)
        if roc:
            figures.plot_transfer_roc(roc, out / f"e2_roc_{family_set}.png", family_set)
            figures.plot_transfer_calibration(calibration, out / f"e2_calibration_{family_set}.png", family_set)
        frame = comparison_frame(e1, family_set)
        for direction in DIRECTIONS:
            for model in figures.MODEL_ORDER:
                run = e2.get((direction, family_set, model))
                if run is not None:
                    frame = pd.concat([frame, pd.DataFrame([{"setting": f"E2 {figures.UNIT_NAMES[direction]}",
                                                             "model": model, **run["metrics"]["overall"]["auc"]}])])
        figures.plot_model_comparison(frame, out / f"model_comparison_{family_set}.png", family_set)
    if len(gap):
        figures.plot_gap_forest(gap[gap.metric == "auc"], out / "e2_gap_auc.png")

    lines = ["# E2: cross-vendor transfer and generalization gap", "",
             "Δ = within-vendor (E1 nested CV inside the test vendor) − cross-vendor (trained on the other vendor), "
             f"on the same test subjects; paired {config['bootstrap_resamples']}-resample subject bootstrap. "
             "Holm adjustment across the models within each direction, family set and metric.", "",
             "## Transfer performance (trained on all of one vendor, tested on the other)", "",
             md_table(formatted(metrics, ["direction", "family_set", "model", "n_test"])
                      .set_index(["direction", "family_set", "model", "n_test"])), ""]
    if len(gap):
        show = gap.assign(
            metric=gap["metric"].map(METRIC_NAMES), within=gap["within"].map("{:.3f}".format),
            cross=gap["cross"].map("{:.3f}".format), delta=gap["difference"].map("{:+.3f}".format),
            ci=[f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(gap.ci_low, gap.ci_high)],
            p=gap["p_value"].map(lambda v: format_p(v, config)),
            p_holm=gap["p_holm"].map(lambda v: format_p(v, config)))
        keys = ["direction", "family_set", "metric", "model"]
        lines += ["## Generalization gap Δ", "", md_table(show[keys + ["within", "cross", "delta", "ci", "p", "p_holm"]]
                                                       .set_index(keys)), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
