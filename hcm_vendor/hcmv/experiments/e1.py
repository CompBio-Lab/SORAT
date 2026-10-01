"""E1 analysis (T40): pooled and within-vendor nested CV.

Reads ``runs/E1`` and writes ``runs/E1/analysis/``:

* ``e1_metrics.csv`` / ``.md`` -- every cohort x family set x model, two readings of
  within-vendor performance: (a) trained and tested inside one vendor (cohorts
  ``siemens`` / ``philips``) and (b) pooled-trained OOF predictions scored per vendor.
* ``e1_model_comparisons.csv`` -- paired bootstrap AUC differences between models on
  the pooled cohort, Holm-adjusted within each family set.
* ``e1_grid_edges.csv`` -- how often each grid axis was chosen at its lowest or
  highest value across outer folds.
* ``e1_roc_<family_set>.png`` and ``e1_model_comparison_<family_set>.png``.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .. import figures
from ..qc import md_table
from ..stats import holm, paired_bootstrap
from .common import analysis_dir, format_p, formatted, load_runs, metric_rows, oof

UNITS = ("pooled", "siemens", "philips")
READING_A = "(a) trained and tested within the cohort"
READING_B = "(b) pooled-trained, scored on one vendor"


def metrics_table(runs: dict) -> pd.DataFrame:
    rows = []
    for (unit, family_set, model), run in runs.items():
        m = run["metrics"]
        scope = "both vendors" if unit == "pooled" else unit.title()
        rows.append(metric_rows(m["overall"], family_set=family_set, model=model, cohort=unit,
                                scored_on=scope, reading=READING_A if unit != "pooled" else "pooled",
                                n=m["n"], n_hcm=m["n_positive"]))
        for vendor, block in m.get("by_vendor", {}).items():
            rows.append(metric_rows(block, family_set=family_set, model=model, cohort=unit, scored_on=vendor,
                                    reading=READING_B, n=block["n"], n_hcm=None))
    table = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(figures.MODEL_ORDER)}
    table["_m"] = table["model"].map(order)
    table["_u"] = table["cohort"].map({u: i for i, u in enumerate(UNITS)})
    return table.sort_values(["family_set", "_u", "scored_on", "_m"]).drop(columns=["_m", "_u"])


def model_comparisons(runs: dict, config: dict, unit: str = "pooled") -> pd.DataFrame:
    """Paired bootstrap ΔAUC for every model pair on one cohort, Holm within family set."""
    rows = []
    family_sets = sorted({fs for (u, fs, _) in runs if u == unit})
    for family_set in family_sets:
        models = [m for m in figures.MODEL_ORDER if (unit, family_set, m) in runs]
        block = []
        for a, b in combinations(models, 2):
            ids_a, y, _, P_a = oof(runs[(unit, family_set, a)])
            ids_b, _, _, P_b = oof(runs[(unit, family_set, b)])
            assert np.array_equal(ids_a, ids_b), "runs must cover the same subjects"
            result = paired_bootstrap(y, P_a, P_b, "auc", n_boot=config["bootstrap_resamples"], seed=config["seed"])
            block.append({"cohort": unit, "family_set": family_set, "model_a": a, "model_b": b, **result})
        if block:
            adjusted = holm([r["p_value"] for r in block])
            for r, p in zip(block, adjusted):
                r["p_holm"] = float(p)
            rows += block
    return pd.DataFrame(rows)


def _axis_order(values):
    """Sort grid values low -> high; None (unlimited depth) is highest, strings are unordered."""
    ordered = [v for v in values if not isinstance(v, str)]
    key = lambda v: (np.inf if v is None else (sum(v) if isinstance(v, (list, tuple)) else v))  # noqa: E731
    return sorted(ordered, key=key)


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and list(a) == list(b)
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.isclose(float(a), float(b)))


def grid_edges(runs: dict, config: dict, threshold: float = 0.5, tol: float = 1e-9) -> pd.DataFrame:
    """Share of outer folds choosing the lowest / highest value of each ordered grid axis.

    GridSearchCV takes the *first* grid point among ties, and at this AUC ceiling many points
    tie, so a choice at the first-listed value often means nothing. ``share_strict_*`` counts
    only folds where the edge value beat every grid point with a different value on that axis
    (needs ``grid_scores`` in hyperparams.json). ``at_edge`` uses the strict shares when available.
    """
    rows = []
    for (unit, family_set, model), run in runs.items():
        grid = config["models"][model]
        folds = run["hyperparams"]["folds"]
        for axis, values in grid.items():
            ordered = _axis_order(values)
            if len(ordered) < 2:
                continue
            low, high = ordered[0], ordered[-1]
            chosen = [f["best_params"][axis] for f in folds]
            row = {"cohort": unit, "family_set": family_set, "model": model, "axis": axis,
                   "n_folds": len(chosen), "lowest": str(low), "highest": str(high),
                   "share_lowest": float(np.mean([_same(c, low) for c in chosen])),
                   "share_highest": float(np.mean([_same(c, high) for c in chosen]))}
            for v in values:
                if isinstance(v, str):
                    row[f"share_{v}"] = float(np.mean([_same(c, v) for c in chosen]))
            strict = "grid_scores" in folds[0]
            if strict:
                for name, edge in (("lowest", low), ("highest", high)):
                    hits = []
                    for f in folds:
                        if not _same(f["best_params"][axis], edge):
                            hits.append(False)
                            continue
                        best = max(g["score"] for g in f["grid_scores"])
                        others = [g["score"] for g in f["grid_scores"] if not _same(g["params"][axis], edge)]
                        hits.append(not others or max(others) < best - tol)
                    row[f"share_strict_{name}"] = float(np.mean(hits))
            key = "share_strict_" if strict else "share_"
            row["at_edge"] = row[f"{key}lowest"] > threshold or row[f"{key}highest"] > threshold
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_frame(runs: dict, family_set: str) -> pd.DataFrame:
    """AUC per model for the model-comparison figure: pooled, and each within-vendor cohort."""
    rows = []
    for unit in UNITS:
        for model in figures.MODEL_ORDER:
            run = runs.get((unit, family_set, model))
            if run is None:
                continue
            auc = run["metrics"]["overall"]["auc"]
            rows.append({"setting": "E1 pooled" if unit == "pooled" else f"E1 {figures.UNIT_NAMES[unit]}",
                         "model": model, **auc})
    return pd.DataFrame(rows)


def roc_curves(runs: dict, family_set: str) -> dict:
    curves = {}
    for (unit, fs, model), run in runs.items():
        if fs != family_set:
            continue
        _, y, _, P = oof(run)
        curves.setdefault(unit, {})[model] = (figures.mean_roc(y, P), run["metrics"]["overall"]["auc"],
                                              int(y.sum()), int((y == 0).sum()))
    return curves


def e1_report(config: dict) -> str:
    runs = load_runs(config, "E1")
    if not runs:
        raise FileNotFoundError("No completed E1 runs; run `run-experiment --experiment E1` first")
    out = analysis_dir(config, "E1")
    metrics = metrics_table(runs)
    metrics.to_csv(out / "e1_metrics.csv", index=False)
    comparisons = model_comparisons(runs, config)
    comparisons.to_csv(out / "e1_model_comparisons.csv", index=False)
    edges = grid_edges(runs, config)
    edges.to_csv(out / "e1_grid_edges.csv", index=False)

    family_sets = sorted({fs for (_, fs, _) in runs})
    for family_set in family_sets:
        figures.plot_e1_roc(roc_curves(runs, family_set), out / f"e1_roc_{family_set}.png", family_set,
                            cv_text=f"{config['cv']['outer_repeats']}×{config['cv']['outer_splits']}")
        figures.plot_model_comparison(comparison_frame(runs, family_set),
                                      out / f"e1_model_comparison_{family_set}.png", family_set)

    lines = ["# E1: pooled and within-vendor nested CV", "",
             f"Runs: {len(runs)}. Outer CV {config['cv']['outer_repeats']}×{config['cv']['outer_splits']}, "
             f"inner {config['cv']['inner_splits']}-fold; CIs are {config['bootstrap_resamples']}-resample "
             "subject bootstraps of the repeat-averaged metric; threshold 0.5.", ""]
    keys = ["family_set", "cohort", "scored_on", "model", "n"]
    for reading in ("pooled", READING_A, READING_B):
        lines += [f"## {reading}", "", md_table(formatted(metrics[metrics.reading == reading], keys).set_index(keys)), ""]
    if len(comparisons):
        show = comparisons.assign(
            difference=comparisons["difference"].map("{:+.3f}".format),
            ci=[f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(comparisons.ci_low, comparisons.ci_high)],
            p=comparisons["p_value"].map(lambda v: format_p(v, config)),
            p_holm=comparisons["p_holm"].map(lambda v: format_p(v, config)))
        lines += ["## Paired model comparisons on the pooled cohort (ΔAUC = A − B)", "",
                  md_table(show[["family_set", "model_a", "model_b", "difference", "ci", "p", "p_holm"]]
                           .set_index(["family_set", "model_a", "model_b"])), ""]
    flagged = edges[edges["at_edge"]]
    lines += ["## Grid edge check", "",
              f"{len(flagged)} of {len(edges)} (cohort, family set, model, axis) combinations pick one edge value "
              "in more than half of the outer folds" + (", counting only folds where the edge value strictly beat "
              "every other value on that axis (ties go to the first grid point)." if "share_strict_lowest" in edges
              else "."), ""]
    if len(flagged):
        columns = [c for c in ("lowest", "share_lowest", "share_strict_lowest", "highest", "share_highest",
                               "share_strict_highest") if c in flagged]
        lines += [md_table(flagged[["cohort", "family_set", "model", "axis"] + columns].round(2)
                           .set_index(["cohort", "family_set", "model", "axis"])), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
