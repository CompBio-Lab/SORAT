"""E4 analysis (T43): feature-family ablation of the cross-vendor gap, plus raw vs normalized texture.

Uses the E1 and E2 runs of every family set (``clinical``, ``clinical+shape``,
``clinical+texture``, ``all``) and both feature configs (``norm``; ``raw`` for the
sets that contain texture). For each direction A→B, family set and model, the gap
is Δ = AUC_{B→B} − AUC_{A→B} on the same B subjects (as in E2).

Hypothesis-relevant contrasts, each a paired subject bootstrap over B of a
*difference of gaps* (both gaps recomputed on every resample):

* H1/H3: Δ(set) − Δ(clinical) for clinical+shape, clinical+texture and all;
  positive means adding that family widens the gap.
* Texture preprocessing: Δ(set, raw) − Δ(set, norm) for the sets with texture.

Holm adjustment within each direction and contrast type, across models and sets.
Writes ``runs/E4/analysis/``: ``e4_e1_auc.csv``, ``e4_gap.csv``, ``e4_contrasts.csv``,
``summary.md``, ``e4_gap_auc.png`` and ``e4_gap_auc_raw_vs_norm.png``.
"""

import numpy as np
import pandas as pd

from .. import figures
from ..qc import md_table
from ..stats import holm, repeated_metric, stratified_bootstrap_indices
from .common import analysis_dir, format_p, load_runs
from .e2 import DIRECTIONS, _aligned, gap_table

ABLATION_SETS = ("clinical", "clinical+shape", "clinical+texture", "all")
TEXTURE_SETS = ("clinical+texture", "all")
CONFIGS = ("norm", "raw")


def _gap_inputs(e1: dict, e2: dict, direction: str, family_set: str, model: str):
    target = direction.split("_to_")[1]
    within, cross = e1.get((target, family_set, model)), e2.get((direction, family_set, model))
    if within is None or cross is None:
        return None
    return _aligned(within, cross)


def gap_difference(inputs_a, inputs_b, config: dict, metric: str = "auc", alpha: float = 0.05) -> dict:
    """Δ_a − Δ_b with a paired bootstrap over the shared test subjects (same rows for all four terms)."""
    y, Pw_a, pc_a = inputs_a
    y_b, Pw_b, pc_b = inputs_b
    if not np.array_equal(y, y_b):
        raise ValueError("gap contrasts need the same test subjects in the same order")

    def delta(rows=None):
        gap_a = repeated_metric(y, Pw_a, metric, rows) - repeated_metric(y, pc_a, metric, rows)
        gap_b = repeated_metric(y, Pw_b, metric, rows) - repeated_metric(y, pc_b, metric, rows)
        return gap_a - gap_b

    draws = np.array([delta(rows) for rows in stratified_bootstrap_indices(y, config["bootstrap_resamples"],
                                                                           config["seed"])])
    draws = draws[~np.isnan(draws)]
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return {"difference": float(delta()), "ci_low": float(low), "ci_high": float(high), "p_value": float(p)}


def e1_auc_table(e1_by_config: dict) -> pd.DataFrame:
    rows = []
    for cfg, runs in e1_by_config.items():
        for (unit, family_set, model), run in runs.items():
            auc = run["metrics"]["overall"]["auc"]
            rows.append({"feature_config": cfg, "cohort": unit, "family_set": family_set, "model": model, **auc})
    return pd.DataFrame(rows)


def contrasts(e1: dict, e2: dict, config: dict) -> pd.DataFrame:
    """``e1``/``e2``: {feature_config: runs}."""
    rows = []
    models = figures.MODEL_ORDER
    for direction in DIRECTIONS:
        for model in models:
            base = _gap_inputs(e1["norm"], e2["norm"], direction, "clinical", model)
            for family_set in ABLATION_SETS[1:]:
                other = _gap_inputs(e1["norm"], e2["norm"], direction, family_set, model)
                if base is None or other is None:
                    continue
                rows.append({"contrast": "family vs clinical", "direction": direction, "model": model,
                             "a": f"{family_set} (norm)", "b": "clinical (norm)",
                             **gap_difference(other, base, config)})
            for family_set in TEXTURE_SETS:
                norm = _gap_inputs(e1["norm"], e2["norm"], direction, family_set, model)
                raw = _gap_inputs(e1.get("raw", {}), e2.get("raw", {}), direction, family_set, model)
                if norm is None or raw is None:
                    continue
                rows.append({"contrast": "raw vs norm", "direction": direction, "model": model,
                             "a": f"{family_set} (raw)", "b": f"{family_set} (norm)",
                             **gap_difference(raw, norm, config)})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = np.nan
        for _, index in table.groupby(["contrast", "direction"]).groups.items():
            table.loc[index, "p_holm"] = holm(table.loc[index, "p_value"].to_numpy())
    return table


def e4_report(config: dict) -> str:
    e1 = {cfg: load_runs(config, "E1", cfg) for cfg in CONFIGS}
    e2 = {cfg: load_runs(config, "E2", cfg) for cfg in CONFIGS}
    keep = lambda runs, sets: {k: v for k, v in runs.items() if k[1] in sets}  # noqa: E731
    e1 = {"norm": keep(e1["norm"], ABLATION_SETS), "raw": keep(e1["raw"], TEXTURE_SETS)}
    e2 = {"norm": keep(e2["norm"], ABLATION_SETS), "raw": keep(e2["raw"], TEXTURE_SETS)}
    if not e2["norm"]:
        raise FileNotFoundError("No E2 runs for the ablation family sets")
    out = analysis_dir(config, "E4")

    aucs = e1_auc_table(e1)
    aucs.to_csv(out / "e4_e1_auc.csv", index=False)
    gaps = []
    for cfg in CONFIGS:
        if e2[cfg]:
            gaps.append(gap_table(e1[cfg], e2[cfg], config, metrics=("auc", "balanced_accuracy", "specificity"))
                        .assign(feature_config=cfg))
    gap = pd.concat(gaps, ignore_index=True)
    gap.to_csv(out / "e4_gap.csv", index=False)
    table = contrasts(e1, e2, config)
    table.to_csv(out / "e4_contrasts.csv", index=False)

    auc_gap = gap[gap.metric == "auc"]
    norm = auc_gap[auc_gap.feature_config == "norm"]
    norm = norm.assign(_o=norm.family_set.map({s: i for i, s in enumerate(ABLATION_SETS)})).sort_values("_o")
    figures.plot_gap_forest(norm, out / "e4_gap_auc.png",
                            title="Cross-vendor AUC gap by feature family (normalized texture)")
    texture = auc_gap[auc_gap.family_set.isin(TEXTURE_SETS)].copy()
    texture["family_set"] = texture["family_set"] + " | " + texture["feature_config"]
    labels = {f"{s} | {c}": f"{figures.FAMILY_SET_NAMES[s]}, {'raw' if c == 'raw' else 'normalized'} texture"
              for s in TEXTURE_SETS for c in CONFIGS}
    figures.plot_gap_forest(texture.sort_values("family_set"), out / "e4_gap_auc_raw_vs_norm.png", labels=labels,
                            title="Cross-vendor AUC gap: raw vs normalized texture")

    lines = ["# E4: feature-family ablation", "",
             "Δ = AUC within the test vendor (E1 nested CV) − AUC trained on the other vendor (E2), on the same "
             f"test subjects. Contrasts are paired {config['bootstrap_resamples']}-resample bootstraps of a "
             "difference of gaps; Holm within each contrast type and direction.", "",
             "## E1 pooled AUC by family set", ""]
    pooled = aucs[aucs.cohort == "pooled"].assign(
        cell=lambda t: [f"{e:.3f} [{lo:.3f}, {hi:.3f}]" for e, lo, hi in zip(t.estimate, t.ci_low, t.ci_high)])
    lines += [md_table(pooled.pivot_table(index=["feature_config", "family_set"], columns="model", values="cell",
                                          aggfunc="first")[[m for m in figures.MODEL_ORDER
                                                            if m in set(pooled.model)]]), ""]
    show = auc_gap.assign(cell=[f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                                for d, lo, hi in zip(auc_gap.difference, auc_gap.ci_low, auc_gap.ci_high)])
    lines += ["## ΔAUC by direction, family set and model", "",
              md_table(show.pivot_table(index=["direction", "feature_config", "family_set"], columns="model",
                                        values="cell", aggfunc="first")[[m for m in figures.MODEL_ORDER
                                                                         if m in set(show.model)]]), ""]
    if len(table):
        cells = table.assign(diff=table["difference"].map("{:+.3f}".format),
                             ci=[f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(table.ci_low, table.ci_high)],
                             p=table["p_value"].map(lambda v: format_p(v, config)),
                             p_holm=table["p_holm"].map(lambda v: format_p(v, config)))
        keys = ["contrast", "direction", "a", "b", "model"]
        lines += ["## Contrasts of the gap (ΔΔAUC = Δ_a − Δ_b; positive = a transfers worse)", "",
                  md_table(cells[keys + ["diff", "ci", "p", "p_holm"]].set_index(keys)), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return "\n".join(lines)
