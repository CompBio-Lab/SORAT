"""Effect of the optional LV-to-MYO post-processing on Dice.

Compares each model's raw output with its post-processed version ('<model>_pp')
on the same held-out test subjects. Reference contours in ACDC and both M&Ms
datasets include papillary muscles and trabeculae in the LV cavity, so the
relabelling is expected to lower agreement with them; the table quantifies by
how much.

Outputs: tables/postprocess_effect.csv, tables/postprocess_effect.md
Run: python analysis/postprocess_effect.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C


def main() -> None:
    rows = []
    for ds in C.DATASETS:
        df = C.primary_subset(C.load_metrics(ds, include_postprocessed=True))
        df = df[df["model_name"].notna()]
        for model in C.MODEL_ORDER:
            sub = df[df["model_name"] == model]
            for key, name in C.STRUCTURES:
                per = sub.groupby(["patient_id", "postprocessed"])[f"dice_{key}"].mean().unstack()
                per = per.dropna()
                diff = per[True] - per[False]
                lo, hi = C.bootstrap_ci(diff)
                p = stats.wilcoxon(per[True], per[False]).pvalue if np.any(diff != 0) else 1.0
                rows.append({"dataset": ds, "model_name": model, "structure": name, "n_patients": len(per),
                             "dice_raw": per[False].mean(), "dice_pp": per[True].mean(),
                             "mean_change": diff.mean(), "ci_lo": lo, "ci_hi": hi, "p": p})
    out = pd.DataFrame(rows)
    myo_lv = out[out["structure"] != "RV"].copy()
    myo_lv["p_holm"] = C.holm(myo_lv["p"])
    out = out.merge(myo_lv[["dataset", "model_name", "structure", "p_holm"]], how="left",
                    on=["dataset", "model_name", "structure"])
    C.TABLES.mkdir(exist_ok=True)
    out.to_csv(C.TABLES / "postprocess_effect.csv", index=False)

    lines = ["| Dataset | Model | Structure | Dice raw | Dice post-processed | Change [95% CI] |", "|---|---|---|---|---|---|"]
    for _, r in out[out["structure"] != "RV"].iterrows():
        lines.append(f"| {C.DATASETS[r['dataset']]['label']} | {r['model_name']} | {r['structure']} | {r['dice_raw']:.3f} | "
                     f"{r['dice_pp']:.3f} | {r['mean_change']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
    lines.append("")
    lines.append("RV is unchanged by construction (max |change| = "
                 f"{out.loc[out['structure'] == 'RV', 'mean_change'].abs().max():.4f}). "
                 "Paired Wilcoxon signed-rank, Holm-adjusted across the MYO and LV rows: "
                 f"{(myo_lv['p_holm'] < 0.05).sum()} of {len(myo_lv)} significant at 0.05.")
    (C.TABLES / "postprocess_effect.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
