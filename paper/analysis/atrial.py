"""Atrial segmentation accuracy (atrial nnU-Net).

Datasets:
  2018 Atrial Segmentation Challenge (training 100 + testing 54 scans), LA
  cavity reference, scored with --evaluation.label_schema atrial_la_cavity;
  scans also in the model's MBAS training data (in_mbas_training = 1 in the
  samplesheets, see data_prep/mbas_overlap.csv) are reported separately.
  AIM-ASD (16), binary biatrial reference, atrial_binary_union.

Outputs: tables/atrial_accuracy.{csv,md}
Run: python analysis/atrial.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

SORAT_DATA = C.SAMPLESHEETS
RUNS = {
    "LASC2018_test": ("LASC2018_test", SORAT_DATA / "lasc2018_test_samplesheet.csv", "la"),
    "LASC2018_train": ("LASC2018_train", SORAT_DATA / "lasc2018_train_samplesheet.csv", "la"),
    "AIM_ASD": ("AIM_ASD", SORAT_DATA / "aim_asd_samplesheet.csv", "biatrial"),
}


def load(run: str) -> pd.DataFrame:
    folder, sheet, structure = RUNS[run]
    m = pd.read_csv(C.RESULTS / folder / "comparison" / "aggregated_metrics.csv", dtype={"patient_id": str})
    s = pd.read_csv(sheet, dtype={"patient_id": str})
    if "in_mbas_training" not in s:
        s["in_mbas_training"] = 0
    m = m.merge(s[["patient_id", "in_mbas_training"]], on="patient_id", how="left", validate="many_to_one")
    if m["in_mbas_training"].isna().any():
        raise ValueError(f"{run}: subjects missing from the samplesheet")
    m["dice"] = m[f"dice_{structure}"]
    m["hd95"] = pd.to_numeric(m[f"hd95_{structure}"], errors="coerce")
    m["run"] = run
    m["reference"] = "LA cavity" if structure == "la" else "Biatrial"
    return m


def summary(df: pd.DataFrame, label: str) -> dict:
    per = df.groupby("patient_id")[["dice", "hd95"]].mean()
    lo, hi = C.bootstrap_ci(per["dice"])
    return {"subset": label, "reference": df["reference"].iloc[0], "n": len(per),
            "dice_mean": per["dice"].mean(), "dice_sd": per["dice"].std(), "dice_ci_lo": lo, "dice_ci_hi": hi,
            "hd95_median": per["hd95"].median(), "hd95_q1": per["hd95"].quantile(0.25),
            "hd95_q3": per["hd95"].quantile(0.75),
            "failures": int(((per["dice"] < 0.5) | (per["hd95"] > 20) | ~np.isfinite(per["hd95"])).sum())}


def main() -> None:
    lasc = pd.concat([load("LASC2018_test"), load("LASC2018_train")], ignore_index=True)
    aim = load("AIM_ASD")
    rows = [summary(lasc[lasc["in_mbas_training"] == 0], "2018 challenge, not in model training"),
            summary(lasc[lasc["in_mbas_training"] == 1], "2018 challenge, also in model training"),
            summary(aim, "AIM-ASD (external)")]
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "atrial_accuracy.csv", index=False)
    lines = ["| Subset | Reference | n | DSC | HD95 (mm) | Failures |", "|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r['subset']} | {r['reference']} | {r['n']} | {r['dice_mean']:.3f} ± {r['dice_sd']:.3f} "
                     f"[{r['dice_ci_lo']:.3f}, {r['dice_ci_hi']:.3f}] | {r['hd95_median']:.1f} "
                     f"[{r['hd95_q1']:.1f}–{r['hd95_q3']:.1f}] | {r['failures']} |")
    lines += ["", "DSC mean ± SD [95% bootstrap CI]; HD95 median [IQR]. Failure: DSC < 0.5, HD95 > 20 mm or missing "
              "structure. 2018 challenge scans that are also in the MBAS training data are shown separately."]
    (C.TABLES / "atrial_accuracy.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
