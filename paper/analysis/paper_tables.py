"""Tables 1 and 2 of the manuscript.

Table 1: test-set composition (subjects by vendor and pathology) and which
checkpoint was trained on which dataset.
Table 2: reproducibility, SORAT vs each model's standalone pipeline and
between two independent SORAT runs.

Outputs: tables/table1_datasets.md, tables/table4_reproducibility.md
Run: python analysis/paper_tables.py   (after standalone_agreement.py and determinism.py)
"""

from __future__ import annotations

import pandas as pd

import common as C


def table1() -> str:
    lines = ["| Dataset | Test subjects | Vendors (n) / imaging | Pathologies (n) |", "|---|---|---|---|"]
    for ds in C.DATASETS:
        m = C.primary_subset(C.load_metrics(ds))
        subj = m[["patient_id", "vendor", "pathology"]].drop_duplicates()
        vend = ", ".join(f"{k} {v}" for k, v in subj["vendor"].value_counts().items())
        path = ", ".join(f"{k} {v}" for k, v in subj["pathology"].value_counts().items())
        lines.append(f"| {C.DATASETS[ds]['label']} | {len(subj)} | {vend} | {path} |")
    lines.append("| 2018 Atrial Segmentation Challenge | 154 (87 not in MBAS training) | LGE, University of Utah | Atrial fibrillation (LA cavity reference) |")
    lines.append("| AIM-ASD | 16 | LGE and MR angiography, 1.5 T | Atrial septal defect (biatrial reference) |")
    lines += ["", "| Model | Training data | In-distribution test set |", "|---|---|---|"]
    train = {"CineMA-ACDC": "ACDC training split (100)", "CineMA-M&Ms": "M&Ms-1 training split",
             "CineMA-M&Ms-2": "M&Ms-2 training split", "nnFormer": "ACDC training split (100)",
             "VSA-3L": "Non-public (MONAI model zoo)"}
    extra = [("Atrial nnU-Net", "MBAS 2024 training set (70 LGE scans)", "2018 challenge scans also in MBAS (67)")]
    for model in C.MODEL_ORDER:
        ind = ", ".join(C.DATASETS[d]["label"] for d in C.TRAINED_ON[model]) or "None"
        lines.append(f"| {model} | {train[model]} | {ind} |")
    for model, data, ind in extra:
        lines.append(f"| {model} | {data} | {ind} |")
    lines.append("")
    lines.append("CineMA checkpoints: three seeds per training dataset, combined by majority vote. "
                 "M&Ms-2 test set: official subjects 201–360.")
    return "\n".join(lines)


def table4() -> str:
    sa = pd.read_csv(C.TABLES / "standalone_agreement_summary.csv")
    det = pd.read_csv(C.TABLES / "determinism.csv")
    lines = ["| Comparison | Model | Masks | Identical | Minimum voxel agreement | Largest DSC difference |",
             "|---|---|---|---|---|---|"]
    full = pd.read_csv(C.TABLES / "standalone_agreement.csv")
    for _, r in sa.iterrows():
        d = full[full["model_name"] == r["model_name"]]
        diff = max((d[f"dice_sorat_vs_gt_{s}"] - d[f"dice_standalone_vs_gt_{s}"]).abs().max() for s in ("RV", "MYO", "LV"))
        label = "CineMA (9 checkpoints)" if r["model_name"] == "CineMA" else r["model_name"]
        lines.append(f"| SORAT vs standalone notebook | {label} | {int(r['n_frames'])} | {int(r['identical_frames'])} | "
                     f"{100 * r['min_voxel_agreement']:.4f}% | {diff:.4f} |")
    names = {"cinema": "CineMA (12 variants)", "nnformer": "nnFormer", "vsa3l": "VSA-3L"}
    for _, r in det.iterrows():
        lines.append(f"| Two independent SORAT runs | {names[r['architecture']]} | {int(r['masks'])} | {int(r['identical'])} | "
                     f"{100 * r['min_voxel_agreement']:.4f}% | {r['max_abs_dice_difference']:.4f} |")
    lines.append("")
    lines.append("ACDC test set, ED and ES frames. DSC difference: largest absolute difference in DSC against the "
                 "reference contours for any structure and frame. CineMA's standalone run uses its notebook with a "
                 "centre crop in place of the ground-truth LV crop (standalone/cinema_centre_crop.py).")
    return "\n".join(lines)


def main() -> None:
    (C.TABLES / "table1_datasets.md").write_text(table1())
    (C.TABLES / "table4_reproducibility.md").write_text(table4())
    print(table1())
    print()
    print(table4())


if __name__ == "__main__":
    main()
