"""Agreement between SORAT and each model run standalone.

Compares SORAT's raw masks with masks from the model's own standalone
inference on the same ACDC test frames: voxel agreement and per-structure
Dice between the two masks, plus each against the ground truth. Identical
outputs mean SORAT adds orchestration without changing model behaviour.

Standalone masks come from repeat 1 of the benchmark runs of each model's
original notebook (benchmark/standalone.sbatch). nnFormer: predict_from_folder
on the ACDC frame files (fold 0, model_best, TTA on). VSA-3L: the notebook's
2-D inference, stacked back to 3-D. CineMA: the notebook's inference with
CineMA's official preprocessing, a centre crop instead of the ground-truth crop,
Info.cfg frames and masks resampled to the original grid
(standalone/cinema_centre_crop.py, benchmark/cinema_centre_crop.sbatch), for
all nine checkpoints, each compared with SORAT's mask from the same checkpoint.

Outputs: tables/standalone_agreement.csv (per frame), tables/standalone_agreement_summary.csv
Run: python analysis/standalone_agreement.py
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd

import common as C

RUNS = C.RUNS / "standalone"
STANDALONE = {
    # Benchmark repeat 1 of each notebook (benchmark/standalone.sbatch), in the SORAT model containers.
    "nnFormer": (RUNS / "nnformer" / "rep1" / "nnformer_predictions", "{pid}_frame{num:02d}.nii.gz",
                 "nnformer", "nnformer__fold0"),
    # 2-D slice predictions stacked to 3-D by standalone/run_notebook.py (assemble_vsa3l).
    "VSA-3L": (RUNS / "vsa3l" / "rep1" / "volumes", "{pid}_{tag}.nii.gz", "vsa3l", "vsa3l__model"),
}
for _ds in ("acdc", "mnms", "mnms2"):
    for _seed in (0, 1, 2):
        STANDALONE[f"CineMA|{_ds}_seed{_seed}"] = (RUNS / "cinema_centre_crop" / "rep1" / "predictions",
                                                  f"{{pid}}_{{tag}}_{_ds}_seed{_seed}.nii.gz", "cinema", f"cinema__{_ds}_seed{_seed}")


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def main() -> None:
    metrics = C.load_metrics("ACDC")
    frames = metrics[["patient_id", "frame_tag", "frame_idx"]].drop_duplicates()
    rows = []
    for name, (folder, pattern, arch, tag) in STANDALONE.items():
        for _, fr in frames.iterrows():
            pid, ft, idx = fr["patient_id"], fr["frame_tag"], int(fr["frame_idx"])
            sa = np.rint(nib.load(folder / pattern.format(pid=pid, num=idx + 1, tag=ft)).get_fdata()).astype(np.int16)
            so = np.rint(nib.load(C.RESULTS / "ACDC" / arch / "segmentations" / f"{pid}_{ft}_{tag}.nii.gz").get_fdata()).astype(np.int16)
            gt = np.rint(nib.load(C.DATA_ROOT / "ACDC" / "database" / "testing" / pid / f"{pid}_frame{idx + 1:02d}_gt.nii.gz").get_fdata()).astype(np.int16)
            row = {"model_name": name.split("|")[0], "checkpoint": name.split("|")[-1], "patient_id": pid, "frame_tag": ft, "identical": bool(np.array_equal(sa, so)),
                   "voxel_agreement": float((sa == so).mean())}
            for label, s in ((1, "RV"), (2, "MYO"), (3, "LV")):
                row[f"dice_sorat_vs_standalone_{s}"] = dice(so == label, sa == label)
                row[f"dice_sorat_vs_gt_{s}"] = dice(so == label, gt == label)
                row[f"dice_standalone_vs_gt_{s}"] = dice(sa == label, gt == label)
            rows.append(row)
    df = pd.DataFrame(rows)
    C.TABLES.mkdir(exist_ok=True)
    df.to_csv(C.TABLES / "standalone_agreement.csv", index=False)
    summary = df.groupby("model_name").agg(
        n_frames=("identical", "size"), identical_frames=("identical", "sum"),
        min_voxel_agreement=("voxel_agreement", "min"),
        **{c: (c, "mean") for c in df.columns if c.startswith("dice_")})
    summary.to_csv(C.TABLES / "standalone_agreement_summary.csv")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
