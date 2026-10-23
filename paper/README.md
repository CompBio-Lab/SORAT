# Reproducing the SORAT paper

This folder holds everything needed to regenerate the tables and figures of the SORAT manuscript. That means data preparation, the SORAT runs, the standalone model runs used for comparison, and the analysis scripts. The analysis reads SORAT's output folders and the public datasets; it never modifies them.

## Settings

Every script reads its locations from environment variables:

| Variable | Points to |
|---|---|
| `SORAT_DATA_ROOT` | Folder holding the datasets: `ACDC/`, `M&Ms/`, `M&Ms_2/`, `Atria_2018/`, `AIM-ASD/` |
| `SORAT_RESULTS` | SORAT output folders, one per dataset: `ACDC/`, `MMS/`, `MMS_2/`, `LASC2018_test/`, `LASC2018_train/`, `AIM_ASD/` |
| `SORAT_RUNS` | Standalone and repeat runs (default: `$SORAT_RESULTS/../paper_runs`) |
| `SORAT_SAMPLESHEETS` | Samplesheets and generated M&Ms-1 `Info.cfg` files (default: `<repo>/data`) |
| `MBAS_TRAINING_ROOT` | MBAS 2024 training set, only for the overlap check |
| `PAPER_SIF` | Apptainer image with Python 3.11, NumPy, pandas, SciPy, matplotlib, nibabel, SimpleITK and MedPy, for the analysis |
| `SORAT_BIND` | Paths to bind into containers (default: data and results roots) |

## 1. Prepare the data

Each step writes into `$SORAT_SAMPLESHEETS`. The scripts were checked to reproduce the samplesheets used for the paper.

| Dataset | Step | Command |
|---|---|---|
| ACDC | Samplesheet of the 50 test subjects | `python bin/generate_acdc_samplesheet.py --acdc_dir $SORAT_DATA_ROOT/ACDC/database --dataset testing --output $SORAT_SAMPLESHEETS/acdc_testing_samplesheet.csv` |
| M&Ms-1 | Samplesheet of the 136 test subjects, then `Info.cfg` files (ED/ES counted from 1) from the dataset information file, checked against the annotated frames | `python bin/generate_samplesheet.py --data_root "$SORAT_DATA_ROOT/M&Ms" --data_split Testing --output $SORAT_SAMPLESHEETS/mms_testing_samplesheet.csv`, then `paper/data_prep/generate_mms1_info_cfg.py` (usage in the file) |
| M&Ms-2 | Reorganize into `MnM2/{SA,LA}/NNN/`, write `Info.cfg` from the ED/ES phase images, write `mms2_sa_samplesheet.csv` (all 360 subjects) | `paper/data_prep/reorganize_mms2.py`, `verify_regen_mms2_cfg.py`, `generate_mms2_samplesheets.py` |
| 2018 Atrial Segmentation Challenge | Samplesheets with spacing 0.625 × 0.625 × 1.25 mm and the LA-cavity reference | `python paper/data_prep/make_lasc2018_samplesheets.py --root $SORAT_DATA_ROOT/Atria_2018 --out $SORAT_SAMPLESHEETS` |
| AIM-ASD | Samplesheet of the LGE images and biatrial references | `python paper/data_prep/make_aim_asd_samplesheet.py --root $SORAT_DATA_ROOT/AIM-ASD/dicoms/AIM-ASD/ASD --out $SORAT_SAMPLESHEETS/aim_asd_samplesheet.csv` |

`paper/data_prep/check_mbas_overlap.py` finds the 2018 and AIM-ASD scans that are also in the atrial model's MBAS training data; its result is `paper/data_prep/mbas_overlap.csv`, which the 2018 samplesheets use.

## 2. Run SORAT

Run from the repository root. These are the settings used in the paper:

```bash
COMMON="-profile slurm --slurm_account <account> --models all \
  --postprocess.enabled true --postprocess.use_for_metrics true \
  --feature_extraction.enabled true --feature_extraction.mask_source predictions"

nextflow run main.nf $COMMON --input $SORAT_SAMPLESHEETS/acdc_testing_samplesheet.csv \
  --evaluation.gt_label_map 'rv=1,myo=2,lv=3' --outdir $SORAT_RESULTS/ACDC
nextflow run main.nf $COMMON --input $SORAT_SAMPLESHEETS/mms_testing_samplesheet.csv \
  --evaluation.gt_label_map 'lv=1,myo=2,rv=3' --outdir $SORAT_RESULTS/MMS
nextflow run main.nf $COMMON --input $SORAT_SAMPLESHEETS/mms2_sa_samplesheet.csv \
  --evaluation.gt_label_map 'lv=1,myo=2,rv=3' --outdir $SORAT_RESULTS/MMS_2

# second, independent ACDC run from a cold cache (run-to-run reproducibility)
nextflow run main.nf -profile slurm --slurm_account <account> --models all --preprocess_cache_enabled false \
  --postprocess.enabled true --postprocess.use_for_metrics true \
  --input $SORAT_SAMPLESHEETS/acdc_testing_samplesheet.csv --evaluation.gt_label_map 'rv=1,myo=2,lv=3' \
  --outdir $SORAT_RUNS/sorat_repeat/ACDC -w $SORAT_RUNS/sorat_repeat/ACDC/work

# atrial model
nextflow run main.nf -profile slurm --slurm_account <account> --models atrial_nnunet \
  --evaluation.label_schema atrial_la_cavity --input $SORAT_SAMPLESHEETS/lasc2018_test_samplesheet.csv \
  --outdir $SORAT_RESULTS/LASC2018_test        # likewise lasc2018_train_samplesheet.csv -> LASC2018_train
nextflow run main.nf -profile slurm --slurm_account <account> --models atrial_nnunet \
  --evaluation.label_schema atrial_binary_union --input $SORAT_SAMPLESHEETS/aim_asd_samplesheet.csv \
  --outdir $SORAT_RESULTS/AIM_ASD
```

## 3. Standalone runs

Each model is also run outside SORAT on the ACDC test set, in the same container SORAT uses. The notebooks in `standalone/notebooks/` follow the preprocessing and inference code in each model's public repository. Set `CINEMA_ROOT`, `NNFORMER_ROOT` and `VSA3L_ROOT` to the model code and weights.

```bash
cd paper
for m in nnformer vsa3l; do sbatch --account=<gpu-account> jobs/standalone.sbatch $m 1; done
sbatch --account=<gpu-account> jobs/cinema_centre_crop.sbatch    # CineMA, centre crop instead of the ground-truth crop
```

## 4. Analysis

```bash
cd paper
sbatch --account=<cpu-account> jobs/analysis_cpu.sbatch analysis/slice_metrics.py   # slow: reads every mask
for s in seg_accuracy postprocess_effect clinical determinism standalone_agreement failures \
         failure_examples atrial case_study supplement paper_tables; do
  ./run_analysis.sh analysis/$s.py
done
./run_analysis.sh analysis/make_figure1.py figures
```

Outputs go to `paper/tables/` and `paper/figures/`.

All analyses use the conventions in `analysis/common.py`. Only held-out test subjects are included; for M&Ms-2 that is the official test split, subjects 201–360. Each checkpoint is represented by one variant (the three CineMA ensembles, nnFormer and VSA-3L), and single CineMA seeds are reported in the supplement. The patient is the unit of analysis, with ED and ES averaged per patient. Confidence intervals come from a patient-level bootstrap (2,000 resamples, seed 0), and models are compared with a Friedman test followed by Wilcoxon signed-rank tests with Holm correction.

## Where each table and figure comes from

| Manuscript | Output | Script |
|---|---|---|
| Table 1 | `tables/table1_datasets.md` | `paper_tables.py` |
| Table 2 | `tables/table4_reproducibility.md` (from `standalone_agreement_summary.csv`, `determinism.csv`) | `standalone_agreement.py`, `determinism.py`, `paper_tables.py` |
| Table 3 | Sources of silent error (written in the manuscript) | n/a |
| Table 4 | `tables/seg_accuracy_main.md` | `seg_accuracy.py` |
| Table 5 | `tables/atrial_accuracy.md` | `atrial.py` |
| Table 6 | `tables/clinical_agreement.md` | `clinical.py` |
| Figure 1 | `figures/figure1_workflow.{svg,pdf,png}` (font Carlito, SIL OFL, in `fonts/`) | `make_figure1.py` |
| Figure 2 | `figures/fig_accuracy_by_dataset.*` | `seg_accuracy.py` |
| Figure 3 | `figures/fig_vendor_heatmap.*` | `seg_accuracy.py` |
| Figure 4 | `figures/fig_dice_by_position.*` | `failures.py` (after `slice_metrics.py`) |
| Figure 5 | `figures/fig_bland_altman_ef.*` | `clinical.py` |
| Figure 6 | `figures/fig_case_study.*` (inputs in `data/case_study/`) | `case_study.py` |
| Tables S1–S6, Figure S1 | `tables/supplement/`, `figures/figS1_postprocess_example.*` | `supplement.py` |
| Figure S2 | `figures/fig_failure_examples.*` | `failure_examples.py` |
