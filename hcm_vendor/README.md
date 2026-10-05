# HCM vs normal across MRI vendors (EECE 568)

This folder holds the analysis for the EECE 568 project: can a classifier that separates
hypertrophic cardiomyopathy (HCM) from normal hearts, built on features from SORAT's
nnFormer segmentations, keep working when the MRI vendor changes?

- `hcmv/`: the analysis package (`python -m hcmv <command>`).
- `configs/study.yaml`: paths, cohorts, seeds, model grids and experiment definitions.
- `scripts/`: environment wrapper, SLURM job scripts and the one-command pipeline.
- `tests/`: unit tests on synthetic data (`pytest`).
- `DASHBOARD.md`: plan, decisions and results log. `FIGURES.md`: every figure and table,
  with the command and inputs that made it.

All outputs go to `results_hcm_vendor/` at the repo root (gitignored).

## 1. Environment

The analysis runs in an existing Apptainer image, not a virtual environment:

```bash
source hcm_vendor/scripts/env.sh        # loads apptainer, defines hcmv_python
hcmv_python -m hcmv --help
hcmv_python -m pytest hcm_vendor/tests -q -p no:cacheprovider
```

`env.sh` runs Python in `jupyter-datascience.sif` with the container home
`/scratch/st-zlaksman-1/pmoheban/my_jupyter`; override with `HCMV_SIF` and
`HCMV_CONTAINER_HOME`. The image provides numpy, pandas, scikit-learn, torch, matplotlib
and SimpleITK; xgboost 3.0.5 and shap 0.49.1 were added with `pip install --user` under a
constraints file so the image's core packages are not upgraded. Exact versions are in
`requirements.txt` (documentation only). `scripts/smoke_env.sbatch` checks the setup on
a compute node. The SLURM account defaults to `st-zlaksman-1` (`HCMV_SLURM_ACCOUNT`).

## 2. Cohort and features

```bash
hcmv_python -m hcmv cohort               # cohort table: M&Ms-2 NOR/HCM (135) + ACDC test NOR/HCM (20)
hcmv_python -m hcmv make-samplesheets    # SORAT samplesheets for the cohort
hcm_vendor/scripts/run_feature_extraction.sh   # Nextflow FEATURES_ONLY, per dataset x radiomics config
hcmv_python -m hcmv check-features       # acceptance checks on the extracted features
hcmv_python -m hcmv feature-tables       # one row per subject: 17 clinical, 28 shape, 84 texture features
hcmv_python -m hcmv qc-report            # data QC and exploratory report
```

Feature extraction uses the SORAT pipeline's corrected wall thickness and two radiomics
configs: `norm` (primary: intensity normalization, 32 bins, 2D, 1.25 mm pixels) and `raw`
(PyRadiomics defaults). It needs the existing nnFormer segmentations under
`results/{MMS_2,ACDC}` and the PyRadiomics environment named in the script. Ground-truth
mask features for validation come from `scripts/extract_gt_features.sbatch` and
`hcmv validate-gt`.

## 3. Experiments, reports, figures

One command submits everything as a chain of SLURM jobs (about an hour of wall time,
most of it E1 and the E3 permutation test):

```bash
hcm_vendor/scripts/run_all.sh                       # full study
SMOKE=1 hcm_vendor/scripts/run_all.sh               # tiny grids into runs-smoke/, minutes
RUNS_DIR=runs-rerun hcm_vendor/scripts/run_all.sh   # full study into a separate result store
```

| Step | Command | Output (`results_hcm_vendor/…`) |
|---|---|---|
| E1 nested CV, pooled and within vendor | `run-experiment --experiment E1` | `runs/E1/` |
| E2 cross-vendor transfer | `run-experiment --experiment E2` | `runs/E2/` |
| E3 vendor probe on normal hearts | `e3-probe` | `runs/E3/` |
| E4 feature-family ablation | E1/E2 with more `--family-sets`, raw texture via `--set experiments.feature_config=raw`, then `e4-report` | `runs/E4/analysis/` |
| E5 external ACDC test, GE check | `run-experiment --experiment E5` / `GE`, then `e5-report` | `runs/{E5,GE}/` |
| Reports | `e1-report`, `e2-report`, `e4-report`, `e5-report` | `runs/*/analysis/` |
| SHAP stability | `shap` | `runs/SHAP/analysis/` |
| Texture check | `texture-check` | `qc/texture_check/` |
| Summary and verdicts | `summary` | `tables/summary.md` |
| Final tables and figures | `tables`, then `figures` | `tables/final/`, `figures/` |

Runs are stored under `runs/<experiment>/<cohort or direction>/<family set>/<model>/<feature config>/`
with predictions, chosen hyperparameters, metrics with bootstrap CIs and a manifest (git
commit, config hash, package versions). A run whose manifest matches the current
configuration is skipped, so jobs can be resubmitted after a timeout; `--force` reruns.
Any config value can be overridden with `--set key=value`.

Feature extraction runs through the SORAT Nextflow pipeline; everything after it is
CPU-only Python in the container.
