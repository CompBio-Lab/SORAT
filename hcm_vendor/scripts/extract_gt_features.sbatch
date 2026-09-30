#!/bin/bash
#SBATCH --job-name=hcmv-gt-features
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# T12: ground-truth-mask features for the study cohort (all feature_configs).
# Submit from the repo root:
#   sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out hcm_vendor/scripts/extract_gt_features.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load gcc apptainer >/dev/null 2>&1

FEATURE_VENV_SITE=/scratch/st-zlaksman-1/pmoheban/venvs/sorat-features-conda/lib/python3.10/site-packages

apptainer exec --bind /scratch,/arc --env PYTHONPATH="${FEATURE_VENV_SITE}" \
    containers/sorat-cinema.sif \
    python hcm_vendor/scripts/extract_gt_features.py --workers "${SLURM_CPUS_PER_TASK}" "$@" \
    2>&1 | grep -vE "unexpected scales in sform|^WARNING: In .*itkNiftiImageIO|^$|GLCM is symmetrical|Fixed bin Count"
