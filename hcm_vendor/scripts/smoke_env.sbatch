#!/bin/bash
#SBATCH --job-name=hcmv-smoke
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
# Submit from the repo root:
#   sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out hcm_vendor/scripts/smoke_env.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/hcm_vendor/scripts/env.sh"
hcmv_python "${HCMV_REPO_ROOT}/hcm_vendor/scripts/smoke_env.py"
