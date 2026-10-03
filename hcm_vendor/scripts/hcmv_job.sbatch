#!/bin/bash
#SBATCH --job-name=hcmv-job
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
# Run any `python -m hcmv <command> ...` on a compute node; arguments are passed through.
# Submit from the repo root, e.g.:
#   sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out --job-name=hcmv-e3 \
#       hcm_vendor/scripts/hcmv_job.sbatch e3-probe

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/hcm_vendor/scripts/env.sh"
export APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
echo "job ${SLURM_JOB_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} cpus, args: $*"
hcmv_python -m hcmv "$@"
