#!/bin/bash
#SBATCH --job-name=hcmv-run
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# Run an experiment into the result store. Arguments go to `python -m hcmv run-experiment`.
# Submit from the repo root, e.g.:
#   sbatch --account=st-zlaksman-1 --output=results_hcm_vendor/logs/%x-%j.out \
#       hcm_vendor/scripts/run_experiment.sbatch --experiment E1
#   sbatch ... --cpus-per-task=4 --time=01:00:00 hcm_vendor/scripts/run_experiment.sbatch --experiment E1 --smoke
# Finished runs are skipped, so a timed-out job can be resubmitted unchanged.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/hcm_vendor/scripts/env.sh"
# One BLAS/OpenMP thread per worker; parallelism comes from joblib over outer folds.
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
echo "job ${SLURM_JOB_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} cpus, args: $*"
hcmv_python -m hcmv run-experiment "$@"
