#!/bin/bash
# CineMA standalone with a centre crop (no ground truth at inference), all nine
# checkpoints, masks in the original geometry. Usage (from paper/):
#   sbatch --account=<gpu-account> jobs/cinema_centre_crop.sbatch
#SBATCH --job-name=cinema_centre_crop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
: "${SORAT_RUNS:?set SORAT_RUNS}" "${SORAT_DATA_ROOT:?set SORAT_DATA_ROOT}"
CONTAINERS="${SORAT_CONTAINERS:-$SLURM_SUBMIT_DIR/../containers}"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv
apptainer exec --nv --cleanenv --env PYTHONNOUSERSITE=1 --env SORAT_DATA_ROOT="$SORAT_DATA_ROOT" \
  --bind "${SORAT_BIND:-$SORAT_DATA_ROOT,$SORAT_RUNS}" "$CONTAINERS/sorat-cinema.sif" \
  python -u standalone/cinema_centre_crop.py --run_dir "$SORAT_RUNS/standalone/cinema_centre_crop/rep1"
