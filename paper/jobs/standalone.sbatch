#!/bin/bash
# One run of a model's standalone notebook on the ACDC test set, in the same
# container SORAT uses. Usage (from paper/):
#   sbatch --account=<gpu-account> jobs/standalone.sbatch <nnformer|cinema|vsa3l> <repeat>
#SBATCH --job-name=sorat_standalone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
MODEL="$1"; REP="$2"
: "${SORAT_RUNS:?set SORAT_RUNS}" "${SORAT_DATA_ROOT:?set SORAT_DATA_ROOT}"
CONTAINERS="${SORAT_CONTAINERS:-$SLURM_SUBMIT_DIR/../containers}"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv
apptainer exec --nv --cleanenv --env PYTHONNOUSERSITE=1 --env MPLCONFIGDIR=/tmp \
  --env SORAT_DATA_ROOT="$SORAT_DATA_ROOT" --env ACDC_ROOT="$SORAT_DATA_ROOT/ACDC/database" \
  --env CINEMA_ROOT="${CINEMA_ROOT:-}" --env NNFORMER_ROOT="${NNFORMER_ROOT:-}" --env VSA3L_ROOT="${VSA3L_ROOT:-}" \
  --bind "${SORAT_BIND:-$SORAT_DATA_ROOT,$SORAT_RUNS}" "$CONTAINERS/sorat-$MODEL.sif" \
  python -u standalone/run_notebook.py --model "$MODEL" --run_dir "$SORAT_RUNS/standalone/$MODEL/rep$REP"
