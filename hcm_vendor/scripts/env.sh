#!/usr/bin/env bash
# Source this file to get `hcmv_python`, which runs Python inside the analysis
# container (the same image and home as the user's `launch-apptainer` alias).
# Extra packages (xgboost, shap) live in the container home's ~/.local.
#
#   source hcm_vendor/scripts/env.sh
#   hcmv_python -m pytest hcm_vendor/tests -q

HCMV_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HCMV_SIF="${HCMV_SIF:-/arc/project/st-singha53-1/pmoheban/jupyter/jupyter-datascience.sif}"
HCMV_CONTAINER_HOME="${HCMV_CONTAINER_HOME:-/scratch/st-zlaksman-1/pmoheban/my_jupyter}"
HCMV_SLURM_ACCOUNT="${HCMV_SLURM_ACCOUNT:-${SORAT_SLURM_ACCOUNT:-st-zlaksman-1}}"
export HCMV_REPO_ROOT HCMV_SIF HCMV_CONTAINER_HOME HCMV_SLURM_ACCOUNT

module load gcc apptainer >/dev/null 2>&1

hcmv_python() {
    apptainer exec \
        --home "${HCMV_CONTAINER_HOME}" \
        --env XDG_CACHE_HOME="${HCMV_CONTAINER_HOME}" \
        --env PYTHONPATH="${HCMV_REPO_ROOT}/hcm_vendor${PYTHONPATH:+:${PYTHONPATH}}" \
        --bind /scratch,/arc \
        --pwd "${HCMV_REPO_ROOT}" \
        "${HCMV_SIF}" python "$@"
}
