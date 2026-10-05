#!/usr/bin/env bash
# Re-extract nnFormer features for the study cohort with every named radiomics
# config in hcm_vendor/configs/study.yaml. Runs the Nextflow FEATURES_ONLY entry
# once per dataset x config, sequentially (Nextflow runs sharing a launch dir lock).
#
# Run from the repo root on a login node (Nextflow submits SLURM jobs):
#   nohup hcm_vendor/scripts/run_feature_extraction.sh > results_hcm_vendor/logs/t14_extraction.log 2>&1 &
# Optional: DATASETS="mms2" CONFIGS="norm" to run a subset.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source hcm_vendor/scripts/env.sh

DATASETS="${DATASETS:-mms2 acdc}"
CONFIGS="${CONFIGS:-norm raw}"
FEATURE_VENV=/scratch/st-zlaksman-1/pmoheban/venvs/sorat-features-conda
OUT_ROOT="${PWD}/results_hcm_vendor/features/nnformer"
RUN_ROOT="${PWD}/results_hcm_vendor/pipeline_runs"

declare -A SORAT_RESULTS=([mms2]=results/MMS_2 [acdc]=results/ACDC)

for dataset in ${DATASETS}; do
    for cfg in ${CONFIGS}; do
        flags="$(hcmv_python -m hcmv --set feature_config="${cfg}" radiomics-flags 2>/dev/null | grep -v '^INFO' | tail -1)"
        echo "=== ${dataset} / ${cfg}: ${flags:-<PyRadiomics defaults>} ==="
        # shellcheck disable=SC2086  # flags are intentionally word-split
        nextflow run main.nf -entry FEATURES_ONLY -profile slurm \
            --slurm_account "${HCMV_SLURM_ACCOUNT}" \
            --models nnformer --feature_extraction.model_tag nnformer__fold0 \
            --input "${PWD}/results_hcm_vendor/inputs/${dataset}_nor_hcm.csv" \
            --feature_extraction.enabled true --feature_extraction.mask_source predictions \
            --feature_extraction.results_dir "${PWD}/${SORAT_RESULTS[$dataset]}" \
            --feature_extraction.output_dir "${OUT_ROOT}/${dataset}/${cfg}" \
            --feature_extraction.virtualenv_path "${FEATURE_VENV}" \
            --feature_extraction.require_virtualenv true \
            ${flags} \
            --outdir "${RUN_ROOT}/${dataset}_${cfg}"
    done
done
echo "=== all feature extraction runs finished ==="
