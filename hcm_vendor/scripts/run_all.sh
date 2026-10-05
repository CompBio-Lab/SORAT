#!/usr/bin/env bash
# Submit the whole analysis as a chain of SLURM jobs, from the feature tables to the
# final figures. Feature extraction (Nextflow, scripts/run_feature_extraction.sh) and
# `hcmv feature-tables` must have been run first.
#
# Run from the repo root on a login node:
#   hcm_vendor/scripts/run_all.sh                    # full study
#   SMOKE=1 hcm_vendor/scripts/run_all.sh            # tiny grids into runs-smoke/ (minutes)
#   RUNS_DIR=runs-rerun hcm_vendor/scripts/run_all.sh  # full study into a separate result store
# Finished runs are skipped (result store keyed by a config hash), so the chain can be
# resubmitted after a failure. Logs go to results_hcm_vendor/logs/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source hcm_vendor/scripts/env.sh

ACCOUNT="--account=${HCMV_SLURM_ACCOUNT}"
LOG="--output=results_hcm_vendor/logs/%x-%j.out"
RUN=hcm_vendor/scripts/run_experiment.sbatch
JOB=hcm_vendor/scripts/hcmv_job.sbatch
SETS=clinical,clinical+shape,clinical+texture,all,all-no-wt
TEXTURE_SETS=clinical+texture,all
EXTRA=()
[[ -n "${SMOKE:-}" ]] && EXTRA+=(--smoke)
[[ -n "${RUNS_DIR:-}" ]] && EXTRA+=(--set "experiments.runs_dir=${RUNS_DIR}")
REPORT_EXTRA=()
[[ -n "${SMOKE:-}" ]] && REPORT_EXTRA+=(--smoke)
[[ -n "${RUNS_DIR:-}" ]] && REPORT_EXTRA+=(--set "experiments.runs_dir=${RUNS_DIR}")

submit() {  # submit <job-name> <sbatch options...> -- <script> <args...>; prints the job id
    local name="$1"; shift
    sbatch --parsable "${ACCOUNT}" "${LOG}" --job-name="hcmv-${name}" "$@"
}

mkdir -p results_hcm_vendor/logs
e1=$(submit e1 "${RUN}" --experiment E1 --family-sets "${SETS}" "${EXTRA[@]}")
e1raw=$(submit e1-raw "${RUN}" --experiment E1 --family-sets "${TEXTURE_SETS}" \
    --set experiments.feature_config=raw "${EXTRA[@]}")
e2=$(submit e2 --cpus-per-task=16 --mem=32G --time=03:00:00 "${RUN}" --experiment E2 --family-sets "${SETS}" \
    "${EXTRA[@]}")
e2raw=$(submit e2-raw --cpus-per-task=16 --mem=32G --time=02:00:00 "${RUN}" --experiment E2 \
    --family-sets "${TEXTURE_SETS}" --set experiments.feature_config=raw "${EXTRA[@]}")
e5=$(submit e5 --cpus-per-task=16 --mem=32G --time=02:00:00 "${RUN}" --experiment E5 --family-sets "${SETS}" \
    "${EXTRA[@]}")
ge=$(submit ge --cpus-per-task=16 --mem=32G --time=02:00:00 "${RUN}" --experiment GE --family-sets "${SETS}" \
    "${EXTRA[@]}")
e3=$(submit e3 --time=08:00:00 "${JOB}" e3-probe "${REPORT_EXTRA[@]}")

small=(--nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=02:00:00)
r1=$(submit e1-report "${small[@]}" --dependency=afterok:${e1} "${JOB}" e1-report "${REPORT_EXTRA[@]}")
r2=$(submit e2-report "${small[@]}" --dependency=afterok:${e1}:${e2} "${JOB}" e2-report "${REPORT_EXTRA[@]}")
r4=$(submit e4-report "${small[@]}" --dependency=afterok:${e1}:${e1raw}:${e2}:${e2raw} "${JOB}" e4-report \
    "${REPORT_EXTRA[@]}")
r5=$(submit e5-report "${small[@]}" --dependency=afterok:${e5}:${ge} "${JOB}" e5-report "${REPORT_EXTRA[@]}")
sh=$(submit shap --mem=64G --time=06:00:00 --dependency=afterok:${e2} "${JOB}" shap "${REPORT_EXTRA[@]}")
echo "submitted: e1=${e1} e1-raw=${e1raw} e2=${e2} e2-raw=${e2raw} e5=${e5} ge=${ge} e3=${e3}"
echo "reports:   e1=${r1} e2=${r2} e4=${r4} e5/ge=${r5} shap=${sh}"

if [[ -z "${SMOKE:-}" && -z "${RUNS_DIR:-}" ]]; then
    # The summary, final tables and figures read the primary result store only.
    tc=$(submit texture-check "${small[@]}" --dependency=afterok:${e2}:${e2raw} "${JOB}" texture-check)
    final=$(submit final "${small[@]}" --dependency=afterok:${r1}:${r2}:${r4}:${r5}:${sh}:${e3}:${tc} \
        --wrap "source hcm_vendor/scripts/env.sh && hcmv_python -m hcmv summary && hcmv_python -m hcmv tables \
&& hcmv_python -m hcmv figures")
    echo "final:     texture-check=${tc} summary+tables+figures=${final}"
fi
