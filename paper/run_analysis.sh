#!/usr/bin/env bash
# Run one analysis script in a Python container (pandas, SciPy, matplotlib, nibabel,
# SimpleITK, MedPy). Usage (from paper/): ./run_analysis.sh analysis/seg_accuracy.py
set -euo pipefail
: "${PAPER_SIF:?set PAPER_SIF to the analysis container image}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_ARGS=()
[[ -n "${PAPER_CONTAINER_HOME:-}" ]] && HOME_ARGS=(--home "$PAPER_CONTAINER_HOME")
exec apptainer exec "${HOME_ARGS[@]}" --bind "${SORAT_BIND:-$SORAT_DATA_ROOT,$SORAT_RESULTS}" --pwd "$ROOT" \
    --env PYTHONPATH="$ROOT/analysis" --env MPLCONFIGDIR=/tmp \
    --env SORAT_DATA_ROOT="$SORAT_DATA_ROOT" --env SORAT_RESULTS="$SORAT_RESULTS" \
    --env SORAT_RUNS="${SORAT_RUNS:-}" --env SORAT_SAMPLESHEETS="${SORAT_SAMPLESHEETS:-}" \
    "$PAPER_SIF" python "$@"
