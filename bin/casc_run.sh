#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
USER_CONFIG="${REPO_ROOT}/.casc/user.config"

if [[ ! -f "${USER_CONFIG}" ]]; then
    echo "Warning: ${USER_CONFIG} not found."
    echo "Run: python3 ${REPO_ROOT}/bin/casc_setup.py"
    echo "Proceeding with defaults and CLI/env overrides..."
fi

exec nextflow run "${REPO_ROOT}/main.nf" "$@"
