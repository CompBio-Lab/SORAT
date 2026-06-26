/*
========================================================================================
    Feature Extraction Module
========================================================================================
    Process for extracting interpretable volumetric, geometric, and radiomics features.
----------------------------------------------------------------------------------------
*/

process EXTRACT_FEATURES {
    tag "$patient_id"
    label 'process_low'

    publishDir "${params.feature_extraction.output_dir ?: "${params.outdir}/features"}", mode: params.publish_dir_mode

    input:
    tuple val(patient_id), val(frame_tag), val(frame_idx), path(raw_image), path(mask), val(info_cfg)

    output:
    path "${patient_id}_features.csv", emit: features
    path "versions.yml", emit: versions

    script:
    def virtualenvPath = (params.feature_extraction.virtualenv_path ?: '').toString().trim()
    def requireVirtualenv = (params.feature_extraction.require_virtualenv == null) ? false : (params.feature_extraction.require_virtualenv as boolean)
    def infoCfgArg = info_cfg ? "--info_cfg '${info_cfg}'" : ""
    """
    # Optional: use packages from a pre-built virtualenv while keeping container python.
    # Do NOT source venv/bin/activate here; that would switch to host python and can
    # trigger GLIBC mismatches inside the container runtime.
    if [[ -n "${virtualenvPath}" ]]; then
        if [[ ! -d "${virtualenvPath}" ]]; then
            echo "ERROR: feature_extraction.virtualenv_path is invalid: ${virtualenvPath}" >&2
            echo "Expected an existing directory path." >&2
            exit 1
        fi

        PY_MM="\$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

        VENV_SITE="${virtualenvPath}/lib/python\${PY_MM}/site-packages"
        if [[ ! -d "\$VENV_SITE" ]]; then
            echo "ERROR: Could not find matching site-packages in virtualenv for container python \${PY_MM}." >&2
            echo "Expected: \$VENV_SITE" >&2
            echo "Create the virtualenv with the same Python major.minor as the container runtime, or use a compatible package path." >&2
            exit 1
        fi

        export PYTHONPATH="\$VENV_SITE:\${PYTHONPATH:-}"

        # Conda-style envs may keep shared libraries in <env>/lib.
        VENV_LIB="${virtualenvPath}/lib"
        if [[ -d "\$VENV_LIB" ]]; then
            export LD_LIBRARY_PATH="\$VENV_LIB:\${LD_LIBRARY_PATH:-}"
        fi
    elif ${requireVirtualenv ? "true" : "false"}; then
        echo "ERROR: feature_extraction.require_virtualenv=true but no feature_extraction.virtualenv_path was provided." >&2
        exit 1
    fi

    # Fail fast with an actionable message if PyRadiomics is not available.
    python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("radiomics") is None:
    sys.stderr.write("PyRadiomics (module 'radiomics') was not found in this runtime.\\n")
    sys.stderr.write("On Sockeye, pass --feature_extraction.virtualenv_path to a venv with compatible site-packages.\\n")
    sys.stderr.write("You can also export SORAT_FEATURE_VENV to avoid passing it each run.\\n")
    sys.exit(1)
PY

    cp ${projectDir}/bin/frame_manifest.py . 2>/dev/null || true

    python /app/bin/extract_features.py \
        --patient_id ${patient_id} \
        --image ${raw_image} \
        --mask ${mask} \
        --frame_tag ${frame_tag} \
        --frame_idx ${frame_idx} \
        ${infoCfgArg} \
        --output_csv ${patient_id}_features.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        simpleitk: \$(python -c "import SimpleITK; print(SimpleITK.__version__)" 2>/dev/null || echo "N/A")
        numpy: \$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "N/A")
        scipy: \$(python -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "N/A")
        pandas: \$(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "N/A")
        pyradiomics: \$(python -c "import radiomics; print(radiomics.__version__)" 2>/dev/null || echo "N/A")
    END_VERSIONS
    """
}
