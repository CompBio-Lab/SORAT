/*
========================================================================================
    Visualization Module
========================================================================================
    Generate quick-look segmentation overlays for ED/ES results.
----------------------------------------------------------------------------------------
*/

process GENERATE_SEGMENTATION_PREVIEWS {
    tag "$patient_id - $model"
    label 'process_low'

    publishDir "${params.outdir}/previews", mode: params.publish_dir_mode

    input:
    tuple val(patient_id), val(model), val(frame_tag), val(frame_idx), path(seg), val(image_path), val(ground_truth_path), val(info_cfg), val(architecture)

    output:
    path "*/${patient_id}_${safe_model}_${frame_tag}_preview.png", emit: previews
    path "versions.yml", emit: versions

    script:
    safe_model = model.replaceAll('[^A-Za-z0-9_.-]', '_')
    def gt_arg = ground_truth_path ? "--ground_truth '${ground_truth_path}'" : ""
    def info_arg = info_cfg ? "--info_cfg '${info_cfg}'" : ""
    """
    cp ${projectDir}/bin/frame_manifest.py . 2>/dev/null || true
    cp ${projectDir}/bin/geometry_utils.py . 2>/dev/null || true
    cp ${projectDir}/bin/generate_segmentation_preview.py . 2>/dev/null || true
    mkdir -p ${safe_model}

    export PYTHONPATH="\$PWD:/app/bin:\${PYTHONPATH:-}"
    MPL_USER="\${USER:-\${LOGNAME:-\$(id -un 2>/dev/null || echo user)}}"
    export MPLCONFIGDIR="\${TMPDIR:-/tmp}/matplotlib-\${MPL_USER}"
    mkdir -p "\$MPLCONFIGDIR"

    python generate_segmentation_preview.py \
        --patient_id ${patient_id} \
        --model ${model} \
        --architecture ${architecture} \
        --image '${image_path}' \
        ${gt_arg} \
        --seg ${seg} \
        --frame_tag ${frame_tag} \
        --frame_idx ${frame_idx} \
        --output_png ${safe_model}/${patient_id}_${safe_model}_${frame_tag}_preview.png \
        ${info_arg}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        matplotlib: \$(python -c "import matplotlib; print(matplotlib.__version__)")
        simpleitk: \$(python -c "import SimpleITK; print(SimpleITK.__version__)")
    END_VERSIONS
    """
}
