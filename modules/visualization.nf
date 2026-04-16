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
    tuple val(patient_id), val(model), path(seg_ed), path(seg_es), val(image_path), val(ground_truth_path), val(info_cfg), val(architecture)

    output:
    path "*/*_ED_preview.png", emit: ed
    path "*/*_ES_preview.png", emit: es
    path "versions.yml", emit: versions

    script:
    safe_model = model.replaceAll('[^A-Za-z0-9_.-]', '_')
    def gt_arg = ground_truth_path ? "--ground_truth ${ground_truth_path}" : ""
    def info_arg = info_cfg ? "--info_cfg ${info_cfg}" : ""
    """
    mkdir -p ${safe_model}

    python /app/bin/generate_segmentation_preview.py \
        --patient_id ${patient_id} \
        --model ${model} \
        --architecture ${architecture} \
        --image ${image_path} \
        ${gt_arg} \
        --seg_ed ${seg_ed} \
        --seg_es ${seg_es} \
        --output_ed_png ${safe_model}/${patient_id}_${safe_model}_ED_preview.png \
        --output_es_png ${safe_model}/${patient_id}_${safe_model}_ES_preview.png \
        ${info_arg}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        matplotlib: \$(python -c "import matplotlib; print(matplotlib.__version__)")
        simpleitk: \$(python -c "import SimpleITK; print(SimpleITK.__version__)")
    END_VERSIONS
    """
}
