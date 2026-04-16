/*
========================================================================================
    Atrial nnUNet Module
========================================================================================
    Processes for atrial nnUNetv2 segmentation model
----------------------------------------------------------------------------------------
*/

process ATRIAL_NNUNET_PREPROCESS {
    tag "$patient_id"
    label 'process_preprocess'

    storeDir {
        params.preprocess_cache_enabled ? "${params.preprocess_cache_dir}/atrial_nnunet/${patient_id}/${preprocess_key}" : null
    }

    publishDir "${params.outdir}/atrial_nnunet/preprocessed", mode: params.publish_dir_mode

    input:
    tuple val(patient_id), path(image), val(ground_truth), val(info_cfg), val(preprocess_key)

    output:
    tuple val(patient_id), path("${patient_id}_preprocessed"), val(ground_truth), val(info_cfg), emit: preprocessed
    path "versions.yml", emit: versions

    script:
    def gt_arg = ground_truth ? "--ground_truth ${ground_truth}" : ""
    def info_arg = info_cfg ? "--info_cfg ${info_cfg}" : ""
    """
    atrial_nnunet_preprocess.py \\
        --input ${image} \\
        --patient_id ${patient_id} \\
        --output_dir ${patient_id}_preprocessed \\
        ${gt_arg} \\
        ${info_arg}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        simpleitk: \$(python -c "import SimpleITK; print(SimpleITK.__version__)")
    END_VERSIONS
    """
}

process ATRIAL_NNUNET_SEGMENT {
    tag "$patient_id"
    label 'process_gpu_light'

    publishDir "${params.outdir}/atrial_nnunet/segmentations", mode: params.publish_dir_mode

    input:
    tuple val(patient_id), path(preprocessed_dir), val(ground_truth), val(info_cfg), val(folds), val(model_tag)

    output:
    tuple val(patient_id), path("${patient_id}_ED_${model_tag}.nii.gz"), path("${patient_id}_ES_${model_tag}.nii.gz"), val(meta), emit: segmentation
    path "versions.yml", emit: versions

    script:
    def probs = params.atrial_nnunet.save_probabilities ? "--save_probabilities" : ""
    meta = [architecture: 'atrial_nnunet', model_tag: model_tag, folds: folds, dataset_id: params.atrial_nnunet.dataset_id, configuration: params.atrial_nnunet.configuration]
    """
    export PYTHONNOUSERSITE=1

    atrial_nnunet_segment.py \\
        --input_dir ${preprocessed_dir} \\
        --patient_id ${patient_id} \\
        --output_prefix ${patient_id} \\
        --model_dir /models/atrial_nnunet \\
        --folds "${folds}" \\
        --model_tag ${model_tag} \\
        --dataset_id ${params.atrial_nnunet.dataset_id} \\
        --configuration ${params.atrial_nnunet.configuration} \\
        ${probs}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        torch: \$(python -c "import torch; print(torch.__version__)")
    END_VERSIONS
    """
}
