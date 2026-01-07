/*
========================================================================================
    nnFormer Module
========================================================================================
    Processes for nnFormer cardiac segmentation model
----------------------------------------------------------------------------------------
*/

/*
 * Preprocess input data for nnFormer model
 * - Convert to nnFormer naming convention (add _0000 suffix)
 * - Extract ED and ES frames if 4D input
 */
process NNFORMER_PREPROCESS {
    tag "$patient_id"
    label 'process_low'
    
    publishDir "${params.outdir}/nnformer/preprocessed", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(image), path(ground_truth), path(info_cfg)
    
    output:
    tuple val(patient_id), path("${patient_id}_preprocessed"), path(ground_truth), path(info_cfg), emit: preprocessed
    path "versions.yml", emit: versions
    
    script:
    def gt_arg = ground_truth ? "--ground_truth ${ground_truth}" : ""
    def info_arg = info_cfg ? "--info_cfg ${info_cfg}" : ""
    """
    nnformer_preprocess.py \\
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

/*
 * Run nnFormer segmentation inference
 */
process NNFORMER_SEGMENT {
    tag "$patient_id"
    label 'process_gpu'
    
    publishDir "${params.outdir}/nnformer/segmentations", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(preprocessed_dir), path(ground_truth), path(info_cfg)
    
    output:
    tuple val(patient_id), path("${patient_id}_ED_nnformer.nii.gz"), path("${patient_id}_ES_nnformer.nii.gz"), val(meta), emit: segmentation
    path "versions.yml", emit: versions
    
    script:
    def tta = params.nnformer.tta ? "--tta" : ""
    def mixed_precision = params.nnformer.mixed_precision ? "--mixed_precision" : ""
    meta = [model: 'nnformer', fold: params.nnformer.fold]
    """
    # Set nnFormer environment variables
    export nnFormer_raw_data_base=/models/nnformer/nnFormer_raw
    export nnFormer_preprocessed=/models/nnformer/nnFormer_preprocessed
    export RESULTS_FOLDER=/models/nnformer/nnFormer_trained_models
    
    nnformer_segment.py \\
        --input_dir ${preprocessed_dir} \\
        --patient_id ${patient_id} \\
        --output_prefix ${patient_id} \\
        --fold ${params.nnformer.fold} \\
        ${tta} \\
        ${mixed_precision} \\
        --model_dir /models/nnformer
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        torch: \$(python -c "import torch; print(torch.__version__)")
        nnformer: \$(python -c "import nnformer; print(nnformer.__version__)" 2>/dev/null || echo "N/A")
    END_VERSIONS
    """
}
