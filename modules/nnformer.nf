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
    label 'process_preprocess'

    storeDir {
        params.preprocess_cache_enabled ? "${params.preprocess_cache_dir}/nnformer/${patient_id}/${preprocess_key}" : null
    }
    
    publishDir "${params.outdir}/nnformer/preprocessed", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(image), val(ground_truth), val(info_cfg), val(preprocess_key)
    
    output:
    tuple val(patient_id), path("${patient_id}_preprocessed"), val(ground_truth), val(info_cfg), emit: preprocessed
    path "versions.yml", emit: versions
    
    script:
    def gt_arg = ground_truth ? "--ground_truth '${ground_truth}'" : ""
    def info_arg = info_cfg ? "--info_cfg '${info_cfg}'" : ""
    def frames_mode = params.frames_mode ?: 'auto'
    def max_frames_arg = params.max_frames ? "--max_frames ${params.max_frames}" : ""
    """
    nnformer_preprocess.py \\
        --input ${image} \\
        --patient_id ${patient_id} \\
        --output_dir ${patient_id}_preprocessed \\
        ${gt_arg} \\
        ${info_arg} \\
        --frames_mode ${frames_mode} \\
        ${max_frames_arg}
    
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
    label 'process_gpu_light'
    
    publishDir "${params.outdir}/nnformer/segmentations", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(preprocessed_dir), val(ground_truth), val(info_cfg), val(fold), val(model_tag)
    
    output:
    tuple val(patient_id), path("${patient_id}_*_${model_tag}.nii.gz"), path("${patient_id}_${model_tag}_manifest.json"), val(meta), emit: segmentation
    path "versions.yml", emit: versions
    
    script:
    def tta = params.nnformer.tta ? "--tta" : ""
    def mixed_precision = params.nnformer.mixed_precision ? "--mixed_precision" : ""
    meta = [architecture: 'nnformer', model_tag: model_tag, fold: fold]
    """
    # Create writable temp directories for nnFormer
    mkdir -p /tmp/nnformer_tmp/raw/nnFormer_raw_data /tmp/nnformer_tmp/raw/nnFormer_cropped_data
    mkdir -p /tmp/nnformer_tmp/preprocessed /tmp/nnformer_tmp/results
    
    # Set APPTAINER environment variables (these get passed into container)
    export APPTAINERENV_nnFormer_raw_data_base=/tmp/nnformer_tmp/raw
    export APPTAINERENV_nnFormer_preprocessed=/tmp/nnformer_tmp/preprocessed
    export APPTAINERENV_RESULTS_FOLDER=/tmp/nnformer_tmp/results
    export APPTAINERENV_MPLCONFIGDIR=/tmp/nnformer_tmp
    
    # Also set regular env vars (for non-containerized runs)
    export nnFormer_raw_data_base=/tmp/nnformer_tmp/raw
    export nnFormer_preprocessed=/tmp/nnformer_tmp/preprocessed
    export RESULTS_FOLDER=/tmp/nnformer_tmp/results
    export MPLCONFIGDIR=/tmp/nnformer_tmp
    
    nnformer_segment.py \\
        --input_dir ${preprocessed_dir} \\
        --patient_id ${patient_id} \\
        --output_prefix ${patient_id} \\
        --fold ${fold} \\
        ${tta} \\
        ${mixed_precision} \\
        --model_tag ${model_tag} \\
        --model_dir /models/nnformer
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        torch: \$(python -c "import torch; print(torch.__version__)")
        nnformer: \$(python -c "import nnformer; print(nnformer.__version__)" 2>/dev/null || echo "N/A")
    END_VERSIONS
    """
}
