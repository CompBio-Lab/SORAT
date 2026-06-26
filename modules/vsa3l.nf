/*
========================================================================================
    VSA-3L (MONAI) Module
========================================================================================
    Processes for MONAI Ventricular Short Axis 3-Label segmentation model
----------------------------------------------------------------------------------------
*/

/*
 * Preprocess input data for VSA-3L model
 * - Extract 2D slices from 4D volume
 * - Resize to model input size (256x256)
 * - Normalize intensities
 */
process VSA3L_PREPROCESS {
    tag "$patient_id"
    label 'process_preprocess'

    storeDir {
        params.preprocess_cache_enabled ? "${params.preprocess_cache_dir}/vsa3l/${patient_id}/${preprocess_key}" : null
    }
    
    publishDir "${params.outdir}/vsa3l/preprocessed", mode: params.publish_dir_mode
    
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
    def input_size = params.vsa3l.input_size.join(' ')
    """
    vsa3l_preprocess.py \\
        --input ${image} \\
        --patient_id ${patient_id} \\
        --output_dir ${patient_id}_preprocessed \\
        ${gt_arg} \\
        ${info_arg} \\
        --frames_mode ${frames_mode} \\
        ${max_frames_arg} \\
        --input_size ${input_size}
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        nibabel: \$(python -c "import nibabel; print(nibabel.__version__)")
    END_VERSIONS
    """
}

/*
 * Run VSA-3L (MONAI) segmentation inference
 */
process VSA3L_SEGMENT {
    tag "$patient_id"
    label 'process_gpu_light'
    
    publishDir "${params.outdir}/vsa3l/segmentations", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(preprocessed_dir), val(ground_truth), val(info_cfg), val(model_path), val(model_tag)
    
    output:
    tuple val(patient_id), path("${patient_id}_*_${model_tag}.nii.gz"), path("${patient_id}_${model_tag}_manifest.json"), val(meta), emit: segmentation
    path "versions.yml", emit: versions
    
    script:
    meta = [architecture: 'vsa3l', model_tag: model_tag, input_size: params.vsa3l.input_size]
    """
    vsa3l_segment.py \\
        --input_dir ${preprocessed_dir} \\
        --patient_id ${patient_id} \\
        --output_prefix ${patient_id} \\
        --model_path ${model_path} \\
        --bundle_root /models/vsa3l \\
        --model_tag ${model_tag}
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        torch: \$(python -c "import torch; print(torch.__version__)")
        monai: \$(python -c "import monai; print(monai.__version__)")
    END_VERSIONS
    """
}
