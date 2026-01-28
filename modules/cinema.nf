/*
========================================================================================
    CineMA Module
========================================================================================
    Processes for CineMA cardiac segmentation model
----------------------------------------------------------------------------------------
*/

/*
 * Preprocess input data for CineMA model
 * - Resample to 1.0 x 1.0 x 10.0 mm spacing
 * - Crop to 192 x 192 based on LV center
 * - Normalize intensity
 */
process CINEMA_PREPROCESS {
    tag "$patient_id"
    label 'process_medium'
    
    publishDir "${params.outdir}/cinema/preprocessed", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(image), path(ground_truth), path(info_cfg)
    
    output:
    tuple val(patient_id), path("${patient_id}_preprocessed"), path(ground_truth), path(info_cfg), emit: preprocessed
    path "versions.yml", emit: versions
    
    script:
    def gt_arg = ground_truth ? "--ground_truth ${ground_truth}" : ""
    def info_arg = info_cfg ? "--info_cfg ${info_cfg}" : ""
    """
    cinema_preprocess.py \\
        --input ${image} \\
        --patient_id ${patient_id} \\
        --output_dir ${patient_id}_preprocessed \\
        ${gt_arg} \\
        ${info_arg} \\
        --spacing 1.0 1.0 10.0 \\
        --crop_size 192 192
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        cinema: \$(python -c "import cinema; print(cinema.__version__)" 2>/dev/null || echo "N/A")
    END_VERSIONS
    """
}

/*
 * Run CineMA segmentation inference
 */
process CINEMA_SEGMENT {
    tag "$patient_id"
    label 'process_gpu'
    
    publishDir "${params.outdir}/cinema/segmentations", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(preprocessed_dir), path(ground_truth), path(info_cfg), val(trained_dataset), val(seeds_csv), val(do_ensemble), val(model_tag)
    
    output:
    tuple val(patient_id), path("${patient_id}_ED_${model_tag}.nii.gz"), path("${patient_id}_ES_${model_tag}.nii.gz"), val(meta), emit: segmentation
    path "versions.yml", emit: versions
    
    script:
    def ensemble = do_ensemble ? "--ensemble" : ""
    meta = [architecture: 'cinema', model_tag: model_tag, trained_dataset: trained_dataset, seeds: seeds_csv, ensemble: do_ensemble]
    """
    cinema_segment.py \\
        --input_dir ${preprocessed_dir} \\
        --patient_id ${patient_id} \\
        --output_prefix ${patient_id} \\
        --trained_dataset ${trained_dataset} \\
        --seeds ${seeds_csv} \\
        ${ensemble} \\
        --model_tag ${model_tag} \\
        --model_dir /models/cinema
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        torch: \$(python -c "import torch; print(torch.__version__)")
        cinema: \$(python -c "import cinema; print(cinema.__version__)" 2>/dev/null || echo "N/A")
    END_VERSIONS
    """
}
