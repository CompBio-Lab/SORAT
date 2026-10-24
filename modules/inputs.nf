/*
========================================================================================
    Input preparation
========================================================================================
    Converts samplesheet images that are not NIfTI (NRRD, MetaImage, DICOM series)
    to NIfTI, applies an explicit spacing override when given, and records
    provenance. NIfTI inputs never reach this process.
----------------------------------------------------------------------------------------
*/

process PREPARE_INPUT {
    tag "$patient_id"
    label 'process_low'

    publishDir "${params.outdir}/inputs/${patient_id}", mode: params.publish_dir_mode

    input:
    tuple val(patient_id), path(image), val(ground_truth), val(gt_directory), val(info_cfg), val(image_spacing), val(dicom_series)

    output:
    // gt_directory: a per-frame ground-truth directory (ACDC/M&Ms-2 layout), passed through untouched
    tuple val(patient_id), path("${patient_id}_image.nii.gz"), val(info_cfg), val(gt_directory), emit: images
    tuple val(patient_id), path("${patient_id}_gt.nii.gz"), optional: true, emit: ground_truth
    path "${patient_id}_input_provenance.json", emit: provenance
    path "versions.yml", emit: versions

    script:
    def gt_arg = ground_truth ? "--ground_truth '${ground_truth}'" : ""
    def spacing_arg = image_spacing ? "--image_spacing '${image_spacing}'" : ""
    def series_arg = dicom_series ? "--dicom_series '${dicom_series}'" : ""
    """
    python /app/bin/prepare_input.py \\
        --patient_id '${patient_id}' \\
        --image '${image}' \\
        ${gt_arg} ${spacing_arg} ${series_arg} \\
        --out_dir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        SimpleITK: \$(python -c "import SimpleITK; print(SimpleITK.Version_VersionString())")
    END_VERSIONS
    """
}
