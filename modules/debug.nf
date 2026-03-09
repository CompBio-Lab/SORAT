/*
========================================================================================
    Debug Module
========================================================================================
    Process for generating execution/resource/scientific utility debug analytics
----------------------------------------------------------------------------------------
*/

process GENERATE_DEBUG_REPORT {
    label 'process_low'

    publishDir "${params.outdir}/debug", mode: params.publish_dir_mode

    input:
    val _trigger
    val outdir
    val input_samplesheet
    val model_selection
    val run_name
    val workflow_duration
    val workflow_start
    val workflow_success

    output:
    path "text/", emit: text
    path "figures/", emit: figures
    path "versions.yml", emit: versions

    script:
    """
    mkdir -p text figures

    python /app/bin/generate_debug_report.py \\
        --outdir "${outdir}" \\
        --input_samplesheet "${input_samplesheet}" \\
        --model_selection "${model_selection}" \\
        --run_name "${run_name}" \\
        --workflow_duration "${workflow_duration}" \\
        --workflow_start "${workflow_start}" \\
        --workflow_success "${workflow_success}" \\
        --text_dir text \\
        --figures_dir figures

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)")
        matplotlib: \$(python -c "import matplotlib; print(matplotlib.__version__)")
    END_VERSIONS
    """
}
