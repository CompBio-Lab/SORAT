/*
========================================================================================
    Model Discovery Module
========================================================================================
    Processes for discovering available model variants inside containers
----------------------------------------------------------------------------------------
*/

process CINEMA_DISCOVER_MODELS {
    label 'process_low'

    publishDir "${params.outdir}/cinema", mode: params.publish_dir_mode

    input:
    path script_file

    output:
    path "cinema_models.csv", emit: models
    path "versions.yml", emit: versions

    script:
    """
    python ${script_file} \\
        --architecture cinema \\
        --output cinema_models.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
    END_VERSIONS
    """
}

process NNFORMER_DISCOVER_MODELS {
    label 'process_low'

    publishDir "${params.outdir}/nnformer", mode: params.publish_dir_mode

    input:
    path script_file

    output:
    path "nnformer_models.csv", emit: models
    path "versions.yml", emit: versions

    script:
    """
    python ${script_file} \\
        --architecture nnformer \\
        --output nnformer_models.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
    END_VERSIONS
    """
}

process VSA3L_DISCOVER_MODELS {
    label 'process_low'

    publishDir "${params.outdir}/vsa3l", mode: params.publish_dir_mode

    input:
    path script_file

    output:
    path "vsa3l_models.csv", emit: models
    path "versions.yml", emit: versions

    script:
    """
    python ${script_file} \\
        --architecture vsa3l \\
        --output vsa3l_models.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
    END_VERSIONS
    """
}
