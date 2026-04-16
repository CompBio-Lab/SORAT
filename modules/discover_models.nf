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

    awk -F',' 'NR>1 {n++} END {exit(n>0?0:1)}' cinema_models.csv || {
        echo "ERROR: No CineMA models discovered. Check container model payload under /models/cinema." >&2
        exit 1
    }

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

    awk -F',' 'NR>1 {n++} END {exit(n>0?0:1)}' nnformer_models.csv || {
        echo "ERROR: No nnFormer models discovered. Check container model payload under /models/nnformer." >&2
        exit 1
    }

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

    awk -F',' 'NR>1 {n++} END {exit(n>0?0:1)}' vsa3l_models.csv || {
        echo "ERROR: No VSA-3L models discovered. Check container model payload under /models/vsa3l." >&2
        exit 1
    }

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
    END_VERSIONS
    """
}

process ATRIAL_NNUNET_DISCOVER_MODELS {
    label 'process_low'

    publishDir "${params.outdir}/atrial_nnunet", mode: params.publish_dir_mode

    input:
    path script_file

    output:
    path "atrial_nnunet_models.csv", emit: models
    path "versions.yml", emit: versions

    script:
    """
    python ${script_file} \
        --architecture atrial_nnunet \
        --output atrial_nnunet_models.csv

    awk -F',' 'NR>1 {n++} END {exit(n>0?0:1)}' atrial_nnunet_models.csv || {
        echo "ERROR: No atrial nnUNet models discovered. Check container model payload under /models/atrial_nnunet." >&2
        exit 1
    }

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
    END_VERSIONS
    """
}
