/*
========================================================================================
    Comparison Module
========================================================================================
    Processes for computing metrics and generating comparison reports
----------------------------------------------------------------------------------------
*/

/*
 * Compute segmentation metrics (Dice, HD95) against ground truth
 */
process COMPUTE_METRICS {
    tag "$patient_id - $model"
    label 'process_low'
    
    publishDir "${params.outdir}/metrics/${model}", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), val(model), path(seg_ed), path(seg_es), path(ground_truth), val(meta)
    
    output:
    tuple val(patient_id), val(model), path("${patient_id}_${model}_metrics.csv"), emit: metrics
    path "versions.yml", emit: versions
    
    script:
    def arch = meta?.architecture ?: ''
    """
    compute_metrics.py \\
        --patient_id ${patient_id} \\
        --model ${model} \\
        --architecture "${arch}" \\
        --seg_ed ${seg_ed} \\
        --seg_es ${seg_es} \\
        --ground_truth ${ground_truth} \\
        --output ${patient_id}_${model}_metrics.csv
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        medpy: \$(python -c "import medpy; print(medpy.__version__)")
    END_VERSIONS
    """
}

/*
 * Aggregate all metrics across models and patients
 */
process AGGREGATE_RESULTS {
    label 'process_low'
    
    publishDir "${params.outdir}/comparison", mode: params.publish_dir_mode
    
    input:
    path metrics_files
    
    output:
    path "aggregated_metrics.csv", emit: summary
    path "model_comparison.csv", emit: comparison
    path "per_patient_summary.csv", emit: per_patient
    path "versions.yml", emit: versions
    
    script:
    """
    aggregate_results.py \\
        --input_files ${metrics_files} \\
        --output_aggregated aggregated_metrics.csv \\
        --output_comparison model_comparison.csv \\
        --output_per_patient per_patient_summary.csv
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        pandas: \$(python -c "import pandas; print(pandas.__version__)")
    END_VERSIONS
    """
}

/*
 * Generate comprehensive comparison report with visualizations
 */
process GENERATE_REPORT {
    label 'process_low'
    
    publishDir "${params.outdir}/comparison", mode: params.publish_dir_mode
    
    input:
    path summary_file
    path segmentation_files
    val models
    
    output:
    path "comparison_report.html", emit: report
    path "figures/", emit: figures
    path "versions.yml", emit: versions
    
    script:
    """
    mkdir -p figures
    
    generate_report.py \\
        --summary ${summary_file} \\
        --segmentations . \\
        --models "${models}" \\
        --output_report comparison_report.html \\
        --output_figures figures/
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //')
        matplotlib: \$(python -c "import matplotlib; print(matplotlib.__version__)")
        plotly: \$(python -c "import plotly; print(plotly.__version__)" 2>/dev/null || echo "N/A")
    END_VERSIONS
    """
}
