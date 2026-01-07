#!/usr/bin/env nextflow

/*
========================================================================================
    CASC - Cardiac Automated Segmentation Comparison Pipeline
========================================================================================
    A modular Nextflow pipeline for cardiac MRI segmentation using multiple deep learning
    models. Supports CineMA, nnFormer, and MONAI VSA-3L models with extensibility for
    adding new models.
    
    GitHub: https://github.com/your-org/CASC
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

// Print pipeline header
log.info """
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ____    _    ____   ____                                                    ║
║  / ___|  / \\  / ___| / ___|                                                   ║
║ | |     / _ \\ \\___ \\| |                                                       ║
║ | |___ / ___ \\ ___) | |___                                                    ║
║  \\____/_/   \\_\\____/ \\____|                                                   ║
║                                                                               ║
║  Cardiac Automated Segmentation Comparison Pipeline                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Pipeline Parameters:
--------------------
Input samplesheet    : ${params.input}
Output directory     : ${params.outdir}
Models to run        : ${params.models}
Compare results      : ${params.compare}
"""

// Include modules
include { CINEMA_PREPROCESS; CINEMA_SEGMENT } from './modules/cinema'
include { NNFORMER_PREPROCESS; NNFORMER_SEGMENT } from './modules/nnformer'
include { VSA3L_PREPROCESS; VSA3L_SEGMENT } from './modules/vsa3l'
include { COMPUTE_METRICS; AGGREGATE_RESULTS; GENERATE_REPORT } from './modules/comparison'
include { validateInput } from './lib/utils'

/*
========================================================================================
    MAIN WORKFLOW
========================================================================================
*/

workflow {
    
    // Validate and parse input samplesheet
    ch_input = Channel
        .fromPath(params.input, checkIfExists: true)
        .splitCsv(header: true)
        .map { row -> 
            def patient_id = row.patient_id
            def image_file = file(row.image, checkIfExists: true)
            def gt_file = row.ground_truth ? file(row.ground_truth, checkIfExists: true) : null
            def info_file = row.info_cfg ? file(row.info_cfg, checkIfExists: true) : null
            [ patient_id, image_file, gt_file, info_file ]
        }
    
    // Parse which models to run
    def models_to_run = params.models.tokenize(',').collect { it.trim().toLowerCase() }
    
    // Initialize result channels
    ch_cinema_results = Channel.empty()
    ch_nnformer_results = Channel.empty()
    ch_vsa3l_results = Channel.empty()
    
    // Run CineMA model
    if ('cinema' in models_to_run || 'all' in models_to_run) {
        log.info "Running CineMA segmentation model..."
        CINEMA_PREPROCESS(ch_input)
        CINEMA_SEGMENT(CINEMA_PREPROCESS.out.preprocessed)
        ch_cinema_results = CINEMA_SEGMENT.out.segmentation
            .map { patient_id, seg_ed, seg_es, meta -> 
                [ patient_id, 'cinema', seg_ed, seg_es, meta ]
            }
    }
    
    // Run nnFormer model
    if ('nnformer' in models_to_run || 'all' in models_to_run) {
        log.info "Running nnFormer segmentation model..."
        NNFORMER_PREPROCESS(ch_input)
        NNFORMER_SEGMENT(NNFORMER_PREPROCESS.out.preprocessed)
        ch_nnformer_results = NNFORMER_SEGMENT.out.segmentation
            .map { patient_id, seg_ed, seg_es, meta -> 
                [ patient_id, 'nnformer', seg_ed, seg_es, meta ]
            }
    }
    
    // Run VSA-3L model
    if ('vsa3l' in models_to_run || 'all' in models_to_run) {
        log.info "Running VSA-3L (MONAI) segmentation model..."
        VSA3L_PREPROCESS(ch_input)
        VSA3L_SEGMENT(VSA3L_PREPROCESS.out.preprocessed)
        ch_vsa3l_results = VSA3L_SEGMENT.out.segmentation
            .map { patient_id, seg_ed, seg_es, meta -> 
                [ patient_id, 'vsa3l', seg_ed, seg_es, meta ]
            }
    }
    
    // Merge all segmentation results
    ch_all_segmentations = ch_cinema_results
        .mix(ch_nnformer_results)
        .mix(ch_vsa3l_results)
    
    // Compute metrics if ground truth is available
    ch_input_with_gt = ch_input
        .filter { patient_id, image, gt, info -> gt != null }
        .map { patient_id, image, gt, info -> [ patient_id, gt ] }
    
    ch_for_metrics = ch_all_segmentations
        .combine(ch_input_with_gt, by: 0)
        .map { patient_id, model, seg_ed, seg_es, meta, gt ->
            [ patient_id, model, seg_ed, seg_es, gt, meta ]
        }
    
    COMPUTE_METRICS(ch_for_metrics)
    
    // Aggregate results across all models and patients
    if (params.compare) {
        ch_all_metrics = COMPUTE_METRICS.out.metrics.collect()
        AGGREGATE_RESULTS(ch_all_metrics)
        GENERATE_REPORT(
            AGGREGATE_RESULTS.out.summary,
            ch_all_segmentations.collect(),
            params.models
        )
    }
}

/*
========================================================================================
    WORKFLOW COMPLETION
========================================================================================
*/

workflow.onComplete {
    log.info """
    ============================================================
    Pipeline execution summary
    ============================================================
    Completed at : ${workflow.complete}
    Duration     : ${workflow.duration}
    Success      : ${workflow.success}
    Work dir     : ${workflow.workDir}
    Output dir   : ${params.outdir}
    ============================================================
    """
}

workflow.onError {
    log.error "Pipeline execution stopped with an error: ${workflow.errorMessage}"
}
