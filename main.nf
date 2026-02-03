#!/usr/bin/env nextflow

/*
========================================================================================
    CASC - Cardiac Automated Segmentation Comparison Pipeline
========================================================================================
    A modular Nextflow pipeline for cardiac MRI segmentation using multiple deep learning
    models. Supports CineMA, nnFormer, and MONAI VSA-3L models with extensibility for
    adding new models.
    
    GitHub: https://github.com/CompBio-Lab/CASC
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

log.info """
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ____    _    ____   ____                                                    ║
║  / ___|  / \\  / ___| / ___|                                                  ║
║ | |     / _ \\ \\___ \\| |                                                    ║
║ | |___ / ___ \\ ___) | |___                                                   ║
║  \\____/_/   \\_\\____/ \\____|                                               ║
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
include { CINEMA_DISCOVER_MODELS; NNFORMER_DISCOVER_MODELS; VSA3L_DISCOVER_MODELS } from './modules/discover_models'
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
    def auto_discover = params.auto_discover_models as boolean
    
    // Initialize result channels
    ch_cinema_results = Channel.empty()
    ch_nnformer_results = Channel.empty()
    ch_vsa3l_results = Channel.empty()
    
    // Run CineMA model
    if ('cinema' in models_to_run || 'all' in models_to_run) {
        log.info "Running CineMA segmentation model..."
        if (auto_discover) {
            CINEMA_DISCOVER_MODELS(file("${projectDir}/bin/discover_models.py"))
            ch_cinema_variants = CINEMA_DISCOVER_MODELS.out.models
                .splitCsv(header: true)
                .map { row ->
                    [ row.trained_dataset, row.seed, row.model_tag ]
                }
        } else {
            ch_cinema_variants = Channel.fromList(params.cinema.seeds)
                .map { seed ->
                    [ params.cinema.trained_dataset, seed.toString(), "cinema__${params.cinema.trained_dataset}_seed${seed}" ]
                }
        }

        ch_cinema_per_seed = ch_cinema_variants
            .map { trained_dataset, seed, model_tag ->
                [ trained_dataset, seed.toString(), false, model_tag ]
            }

        ch_cinema_ensemble = ch_cinema_variants
            .groupTuple(by: 0)
            .map { trained_dataset, seeds, tags ->
                def unique_seeds = seeds.unique().sort()
                def seeds_csv = unique_seeds.join(',')
                def tag = "cinema__${trained_dataset}_ensemble"
                [ trained_dataset, seeds_csv, unique_seeds.size(), true, tag ]
            }
            .filter { trained_dataset, seeds_csv, seed_count, do_ensemble, tag -> params.cinema.ensemble && seed_count > 1 }
            .map { trained_dataset, seeds_csv, seed_count, do_ensemble, tag -> [ trained_dataset, seeds_csv, do_ensemble, tag ] }

        ch_cinema_models = ch_cinema_per_seed.mix(ch_cinema_ensemble)

        CINEMA_PREPROCESS(ch_input)
        ch_cinema_inputs = CINEMA_PREPROCESS.out.preprocessed
            .combine(ch_cinema_models)
            .map { patient_id, preprocessed_dir, gt, info, trained_dataset, seeds_csv, do_ensemble, model_tag ->
                [ patient_id, preprocessed_dir, gt, info, trained_dataset, seeds_csv, do_ensemble, model_tag ]
            }

        CINEMA_SEGMENT(ch_cinema_inputs)

        ch_cinema_results = CINEMA_SEGMENT.out.segmentation
            .map { patient_id, seg_ed, seg_es, meta -> 
                [ patient_id, meta.model_tag, seg_ed, seg_es, meta ]
            }
    }
    
    // Run nnFormer model
    if ('nnformer' in models_to_run || 'all' in models_to_run) {
        log.info "Running nnFormer segmentation model..."
        if (auto_discover) {
            NNFORMER_DISCOVER_MODELS(file("${projectDir}/bin/discover_models.py"))
            ch_nnformer_variants = NNFORMER_DISCOVER_MODELS.out.models
                .splitCsv(header: true)
                .map { row ->
                    [ (row.fold as int), row.model_tag ]
                }
        } else {
            ch_nnformer_variants = Channel.of(params.nnformer.fold as int)
                .map { fold -> [ fold, "nnformer__fold${fold}" ] }
        }

        NNFORMER_PREPROCESS(ch_input)
        ch_nnformer_inputs = NNFORMER_PREPROCESS.out.preprocessed
            .combine(ch_nnformer_variants)
            .map { patient_id, preprocessed_dir, gt, info, fold, model_tag ->
                [ patient_id, preprocessed_dir, gt, info, fold, model_tag ]
            }

        NNFORMER_SEGMENT(ch_nnformer_inputs)
        ch_nnformer_results = NNFORMER_SEGMENT.out.segmentation
            .map { patient_id, seg_ed, seg_es, meta -> 
                [ patient_id, meta.model_tag, seg_ed, seg_es, meta ]
            }
    }
    
    // Run VSA-3L model
    if ('vsa3l' in models_to_run || 'all' in models_to_run) {
        log.info "Running VSA-3L (MONAI) segmentation model..."
        if (auto_discover) {
            VSA3L_DISCOVER_MODELS(file("${projectDir}/bin/discover_models.py"))
            ch_vsa3l_variants = VSA3L_DISCOVER_MODELS.out.models
                .splitCsv(header: true)
                .map { row ->
                    [ row.model_path, row.model_tag ]
                }
        } else {
            ch_vsa3l_variants = Channel.of(["/models/vsa3l/model.pt", "vsa3l__model"])
        }

        VSA3L_PREPROCESS(ch_input)
        ch_vsa3l_inputs = VSA3L_PREPROCESS.out.preprocessed
            .combine(ch_vsa3l_variants)
            .map { patient_id, preprocessed_dir, gt, info, model_path, model_tag ->
                [ patient_id, preprocessed_dir, gt, info, model_path, model_tag ]
            }

        VSA3L_SEGMENT(ch_vsa3l_inputs)
        ch_vsa3l_results = VSA3L_SEGMENT.out.segmentation
            .map { patient_id, seg_ed, seg_es, meta -> 
                [ patient_id, meta.model_tag, seg_ed, seg_es, meta ]
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
            def gt_for_metrics = gt
            if (meta?.architecture == 'cinema') {
                def pre_dir = "${params.outdir}/cinema/preprocessed/${patient_id}_preprocessed"
                if (file(pre_dir).exists()) {
                    gt_for_metrics = file(pre_dir)
                }
            }
            [ patient_id, model, seg_ed, seg_es, gt_for_metrics, meta ]
        }
    
    COMPUTE_METRICS(ch_for_metrics)
    
    // Aggregate results across all models and patients
    if (params.compare) {
        ch_all_metrics = COMPUTE_METRICS.out.metrics
            .map { patient_id, model, metrics_csv -> metrics_csv }
            .collect()
        AGGREGATE_RESULTS(ch_all_metrics)
        ch_all_seg_files = ch_all_segmentations
            .map { patient_id, model, seg_ed, seg_es, meta -> [seg_ed, seg_es] }
            .flatten()
            .collect()
        GENERATE_REPORT(
            AGGREGATE_RESULTS.out.summary,
            ch_all_seg_files,
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
