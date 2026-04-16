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
Input samplesheet    : ${params.input ?: 'AUTO (model-dependent default)'}
Output directory     : ${params.outdir}
Models to run        : ${params.models}
Compare results      : ${params.compare}
Debug mode           : ${params.debug}
"""

// Include modules
include { CINEMA_PREPROCESS; CINEMA_SEGMENT } from './modules/cinema'
include { NNFORMER_PREPROCESS; NNFORMER_SEGMENT } from './modules/nnformer'
include { VSA3L_PREPROCESS; VSA3L_SEGMENT } from './modules/vsa3l'
include { ATRIAL_NNUNET_PREPROCESS; ATRIAL_NNUNET_SEGMENT } from './modules/atrial_nnunet'
include { CINEMA_DISCOVER_MODELS; NNFORMER_DISCOVER_MODELS; VSA3L_DISCOVER_MODELS; ATRIAL_NNUNET_DISCOVER_MODELS } from './modules/discover_models'
include { COMPUTE_METRICS; AGGREGATE_RESULTS; GENERATE_REPORT } from './modules/comparison'
include { GENERATE_DEBUG_REPORT } from './modules/debug'
include { DISCOVER_POSTPROCESS_INPUTS; POSTPROCESS_LV_MYO; VISUALIZE_POSTPROCESS_DELTA } from './modules/postprocess'
include { GENERATE_SEGMENTATION_PREVIEWS } from './modules/visualization'
include { validateInput } from './lib/utils'

def normalizeOptionalPath(value) {
    if (value == null) {
        return null
    }
    def normalized = value.toString().trim()
    return normalized ? normalized : null
}

def parseModels(modelsParam) {
    return (modelsParam ?: '')
        .toString()
        .tokenize(',')
        .collect { it.trim().toLowerCase() }
        .findAll { it }
}

def stageMbasFile(File source, File target, String contextLabel) {
    target.parentFile?.mkdirs()

    if (target.exists()) {
        try {
            if (target.canonicalPath == source.canonicalPath) {
                return
            }
        } catch (Exception ignored) {
            // If canonical path comparison fails, replace target conservatively below.
        }
        if (!target.delete()) {
            exit 1, "ERROR: ${contextLabel}: Could not replace staged file ${target}"
        }
    }

    try {
        java.nio.file.Files.createSymbolicLink(target.toPath(), source.toPath())
    } catch (Exception ignored) {
        java.nio.file.Files.copy(
            source.toPath(),
            target.toPath(),
            java.nio.file.StandardCopyOption.REPLACE_EXISTING
        )
    }
}

def resolveMbasSamplesheet(mbasDefault, mbasRoot, contextLabel) {
    def configuredMbas = normalizeOptionalPath(mbasDefault)
    if (configuredMbas) {
        return file(configuredMbas).toAbsolutePath().toString()
    }

    def datasetRootPath = normalizeOptionalPath(mbasRoot)
    if (!datasetRootPath) {
        return null
    }

    def datasetRoot = new File(datasetRootPath)
    def generatedDir = file("${projectDir}/.cache/generated_inputs")
    generatedDir.mkdirs()
    def generatedCsv = new File(generatedDir.toString(), 'mbas_default_samplesheet.csv')

    def imagesDir = new File(datasetRoot.toString(), 'imagesTr')
    def labelsDir = new File(datasetRoot.toString(), 'labelsTr')

    def rowCount = 0

    generatedCsv.withWriter('UTF-8') { writer ->
        writer.writeLine('patient_id,image,ground_truth,info_cfg')

        if (imagesDir.exists() && labelsDir.exists()) {
            def imageFiles = imagesDir.listFiles()?.findAll { it.name.endsWith('_0000.nii.gz') }?.sort { it.name }
            if (!imageFiles) {
                exit 1, "ERROR: ${contextLabel}: No MBAS inputs found under ${imagesDir}"
            }

            imageFiles.each { imageFile ->
                def patientId = imageFile.name.replace('_0000.nii.gz', '')
                def gtFile = new File(labelsDir, "${patientId}.nii.gz")
                if (!gtFile.exists()) {
                    log.warn "Skipping MBAS sample ${patientId}: missing label file ${gtFile}"
                    return
                }
                writer.writeLine("${patientId},${imageFile.absolutePath},${gtFile.absolutePath},")
                rowCount++
            }
            return
        }

        // Support MBAS training-style layout:
        //   MBAS_001/MBAS_001_gt.nii.gz, MBAS_001/MBAS_001_label.nii.gz
        def caseDirs = datasetRoot.listFiles()?.findAll {
            it.isDirectory() && it.name ==~ /MBAS_\d{3}/
        }?.sort { it.name }

        if (!caseDirs) {
            exit 1, "ERROR: ${contextLabel}: MBAS root must be either nnUNet layout (imagesTr/labelsTr) or MBAS training layout (MBAS_###/). Got: ${datasetRoot}"
        }

        def stagedRoot = new File(generatedDir.toString(), 'mbas_dataset001_lge')
        def stagedImagesDir = new File(stagedRoot, 'imagesTr')
        def stagedLabelsDir = new File(stagedRoot, 'labelsTr')
        stagedImagesDir.mkdirs()
        stagedLabelsDir.mkdirs()

        caseDirs.each { caseDir ->
            def patientId = caseDir.name

            def imageFile = new File(caseDir, "${patientId}_gt.nii.gz")
            if (!imageFile.exists()) {
                imageFile = caseDir.listFiles()?.find { it.name.endsWith('.nii.gz') && !it.name.contains('_label') }
            }

            def gtFile = new File(caseDir, "${patientId}_label.nii.gz")
            if (!gtFile.exists()) {
                gtFile = caseDir.listFiles()?.find { it.name.endsWith('_label.nii.gz') }
            }

            if (!imageFile?.exists() || !gtFile?.exists()) {
                log.warn "Skipping MBAS sample ${patientId}: expected image/label pair not found in ${caseDir}"
                return
            }

            def stagedImage = new File(stagedImagesDir, "${patientId}_0000.nii.gz")
            def stagedLabel = new File(stagedLabelsDir, "${patientId}.nii.gz")

            stageMbasFile(imageFile, stagedImage, contextLabel)
            stageMbasFile(gtFile, stagedLabel, contextLabel)

            writer.writeLine("${patientId},${stagedImage.absolutePath},${stagedLabel.absolutePath},")
            rowCount++
        }
    }

    if (rowCount == 0) {
        exit 1, "ERROR: ${contextLabel}: Generated MBAS samplesheet has no valid rows. Check MBAS files under ${datasetRoot}."
    }

    return generatedCsv.toString()
}

def resolveAcdcSamplesheet(acdcDefault, acdcRoot, acdcDataset, contextLabel) {
    def configuredAcdc = normalizeOptionalPath(acdcDefault)
    if (configuredAcdc) {
        return file(configuredAcdc).toAbsolutePath().toString()
    }

    def datasetRootPath = normalizeOptionalPath(acdcRoot)
    if (!datasetRootPath) {
        return null
    }

    def datasetName = normalizeOptionalPath(acdcDataset) ?: 'testing'
    def dataDir = new File(datasetRootPath, datasetName)
    if (!dataDir.exists()) {
        exit 1, "ERROR: ${contextLabel}: ACDC dataset directory not found: ${dataDir}. Set --acdc_dir (or CASC_ACDC_DIR) and optionally --acdc_dataset."
    }

    def patientDirs = dataDir.listFiles()?.findAll {
        it.isDirectory() && it.name.startsWith('patient')
    }?.sort { it.name }

    if (!patientDirs) {
        exit 1, "ERROR: ${contextLabel}: No patient folders found under ${dataDir}."
    }

    def generatedDir = file("${projectDir}/.cache/generated_inputs")
    generatedDir.mkdirs()
    def generatedCsv = new File(generatedDir.toString(), "acdc_${datasetName}_samplesheet.csv")

    def rowCount = 0
    generatedCsv.withWriter('UTF-8') { writer ->
        writer.writeLine('patient_id,image,ground_truth,info_cfg')

        patientDirs.each { patientDir ->
            def patientId = patientDir.name
            def image4d = new File(patientDir, "${patientId}_4d.nii.gz")
            if (!image4d.exists()) {
                log.warn "Skipping ACDC sample ${patientId}: missing 4D image ${image4d}"
                return
            }

            def infoCfg = new File(patientDir, 'Info.cfg')
            def infoPath = infoCfg.exists() ? infoCfg.absolutePath : ''
            writer.writeLine("${patientId},${image4d.absolutePath},${patientDir.absolutePath},${infoPath}")
            rowCount++
        }
    }

    if (rowCount == 0) {
        exit 1, "ERROR: ${contextLabel}: Generated ACDC samplesheet has no valid rows under ${dataDir}."
    }

    return generatedCsv.toString()
}

def assertSlurmAccountForProfile(contextLabel) {
    def profiles = (workflow.profile ?: '')
        .toString()
        .tokenize(',')
        .collect { it.trim().toLowerCase() }
        .findAll { it }

    if ('slurm' in profiles && !normalizeOptionalPath(params.slurm_account)) {
        exit 1, "ERROR: ${contextLabel}: SLURM profile is active but no account is configured. Set --slurm_account, CASC_SLURM_ACCOUNT, or .casc/user.config."
    }
}

def resolveEffectiveSamplesheet(inputParam, modelsToRun, acdcDefault, acdcRoot, acdcDataset, mbasDefault, mbasRoot, contextLabel = 'main workflow') {
    def explicitInput = normalizeOptionalPath(inputParam)
    if (explicitInput) {
        return file(explicitInput).toAbsolutePath().toString()
    }

    def hasAtrial = 'atrial_nnunet' in modelsToRun
    def hasSax = ('all' in modelsToRun) || ['cinema', 'nnformer', 'vsa3l'].any { it in modelsToRun }

    if (hasAtrial && hasSax) {
        exit 1, "ERROR: ${contextLabel}: Mixed SAX+atrial models were requested without an explicit --input. Please provide --input <samplesheet.csv>."
    }

    if (hasAtrial) {
        def mbasInput = resolveMbasSamplesheet(mbasDefault, mbasRoot, contextLabel)
        if (!mbasInput) {
            exit 1, "ERROR: ${contextLabel}: atrial_nnunet requested without --input, but MBAS default is not configured. Set --default_inputs.mbas, CASC_MBAS_SAMPLESHEET, or --atrial_nnunet.mbas_root (or CASC_MBAS_ROOT)."
        }
        return file(mbasInput).toAbsolutePath().toString()
    }

    def acdcInput = resolveAcdcSamplesheet(acdcDefault, acdcRoot, acdcDataset, contextLabel)
    if (!acdcInput) {
        exit 1, "ERROR: ${contextLabel}: ACDC default input is not configured. Set --default_inputs.acdc or --acdc_dir (or CASC_ACDC_DIR)."
    }
    return file(acdcInput).toAbsolutePath().toString()
}

/*
========================================================================================
    MAIN WORKFLOW
========================================================================================
*/

workflow {

    assertSlurmAccountForProfile('main workflow')

    def models_to_run = parseModels(params.models)
    def effective_input_samplesheet = resolveEffectiveSamplesheet(
        params.input,
        models_to_run,
        params.default_inputs.acdc,
        params.acdc_dir,
        params.acdc_dataset,
        params.default_inputs.mbas,
        params.atrial_nnunet.mbas_root,
        'main workflow'
    )

    log.info "Resolved input samplesheet : ${effective_input_samplesheet}"

    def debug_outdir = file(params.outdir).toAbsolutePath().toString()
    def debug_input_samplesheet = file(effective_input_samplesheet).toAbsolutePath().toString()
    
    // Validate and parse input samplesheet
    ch_input = Channel
        .fromPath(effective_input_samplesheet, checkIfExists: true)
        .splitCsv(header: true)
        .map { row -> 
            def patient_id = row.patient_id
            def image_file = file(row.image, checkIfExists: true)
            def gt_file = row.ground_truth ? file(row.ground_truth, checkIfExists: true) : null
            def info_file = row.info_cfg ? file(row.info_cfg, checkIfExists: true) : null
            [ patient_id, image_file, gt_file, info_file ]
        }

    def auto_discover = params.auto_discover_models as boolean
    
    // Initialize result channels
    ch_cinema_results = Channel.empty()
    ch_nnformer_results = Channel.empty()
    ch_vsa3l_results = Channel.empty()
    ch_atrial_nnunet_results = Channel.empty()

    // Deterministic cache keys for model-specific preprocessing outputs
    def cinema_preprocess_key = 'cinema_s1.0x1.0x10.0_crop192x192_v1'
    def nnformer_preprocess_key = 'nnformer_frame_extract_v1'
    def vsa3l_preprocess_key = "vsa3l_input${params.vsa3l.input_size.join('x')}_v1"
    def atrial_nnunet_preprocess_key = 'atrial_nnunet_frame_extract_v1'
    
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

        ch_cinema_preprocess_input = ch_input
            .map { patient_id, image, gt, info -> [ patient_id, image, gt, info, cinema_preprocess_key ] }

        CINEMA_PREPROCESS(ch_cinema_preprocess_input)
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

        ch_nnformer_preprocess_input = ch_input
            .map { patient_id, image, gt, info -> [ patient_id, image, gt, info, nnformer_preprocess_key ] }

        NNFORMER_PREPROCESS(ch_nnformer_preprocess_input)
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

        ch_vsa3l_preprocess_input = ch_input
            .map { patient_id, image, gt, info -> [ patient_id, image, gt, info, vsa3l_preprocess_key ] }

        VSA3L_PREPROCESS(ch_vsa3l_preprocess_input)
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

    // Run atrial nnUNet model
    if ('atrial_nnunet' in models_to_run) {
        log.info "Running atrial nnUNet segmentation model..."
        if (auto_discover) {
            ATRIAL_NNUNET_DISCOVER_MODELS(file("${projectDir}/bin/discover_models.py"))
            ch_atrial_nnunet_variants = ATRIAL_NNUNET_DISCOVER_MODELS.out.models
                .splitCsv(header: true)
                .map { row ->
                    def folds_csv = row.fold ? row.fold.toString() : params.atrial_nnunet.folds.join(',')
                    [ folds_csv, row.model_tag ]
                }
        } else {
            def folds_csv = params.atrial_nnunet.folds.join(',')
            ch_atrial_nnunet_variants = Channel.of([folds_csv, "atrial_nnunet__dataset001_lge_2d_folds${params.atrial_nnunet.folds.size()}"])
        }

        ch_atrial_nnunet_preprocess_input = ch_input
            .map { patient_id, image, gt, info -> [ patient_id, image, gt, info, atrial_nnunet_preprocess_key ] }

        ATRIAL_NNUNET_PREPROCESS(ch_atrial_nnunet_preprocess_input)
        ch_atrial_nnunet_inputs = ATRIAL_NNUNET_PREPROCESS.out.preprocessed
            .combine(ch_atrial_nnunet_variants)
            .map { patient_id, preprocessed_dir, gt, info, folds, model_tag ->
                [ patient_id, preprocessed_dir, gt, info, folds, model_tag ]
            }

        ATRIAL_NNUNET_SEGMENT(ch_atrial_nnunet_inputs)
        ch_atrial_nnunet_results = ATRIAL_NNUNET_SEGMENT.out.segmentation
            .map { patient_id, seg_ed, seg_es, meta ->
                [ patient_id, meta.model_tag, seg_ed, seg_es, meta ]
            }
    }
    
    // Merge all segmentation results
    ch_all_segmentations = ch_cinema_results
        .mix(ch_nnformer_results)
        .mix(ch_vsa3l_results)
        .mix(ch_atrial_nnunet_results)
    
    // Optional postprocess context (original image + optional Info.cfg)
    ch_input_context = ch_input
        .map { patient_id, image, gt, info ->
            [ patient_id, image.toString(), gt ? gt.toString() : '', info ? info.toString() : '' ]
        }

    if (params.visualization.enabled) {
        ch_preview_inputs = ch_all_segmentations
            .combine(ch_input_context, by: 0)
            .map { patient_id, model, seg_ed, seg_es, meta, image_path, gt_path, info_cfg ->
                def resolved_image_path = image_path
                def resolved_gt_path = gt_path

                if (meta?.architecture == 'cinema') {
                    def preprocessed_img = file("${params.outdir}/cinema/preprocessed/${patient_id}_preprocessed/${patient_id}_sax_t.nii.gz")
                    if (preprocessed_img.exists()) {
                        resolved_image_path = preprocessed_img.toAbsolutePath().toString()
                    }

                    if (gt_path) {
                        def preprocessed_gt_dir = file("${params.outdir}/cinema/preprocessed/${patient_id}_preprocessed")
                        if (preprocessed_gt_dir.exists()) {
                            resolved_gt_path = preprocessed_gt_dir.toAbsolutePath().toString()
                        }
                    }
                }
                [ patient_id, model, seg_ed, seg_es, resolved_image_path, resolved_gt_path, info_cfg, meta?.architecture ?: 'unknown' ]
            }

        GENERATE_SEGMENTATION_PREVIEWS(ch_preview_inputs)
    }

    ch_segmentations_for_metrics = ch_all_segmentations

    if (params.postprocess.enabled) {
        ch_postprocess_candidates = ch_all_segmentations
            .filter { patient_id, model, seg_ed, seg_es, meta ->
                meta?.architecture != 'atrial_nnunet'
            }

        ch_atrial_segmentations = ch_all_segmentations
            .filter { patient_id, model, seg_ed, seg_es, meta ->
                meta?.architecture == 'atrial_nnunet'
            }

        ch_postprocess_inputs = ch_postprocess_candidates
            .combine(ch_input_context, by: 0)
            .map { patient_id, model, seg_ed, seg_es, meta, image_path, gt_path, info_cfg ->
                def resolved_image_path = image_path
                if (meta?.architecture == 'cinema') {
                    def preprocessed_img = file("${params.outdir}/cinema/preprocessed/${patient_id}_preprocessed/${patient_id}_sax_t.nii.gz")
                    if (preprocessed_img.exists()) {
                        resolved_image_path = preprocessed_img.toAbsolutePath().toString()
                    } else {
                        log.warn "CineMA preprocessed image not found for ${patient_id}; falling back to original image for postprocess: ${image_path}"
                    }
                }
                [ patient_id, model, seg_ed, seg_es, meta, resolved_image_path, info_cfg ]
            }

        POSTPROCESS_LV_MYO(ch_postprocess_inputs)

        if (params.postprocess.use_for_metrics) {
            ch_segmentations_for_metrics = POSTPROCESS_LV_MYO.out.segmentations
                .map { patient_id, model, seg_ed, seg_es, meta_pp, image_path, info_cfg ->
                    [ patient_id, model, seg_ed, seg_es, meta_pp ]
                }
                .mix(ch_atrial_segmentations)
        }

        VISUALIZE_POSTPROCESS_DELTA(POSTPROCESS_LV_MYO.out.before_after)
    }

    // Compute metrics if ground truth is available
    ch_input_with_gt = ch_input
        .filter { patient_id, image, gt, info -> gt != null }
        .map { patient_id, image, gt, info -> [ patient_id, gt ] }
    
    ch_for_metrics = ch_segmentations_for_metrics
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
        ch_all_seg_files = ch_segmentations_for_metrics
            .map { patient_id, model, seg_ed, seg_es, meta -> [seg_ed, seg_es] }
            .flatten()
            .collect()
        GENERATE_REPORT(
            AGGREGATE_RESULTS.out.summary,
            ch_all_seg_files,
            params.models
        )
    }

    if (params.debug) {
        def ch_debug_trigger = params.compare
            ? GENERATE_REPORT.out.report.collect()
            : COMPUTE_METRICS.out.metrics.collect()

        GENERATE_DEBUG_REPORT(
            ch_debug_trigger,
            debug_outdir,
            debug_input_samplesheet,
            params.models,
            workflow.runName,
            workflow.duration.toString(),
            workflow.start.toString(),
            workflow.success
        )
    }
}


workflow POSTPROCESS_ONLY {
    assertSlurmAccountForProfile('POSTPROCESS_ONLY')

    def postprocess_samplesheet = normalizeOptionalPath(params.postprocess.samplesheet)
    def models_to_run = parseModels(params.models)
    def samplesheet_path = postprocess_samplesheet
        ? file(postprocess_samplesheet).toAbsolutePath().toString()
        : resolveEffectiveSamplesheet(
            params.input,
            models_to_run,
            params.default_inputs.acdc,
            params.acdc_dir,
            params.acdc_dataset,
            params.default_inputs.mbas,
            params.atrial_nnunet.mbas_root,
            'POSTPROCESS_ONLY'
        )
    def results_dir = file(params.postprocess.results_dir ?: params.outdir).toAbsolutePath().toString()

    ch_samplesheet = Channel.fromPath(samplesheet_path, checkIfExists: true)

    DISCOVER_POSTPROCESS_INPUTS(
        ch_samplesheet,
        results_dir,
        params.models
    )

    ch_discovered = DISCOVER_POSTPROCESS_INPUTS.out.inputs
        .splitCsv(header: true)
        .map { row ->
            def architecture = (row.architecture ?: 'unknown').toLowerCase()
            def resolvedImage = row.image
            if (architecture == 'cinema') {
                def preprocessedImg = file("${results_dir}/cinema/preprocessed/${row.patient_id}_preprocessed/${row.patient_id}_sax_t.nii.gz")
                if (preprocessedImg.exists()) {
                    resolvedImage = preprocessedImg.toAbsolutePath().toString()
                } else {
                    log.warn "CineMA preprocessed image not found for ${row.patient_id} in POSTPROCESS_ONLY; using samplesheet image ${row.image}"
                }
            }
            def meta = [architecture: row.architecture ?: 'unknown', model_tag: row.model, discovered: true]
            [
                row.patient_id,
                row.model,
                file(row.seg_ed, checkIfExists: true),
                file(row.seg_es, checkIfExists: true),
                meta,
                resolvedImage,
                row.info_cfg ?: ''
            ]
        }

    POSTPROCESS_LV_MYO(ch_discovered)

    VISUALIZE_POSTPROCESS_DELTA(POSTPROCESS_LV_MYO.out.before_after)
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
