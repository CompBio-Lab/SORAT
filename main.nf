#!/usr/bin/env nextflow

import groovy.io.FileType
import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import groovy.transform.Field
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/*
========================================================================================
    SORAT - Segmentation Orchestration and Reproducible Analysis Toolkit
========================================================================================
    A modular Nextflow pipeline for cardiac MRI segmentation using multiple deep learning
    models. Supports CineMA, nnFormer, nnU-Net, and MONAI models with extensibility for
    adding new models.

    GitHub: https://github.com/CompBio-Lab/SORAT
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

log.info """
╔═══════════════════════════════════════════════════════════════════════════════╗
║  SORAT: Segmentation Orchestration and Reproducible Analysis Toolkit          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Pipeline Parameters:
--------------------
Input samplesheet    : ${params.input ?: 'AUTO (model-family default)'}
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
include { EXTRACT_FEATURES } from './modules/features'
include { validateInput } from './lib/utils'
include { parseManifest; frameTagFromFilename; discoverFramePattern; normalizeSegList } from './lib/frame_utils'

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

def countSamplesheetRows(String samplesheetPath) {
    def path = normalizeOptionalPath(samplesheetPath)
    if (!path) {
        return 0
    }
    def f = new File(path)
    if (!f.exists()) {
        return 0
    }
    def lines = f.readLines('UTF-8')
    if (!lines || lines.size() <= 1) {
        return 0
    }
    return lines.size() - 1
}

def resolveSinglePatient(samplesheetPath, targetPatientName) {
    def path = normalizeOptionalPath(samplesheetPath)
    if (!path) {
        error("Cannot resolve target patient: invalid samplesheet path '$samplesheetPath'")
    }
    def f = new File(path)
    if (!f.exists()) {
        error("Cannot resolve target patient: samplesheet not found at '$path'")
    }
    def lines = f.readLines('UTF-8')
    if (!lines || lines.size() <= 1) {
        error("Cannot resolve target patient: samplesheet '$path' has no data rows")
    }
    def header = lines[0].split(',')
    def patientIdIdx = header.findIndexOf { it.trim() == 'patient_id' }
    if (patientIdIdx < 0) {
        error("Cannot resolve target patient: samplesheet '$path' missing 'patient_id' column")
    }
    if (targetPatientName) {
        def found = false
        for (int i = 1; i < lines.size(); i++) {
            def cols = lines[i].split(',')
            if (cols.size() > patientIdIdx && cols[patientIdIdx].trim() == targetPatientName) {
                found = true
                break
            }
        }
        if (!found) {
            error("Patient '$targetPatientName' not found in samplesheet: $path")
        }
        return targetPatientName
    }
    return lines[1].split(',')[patientIdIdx].trim()
}

def estimateRequestedModelCount(List modelsToRun) {
    def normalized = (modelsToRun ?: []).collect { it.toString().toLowerCase() }
    if (normalized.contains('all')) {
        return 3
    }
    return Math.max(1, normalized.count { it in ['cinema', 'nnformer', 'vsa3l', 'atrial_nnunet'] })
}

def etaMinutesPerUnitForMode(String modeName) {
    switch ((modeName ?: '').toUpperCase()) {
        case 'MAIN':
            return parseDoubleSafe(params.eta.minutes_per_case_main)
        case 'POSTPROCESS_ONLY':
            return parseDoubleSafe(params.eta.minutes_per_case_postprocess_only)
        case 'FEATURES_ONLY':
            return parseDoubleSafe(params.eta.minutes_per_case_features_only)
        case 'DEBUG_ONLY':
            return parseDoubleSafe(params.eta.minutes_debug_only)
        default:
            return Double.NaN
    }
}

@Field
def __etaState = [
    timer: null,
    mode: null,
    startMillis: System.currentTimeMillis(),
    expectedSeconds: null,
    stopRequested: false
]

@Field
def __etaFormatter = DateTimeFormatter.ofPattern('yyyy-MM-dd HH:mm:ss z')
    .withZone(ZoneId.systemDefault())

def formatElapsed(long totalSeconds) {
    long safe = Math.max(0L, totalSeconds)
    long hh = (long) (safe / 3600L)
    long mm = (long) ((safe % 3600L) / 60L)
    long ss = (long) (safe % 60L)
    return String.format('%02d:%02d:%02d', hh, mm, ss)
}

def parseDoubleSafe(value) {
    if (value == null) {
        return Double.NaN
    }
    def text = value.toString().trim()
    if (!text || text.equalsIgnoreCase('nan') || text.equalsIgnoreCase('na')) {
        return Double.NaN
    }
    try {
        return Double.parseDouble(text)
    } catch (Exception ignored) {
        return Double.NaN
    }
}

def inferArchitectureFromModel(String modelTag) {
    if (!modelTag) {
        return 'unknown'
    }
    if (modelTag.startsWith('cinema')) {
        return 'cinema'
    }
    if (modelTag.startsWith('nnformer')) {
        return 'nnformer'
    }
    if (modelTag.startsWith('vsa3l')) {
        return 'vsa3l'
    }
    if (modelTag.startsWith('atrial_nnunet')) {
        return 'atrial_nnunet'
    }
    return 'unknown'
}

def modelFamilyKey(String modelTag) {
    def base = (modelTag ?: '').toString()
    base = base.replaceFirst(/_ensemble$/, '')
    base = base.replaceFirst(/_seeds\d+$/, '')
    base = base.replaceFirst(/_seed\d+$/, '')
    base = base.replaceFirst(/_fold\d+$/, '')
    return base
}

def isEnsembleVariant(String modelTag) {
    def m = (modelTag ?: '').toString()
    return (m ==~ /.*_ensemble$/) || (m ==~ /.*_seeds\d+$/)
}

def parseFeatureModelTagPreference() {
    def raw = normalizeOptionalPath(params.feature_extraction.model_tag)
    if (!raw) {
        return [] as Set
    }
    return raw
        .tokenize(',')
        .collect { it.trim() }
        .findAll { it }
        .toSet()
}

def parseFeatureSeedPreference() {
    return (normalizeOptionalPath(params.feature_extraction.seed) ?: 'ensemble').toLowerCase()
}

def discoverFeatureCandidateModels(String resultsDir, List modelsToRun, String contextLabel) {
    def root = new File(resultsDir)
    if (!root.exists()) {
        log.warn "${contextLabel}: feature selection scan skipped; results_dir does not exist: ${resultsDir}"
        return [] as Set
    }

    def allowedArchitectures = (modelsToRun ?: []).collect { it.toString().toLowerCase() }
    boolean allowAll = allowedArchitectures.contains('all') || allowedArchitectures.isEmpty()

    def models = [] as Set

    root.eachFileRecurse(FileType.FILES) { f ->
        def relParts = f.toPath().normalize().toString().split('/')*.toLowerCase()
        if (relParts.contains('postprocess')) {
            return
        }

        def matcher = (f.name =~ discoverFramePattern())
        if (!matcher.matches()) {
            return
        }

        def model = matcher[0][3]
        if (!model || model.endsWith('_pp')) {
            return
        }

        def arch = inferArchitectureFromModel(model)
        if (arch == 'atrial_nnunet') {
            return
        }
        if (!allowAll && !(arch in allowedArchitectures)) {
            return
        }
        models << model
    }

    return models
}

def selectPreferredFeatureModels(String resultsDir, List modelsToRun, String contextLabel) {
    def candidates = discoverFeatureCandidateModels(resultsDir, modelsToRun, contextLabel)
    if (!candidates) {
        log.warn "${contextLabel}: no feature candidate segmentations were discovered in ${resultsDir}; no model filtering will be applied."
        return null
    }

    def requestedModelTags = parseFeatureModelTagPreference()
    if (requestedModelTags) {
        def matched = candidates.findAll { requestedModelTags.contains(it.toString()) } as Set
        def missing = requestedModelTags.findAll { !matched.contains(it) }
        if (missing) {
            log.warn "${contextLabel}: requested --feature_extraction.model_tag entries were not found: ${missing.join(', ')}"
        }
        if (matched) {
            log.info "${contextLabel}: using explicit feature model_tag filter: ${matched.sort().join(', ')}"
            return matched
        }

        exit 1, "ERROR: ${contextLabel}: none of the requested --feature_extraction.model_tag entries were discovered under ${resultsDir}."
    }

    def seedPreference = parseFeatureSeedPreference()
    def selected = [] as Set

    def grouped = candidates.groupBy { modelFamilyKey(it as String) }
    grouped.each { family, familyModels ->
        def models = familyModels.collect { it.toString() }.unique().sort()
        def ensembles = models.findAll { isEnsembleVariant(it) }

        if (seedPreference in ['all', '*']) {
            selected.addAll(models)
            return
        }

        if (seedPreference in ['ensemble', 'default']) {
            if (ensembles) {
                selected << ensembles[0]
                return
            }

            if (models.size() == 1) {
                selected << models[0]
                return
            }

            def fallback = models.find { it ==~ /.*_seed0$/ } ?: models[0]
            log.warn "${contextLabel}: no ensemble found for ${family}; defaulting to ${fallback}."
            selected << fallback
            return
        }

        def seedMatcher = (seedPreference =~ /^seed?(\d+)$/)
        if (seedMatcher.matches()) {
            def seedNum = seedMatcher[0][1]
            def seedMatches = models.findAll { it ==~ /.*_seed${seedNum}$/ }
            if (seedMatches) {
                selected.addAll(seedMatches)
                return
            }

            if (models.size() == 1) {
                selected << models[0]
                return
            }

            exit 1, "ERROR: ${contextLabel}: requested seed '${seedPreference}' for ${family} was not found. Available models: ${models.join(', ')}"
        }

        exit 1, "ERROR: ${contextLabel}: invalid --feature_extraction.seed '${seedPreference}'. Valid examples: ensemble, all, seed0, 0"
    }

    log.info "${contextLabel}: selected ${selected.size()} feature models from ${candidates.size()} discovered candidates (seed preference: ${seedPreference})."
    return selected
}

def resolveFeatureOutputDir(String resultsDir, String selectedMaskSource) {
    def leaf = (selectedMaskSource == 'postprocess') ? 'postprocessed' : 'raw'
    return file("${resultsDir}/features/${leaf}").toAbsolutePath().toString()
}

def loadEtaHistory() {
    def etaFile = file("${projectDir}/.cache/eta_history.json").toFile()
    if (!etaFile.exists()) {
        return [:]
    }
    try {
        def parsed = new JsonSlurper().parseText(etaFile.getText('UTF-8'))
        return (parsed instanceof Map) ? parsed : [:]
    } catch (Exception ignored) {
        return [:]
    }
}

def saveEtaHistory(Map history) {
    def etaFile = file("${projectDir}/.cache/eta_history.json").toFile()
    etaFile.parentFile?.mkdirs()
    etaFile.text = JsonOutput.prettyPrint(JsonOutput.toJson(history)) + '\n'
}

def startEtaTimer(String modeName, int estimatedUnits = 1) {
    if (__etaState.timer != null) {
        return
    }

    __etaState.mode = modeName
    __etaState.startMillis = System.currentTimeMillis()
    __etaState.stopRequested = false

    def history = loadEtaHistory()
    def etaSec = parseDoubleSafe(history[modeName])
    __etaState.expectedSeconds = (!Double.isNaN(etaSec) && etaSec > 0) ? Math.round(etaSec) : null

    if (__etaState.expectedSeconds == null) {
        def minutesPerUnit = etaMinutesPerUnitForMode(modeName)
        if (!Double.isNaN(minutesPerUnit) && minutesPerUnit > 0) {
            __etaState.expectedSeconds = Math.round(minutesPerUnit * Math.max(1, estimatedUnits) * 60.0d)
        }
    }

    if (__etaState.expectedSeconds != null) {
        def etaInstant = Instant.ofEpochMilli(__etaState.startMillis + (__etaState.expectedSeconds * 1000L))
        log.info "ETA [${modeName}] baseline: ${formatElapsed(__etaState.expectedSeconds as long)} (expected completion around ${__etaFormatter.format(etaInstant)})"
    } else {
        log.info "ETA [${modeName}] baseline unavailable (no historical run yet)."
    }

    __etaState.timer = Thread.startDaemon("sorat-eta-${modeName}") {
        while (!(__etaState.stopRequested as boolean)) {
            try {
                Thread.sleep(60_000L)
            } catch (InterruptedException ignored) {
                break
            }

            if (__etaState.stopRequested as boolean) {
                break
            }

            long elapsed = Math.round((System.currentTimeMillis() - (__etaState.startMillis as long)) / 1000.0d)
            if (__etaState.expectedSeconds == null) {
                log.info "ETA [${modeName}] elapsed ${formatElapsed(elapsed)}"
                continue
            }

            long remaining = (__etaState.expectedSeconds as long) - elapsed
            long remainingSafe = Math.max(0L, remaining)
            def etaInstant = Instant.ofEpochMilli(System.currentTimeMillis() + (remainingSafe * 1000L))
            log.info "ETA [${modeName}] elapsed ${formatElapsed(elapsed)} | remaining ${formatElapsed(remainingSafe)} | eta ${__etaFormatter.format(etaInstant)}"
        }
    }
}

def finalizeEtaTimer(boolean success) {
    if (__etaState.timer != null) {
        __etaState.stopRequested = true
        try {
            __etaState.timer.interrupt()
        } catch (Exception ignored) {
            // Best-effort shutdown for daemon ETA thread.
        }
        __etaState.timer = null
    }

    def modeName = (__etaState.mode ?: 'MAIN').toString()
    long elapsed = Math.round((System.currentTimeMillis() - (__etaState.startMillis as long)) / 1000.0d)

    if (success) {
        def history = loadEtaHistory()
        def prev = parseDoubleSafe(history[modeName])
        def updated = Double.isNaN(prev) ? (double) elapsed : ((0.7d * prev) + (0.3d * elapsed))
        history[modeName] = updated
        saveEtaHistory(history)
    }
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

def resolveMbasSamplesheet(atrialDefault, atrialRoot, contextLabel) {
    def configuredAtrial = normalizeOptionalPath(atrialDefault)
    if (configuredAtrial) {
        return file(configuredAtrial).toAbsolutePath().toString()
    }

    def datasetRootPath = normalizeOptionalPath(atrialRoot)
    if (!datasetRootPath) {
        return null
    }

    def datasetRoot = new File(datasetRootPath)
    def generatedDir = file("${projectDir}/.cache/generated_inputs")
    generatedDir.mkdirs()
    def generatedCsv = new File(generatedDir.toString(), 'atrial_default_samplesheet.csv')

    def imagesDir = new File(datasetRoot.toString(), 'imagesTr')
    def labelsDir = new File(datasetRoot.toString(), 'labelsTr')

    def rowCount = 0

    generatedCsv.withWriter('UTF-8') { writer ->
        writer.writeLine('patient_id,image,ground_truth,info_cfg')

        if (imagesDir.exists() && labelsDir.exists()) {
            def imageFiles = imagesDir.listFiles()?.findAll { it.name.endsWith('_0000.nii.gz') }?.sort { it.name }
            if (!imageFiles) {
                exit 1, "ERROR: ${contextLabel}: No atrial default inputs found under ${imagesDir}"
            }

            imageFiles.each { imageFile ->
                def patientId = imageFile.name.replace('_0000.nii.gz', '')
                def gtFile = new File(labelsDir, "${patientId}.nii.gz")
                if (!gtFile.exists()) {
                    log.warn "Skipping atrial sample ${patientId}: missing label file ${gtFile}"
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
            exit 1, "ERROR: ${contextLabel}: Atrial data root must be either nnUNet layout (imagesTr/labelsTr) or MBAS-style training layout (MBAS_###/). Got: ${datasetRoot}"
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
                log.warn "Skipping atrial sample ${patientId}: expected image/label pair not found in ${caseDir}"
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
        exit 1, "ERROR: ${contextLabel}: Generated atrial default samplesheet has no valid rows. Check files under ${datasetRoot}."
    }

    return generatedCsv.toString()
}

def resolveAcdcSamplesheet(saxDefault, saxRoot, saxSplit, contextLabel) {
    def configuredSax = normalizeOptionalPath(saxDefault)
    if (configuredSax) {
        return file(configuredSax).toAbsolutePath().toString()
    }

    def datasetRootPath = normalizeOptionalPath(saxRoot)
    if (!datasetRootPath) {
        return null
    }

    def datasetName = normalizeOptionalPath(saxSplit) ?: 'testing'
    def dataDir = new File(datasetRootPath, datasetName)
    if (!dataDir.exists()) {
        exit 1, "ERROR: ${contextLabel}: SAX dataset directory not found: ${dataDir}. Set --sax_data_root (or legacy --acdc_dir) and optionally --sax_data_split."
    }

    def patientDirs = dataDir.listFiles()?.findAll {
        it.isDirectory() && it.name.startsWith('patient')
    }?.sort { it.name }

    if (!patientDirs) {
        exit 1, "ERROR: ${contextLabel}: No patient folders found under ${dataDir}."
    }

    def generatedDir = file("${projectDir}/.cache/generated_inputs")
    generatedDir.mkdirs()
    def generatedCsv = new File(generatedDir.toString(), "sax_${datasetName}_samplesheet.csv")

    def rowCount = 0
    generatedCsv.withWriter('UTF-8') { writer ->
        writer.writeLine('patient_id,image,ground_truth,info_cfg')

        patientDirs.each { patientDir ->
            def patientId = patientDir.name
            def image4d = new File(patientDir, "${patientId}_4d.nii.gz")
            if (!image4d.exists()) {
                log.warn "Skipping SAX sample ${patientId}: missing 4D image ${image4d}"
                return
            }

            def infoCfg = new File(patientDir, 'Info.cfg')
            def infoPath = infoCfg.exists() ? infoCfg.absolutePath : ''
            writer.writeLine("${patientId},${image4d.absolutePath},${patientDir.absolutePath},${infoPath}")
            rowCount++
        }
    }

    if (rowCount == 0) {
        exit 1, "ERROR: ${contextLabel}: Generated SAX default samplesheet has no valid rows under ${dataDir}."
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
        exit 1, "ERROR: ${contextLabel}: SLURM profile is active but no account is configured. Set --slurm_account, SORAT_SLURM_ACCOUNT, or .sorat/user.config."
    }
}

def resolveEffectiveSamplesheet(inputParam, modelsToRun, saxDefault, saxRoot, saxSplit, atrialDefault, atrialRoot, contextLabel = 'main workflow') {
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
        def atrialInput = resolveMbasSamplesheet(atrialDefault, atrialRoot, contextLabel)
        if (!atrialInput) {
            exit 1, "ERROR: ${contextLabel}: atrial_nnunet requested without --input, but no atrial default is configured. Set --default_inputs.atrial (or --default_inputs.mbas), SORAT_ATRIAL_SAMPLESHEET (or SORAT_MBAS_SAMPLESHEET), or --atrial_nnunet.dataset_root (or --atrial_nnunet.mbas_root)."
        }
        return file(atrialInput).toAbsolutePath().toString()
    }

    def saxInput = resolveAcdcSamplesheet(saxDefault, saxRoot, saxSplit, contextLabel)
    if (!saxInput) {
        exit 1, "ERROR: ${contextLabel}: No SAX default input is configured. Set --default_inputs.sax (or --default_inputs.acdc), --sax_data_root (or --acdc_dir), or pass --input."
    }
    return file(saxInput).toAbsolutePath().toString()
}

def resolveOptionalSamplesheet(inputParam, modelsToRun, saxDefault, saxRoot, saxSplit, atrialDefault, atrialRoot, contextLabel = 'debug workflow') {
    def explicitInput = normalizeOptionalPath(inputParam)
    if (explicitInput) {
        return file(explicitInput).toAbsolutePath().toString()
    }

    def hasAtrial = 'atrial_nnunet' in modelsToRun
    def hasSax = ('all' in modelsToRun) || ['cinema', 'nnformer', 'vsa3l'].any { it in modelsToRun }

    if (hasAtrial && hasSax) {
        return null
    }

    if (hasAtrial) {
        def atrialInput = resolveMbasSamplesheet(atrialDefault, atrialRoot, contextLabel)
        return atrialInput ? file(atrialInput).toAbsolutePath().toString() : null
    }

    def saxInput = resolveAcdcSamplesheet(saxDefault, saxRoot, saxSplit, contextLabel)
    return saxInput ? file(saxInput).toAbsolutePath().toString() : null
}

def resolveFeatureMaskSource(contextLabel = 'main workflow') {
    def configured = normalizeOptionalPath(params.feature_extraction.mask_source) ?: 'predictions'
    def source = configured.toLowerCase()

    if (!(source in ['predictions', 'postprocess', 'prompt'])) {
        exit 1, "ERROR: ${contextLabel}: Invalid --feature_extraction.mask_source '${configured}'. Valid values: predictions, postprocess, prompt."
    }

    if (source != 'prompt') {
        return source
    }

    def console = System.console()
    if (console == null) {
        log.warn "Feature extraction mask source is set to 'prompt' but no interactive console is available. Falling back to 'predictions'."
        return 'predictions'
    }

    console.printf("\nFeature extraction mask source:\n")
    console.printf("  1) predictions (direct model output)\n")
    console.printf("  2) postprocess (post-processed masks)\n")
    def choice = (console.readLine("Select mask source [1/2] (default: 1): ") ?: '').trim().toLowerCase()

    if (choice in ['2', 'postprocess']) {
        return 'postprocess'
    }
    return 'predictions'
}

/*
========================================================================================
    MAIN WORKFLOW
========================================================================================
*/

workflow {

    assertSlurmAccountForProfile('main workflow')

    def models_to_run = parseModels(params.models)
    def labelSchema = (params.evaluation?.label_schema ?: 'architecture_default').toString().trim()
    if (!(labelSchema in ['architecture_default', 'atrial_binary_union'])) {
        exit 1, "ERROR: main workflow: invalid evaluation.label_schema '${labelSchema}'. Valid values: architecture_default, atrial_binary_union."
    }
    if (labelSchema == 'atrial_binary_union' && !models_to_run.contains('atrial_nnunet')) {
        exit 1, "ERROR: evaluation.label_schema=atrial_binary_union requires atrial_nnunet in --models."
    }
    log.info "Evaluation label schema: ${labelSchema}"
    def effective_input_samplesheet = resolveEffectiveSamplesheet(
        params.input,
        models_to_run,
        normalizeOptionalPath(params.default_inputs.sax) ?: normalizeOptionalPath(params.default_inputs.acdc),
        normalizeOptionalPath(params.sax_data_root) ?: normalizeOptionalPath(params.acdc_dir),
        normalizeOptionalPath(params.sax_data_split) ?: normalizeOptionalPath(params.acdc_dataset),
        normalizeOptionalPath(params.default_inputs.atrial) ?: normalizeOptionalPath(params.default_inputs.mbas),
        normalizeOptionalPath(params.atrial_nnunet.dataset_root) ?: normalizeOptionalPath(params.atrial_nnunet.mbas_root),
        'main workflow'
    )

    if (params.eta.enabled as boolean) {
        def etaUnits = Math.max(1, countSamplesheetRows(effective_input_samplesheet) * estimateRequestedModelCount(models_to_run))
        startEtaTimer('MAIN', etaUnits)
    }

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

    def singlePatient = params.single_patient as boolean
    def providedPatientName = params.patient_name ? params.patient_name.toString().trim() : null
    if (providedPatientName) { singlePatient = true }
    if (singlePatient) {
        def targetPatient = resolveSinglePatient(effective_input_samplesheet, providedPatientName)
        log.info "Single-patient mode: running patient '$targetPatient' only"
        ch_input = ch_input
            .filter { patient_id, image_file, gt_file, info_file ->
                patient_id == targetPatient
            }
    }

    def auto_discover = params.auto_discover_models as boolean
    
    // Initialize result channels
    ch_cinema_results = Channel.empty()
    ch_nnformer_results = Channel.empty()
    ch_vsa3l_results = Channel.empty()
    ch_atrial_nnunet_results = Channel.empty()

    // Deterministic cache keys for model-specific preprocessing outputs.
    // Bumped to _v4 to invalidate _v3 caches, which wrote CineMA segmentations
    // in the 192x192 preprocessed coordinate space; _v4 records the original
    // image geometry in metadata.json so segmentations are mapped back into
    // the original space (matching the ground truth + other architectures).
    // VSA-3L _v4 likewise records the original origin / direction so its
    // segmentations are physically aligned with the original image (was
    // origin 0 + identity, which broke feature extraction on oblique / offset
    // acquisitions such as M&Ms).
    def cinema_preprocess_key = 'cinema_s1.0x1.0x10.0_crop192x192_v4'
    def nnformer_preprocess_key = 'nnformer_frame_extract_v3'
    def vsa3l_preprocess_key = "vsa3l_input${params.vsa3l.input_size.join('x')}_v4"
    def atrial_nnunet_preprocess_key = 'atrial_nnunet_frame_extract_v3'
    
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
            .flatMap { patient_id, seg_list, manifest, meta ->
                def frames = parseManifest(manifest)
                def segs = normalizeSegList(seg_list)
                segs.collect { seg ->
                    def tag = frameTagFromFilename(seg)
                    def idx = frames.find { it.tag == tag }?.idx ?: 0
                    [ patient_id, meta.model_tag, tag, idx, seg, meta ]
                }
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
            .flatMap { patient_id, seg_list, manifest, meta ->
                def frames = parseManifest(manifest)
                def segs = normalizeSegList(seg_list)
                segs.collect { seg ->
                    def tag = frameTagFromFilename(seg)
                    def idx = frames.find { it.tag == tag }?.idx ?: 0
                    [ patient_id, meta.model_tag, tag, idx, seg, meta ]
                }
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
            .flatMap { patient_id, seg_list, manifest, meta ->
                def frames = parseManifest(manifest)
                def segs = normalizeSegList(seg_list)
                segs.collect { seg ->
                    def tag = frameTagFromFilename(seg)
                    def idx = frames.find { it.tag == tag }?.idx ?: 0
                    [ patient_id, meta.model_tag, tag, idx, seg, meta ]
                }
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
            .flatMap { patient_id, seg_list, manifest, meta ->
                def frames = parseManifest(manifest)
                def segs = normalizeSegList(seg_list)
                segs.collect { seg ->
                    def tag = frameTagFromFilename(seg)
                    def idx = frames.find { it.tag == tag }?.idx ?: 0
                    [ patient_id, meta.model_tag, tag, idx, seg, meta ]
                }
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
            .map { patient_id, model, frame_tag, frame_idx, seg, meta, image_path, gt_path, info_cfg ->
                def architecture = meta?.architecture ?: 'unknown'
                // CineMA segmentations are now written in the original image
                // coordinate space (like nnFormer / VSA-3L), so the original
                // image and ground-truth paths align with the seg directly.
                [ patient_id, model, frame_tag, frame_idx, seg, image_path, gt_path, info_cfg, architecture ]
            }
            .filter { patient_id, model, frame_tag, frame_idx, seg, image_path, gt_path, info_cfg, architecture ->
                // Always keep ED/ES frames (back-compat with ACDC)
                if (frame_tag in ['ED', 'ES']) { return true }
                // In all-frames mode: keep only frame00 by default, or all frames when flag is set
                if (params.visualization.all_frames) { return true }
                return frame_tag == 'frame00'
            }

        GENERATE_SEGMENTATION_PREVIEWS(ch_preview_inputs)
    }

    ch_segmentations_for_metrics = ch_all_segmentations

    if (params.postprocess.enabled) {
        ch_postprocess_candidates = ch_all_segmentations
            .filter { patient_id, model, frame_tag, frame_idx, seg, meta ->
                meta?.architecture != 'atrial_nnunet'
            }

        ch_atrial_segmentations = ch_all_segmentations
            .filter { patient_id, model, frame_tag, frame_idx, seg, meta ->
                meta?.architecture == 'atrial_nnunet'
            }

        ch_postprocess_inputs = ch_postprocess_candidates
            .combine(ch_input_context, by: 0)
            .map { patient_id, model, frame_tag, frame_idx, seg, meta, image_path, gt_path, info_cfg ->
                // CineMA segmentations are now in the original image space, so
                // the original image aligns with the seg directly (no redirect
                // to the 192x192 preprocessed volume needed).
                [ patient_id, model, frame_tag, frame_idx, seg, meta, image_path, info_cfg ]
            }

        POSTPROCESS_LV_MYO(ch_postprocess_inputs)

        if (params.postprocess.use_for_metrics) {
            ch_segmentations_for_metrics = POSTPROCESS_LV_MYO.out.segmentations
                .map { patient_id, model, frame_tag, frame_idx, seg, meta_pp, image_path, info_cfg ->
                    [ patient_id, model, frame_tag, frame_idx, seg, meta_pp ]
                }
                .mix(ch_atrial_segmentations)
        }

        VISUALIZE_POSTPROCESS_DELTA(POSTPROCESS_LV_MYO.out.before_after)
    }

    if (params.feature_extraction.enabled) {
        def selectedMaskSource = resolveFeatureMaskSource('main workflow')
        def effectiveMaskSource = selectedMaskSource
        def featureResultsDir = file(params.feature_extraction.results_dir ?: params.outdir).toAbsolutePath().toString()
        def selectedModels = selectPreferredFeatureModels(featureResultsDir, models_to_run, 'main workflow')
        def ch_features_source = ch_all_segmentations

        if (selectedMaskSource == 'postprocess' && params.postprocess.enabled) {
            ch_features_source = POSTPROCESS_LV_MYO.out.segmentations
                .map { patient_id, model, frame_tag, frame_idx, seg, meta_pp, image_path, info_cfg ->
                    [ patient_id, model, frame_tag, frame_idx, seg, meta_pp ]
                }
        } else if (selectedMaskSource == 'postprocess' && !params.postprocess.enabled) {
            log.warn "Feature extraction requested postprocess masks, but postprocessing is disabled. Falling back to predictions."
            effectiveMaskSource = 'predictions'
        }

        params.feature_extraction.output_dir = normalizeOptionalPath(params.feature_extraction.output_dir) ?: resolveFeatureOutputDir(featureResultsDir, effectiveMaskSource)

        log.info "main workflow: feature output directory = ${params.feature_extraction.output_dir}"
        if (selectedModels) {
            log.info "main workflow: feature extraction model filter = ${selectedModels.sort().join(', ')}"
        }

        def ch_features_inputs = ch_features_source
            .filter { patient_id, model, frame_tag, frame_idx, seg, meta ->
                meta?.architecture != 'atrial_nnunet'
            }
            .filter { patient_id, model, frame_tag, frame_idx, seg, meta ->
                selectedModels == null || selectedModels.contains(model.toString())
            }
            .combine(ch_input_context, by: 0)
            .map { patient_id, model, frame_tag, frame_idx, seg, meta, image_path, gt_path, info_cfg ->
                def image_file = file(image_path, checkIfExists: true)
                def safe_model = model.toString().replaceAll('[^A-Za-z0-9_.-]', '_')
                [ "${patient_id}_${safe_model}_${frame_tag}", frame_tag, frame_idx, image_file, seg, info_cfg ?: '' ]
            }

        EXTRACT_FEATURES(ch_features_inputs)
    }

    if (!params.inference_only) {
        // Compute metrics if ground truth is available
        ch_input_with_gt = ch_input
            .filter { patient_id, image, gt, info -> gt != null }
            .map { patient_id, image, gt, info -> [ patient_id, gt ] }

        ch_for_metrics = ch_segmentations_for_metrics
            .combine(ch_input_with_gt, by: 0)
            .map { patient_id, model, frame_tag, frame_idx, seg, meta, gt ->
                // CineMA segmentations are now in the original image coordinate
                // space, so the original ground truth aligns directly (no redirect
                // to the 192x192 preprocessed GT needed) -- same as nnFormer /
                // VSA-3L.
                [ patient_id, model, frame_tag, frame_idx, seg, gt, meta ]
            }

        COMPUTE_METRICS(ch_for_metrics)

        // Aggregate results across all models and patients
        if (params.compare) {
            ch_all_metrics = COMPUTE_METRICS.out.metrics
                .map { patient_id, model, metrics_csv -> metrics_csv }
                .collect()
            AGGREGATE_RESULTS(ch_all_metrics)
            ch_all_seg_files = ch_segmentations_for_metrics
                .map { patient_id, model, frame_tag, frame_idx, seg, meta -> seg }
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
    } else {
        log.info "Inference-only mode: skipping metrics, aggregation, comparison report, and debug report."
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
            normalizeOptionalPath(params.default_inputs.sax) ?: normalizeOptionalPath(params.default_inputs.acdc),
            normalizeOptionalPath(params.sax_data_root) ?: normalizeOptionalPath(params.acdc_dir),
            normalizeOptionalPath(params.sax_data_split) ?: normalizeOptionalPath(params.acdc_dataset),
            normalizeOptionalPath(params.default_inputs.atrial) ?: normalizeOptionalPath(params.default_inputs.mbas),
            normalizeOptionalPath(params.atrial_nnunet.dataset_root) ?: normalizeOptionalPath(params.atrial_nnunet.mbas_root),
            'POSTPROCESS_ONLY'
        )
    def results_dir = file(params.postprocess.results_dir ?: params.outdir).toAbsolutePath().toString()

    if (params.eta.enabled as boolean) {
        def etaUnits = Math.max(1, countSamplesheetRows(samplesheet_path) * estimateRequestedModelCount(models_to_run))
        startEtaTimer('POSTPROCESS_ONLY', etaUnits)
    }

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
            // CineMA segmentations are now in the original image space, so the
            // samplesheet image aligns with the seg directly (no redirect to the
            // 192x192 preprocessed volume needed).
            def resolvedImage = row.image
            def meta = [architecture: row.architecture ?: 'unknown', model_tag: row.model, discovered: true]
            def frameTag = row.frame_tag ?: 'ED'
            def frameIdx = row.frame_idx ? row.frame_idx as int : 0
            [
                row.patient_id,
                row.model,
                frameTag,
                frameIdx,
                file(row.seg, checkIfExists: true),
                meta,
                resolvedImage,
                row.info_cfg ?: ''
            ]
        }

    POSTPROCESS_LV_MYO(ch_discovered)

    VISUALIZE_POSTPROCESS_DELTA(POSTPROCESS_LV_MYO.out.before_after)
}

workflow FEATURES_ONLY {
    assertSlurmAccountForProfile('FEATURES_ONLY')

    def feature_samplesheet = normalizeOptionalPath(params.feature_extraction.samplesheet)
    def models_to_run = parseModels(params.models)
    def samplesheet_path = feature_samplesheet
        ? file(feature_samplesheet).toAbsolutePath().toString()
        : resolveEffectiveSamplesheet(
            params.input,
            models_to_run,
            normalizeOptionalPath(params.default_inputs.sax) ?: normalizeOptionalPath(params.default_inputs.acdc),
            normalizeOptionalPath(params.sax_data_root) ?: normalizeOptionalPath(params.acdc_dir),
            normalizeOptionalPath(params.sax_data_split) ?: normalizeOptionalPath(params.acdc_dataset),
            normalizeOptionalPath(params.default_inputs.atrial) ?: normalizeOptionalPath(params.default_inputs.mbas),
            normalizeOptionalPath(params.atrial_nnunet.dataset_root) ?: normalizeOptionalPath(params.atrial_nnunet.mbas_root),
            'FEATURES_ONLY'
        )

    def results_dir = file(params.feature_extraction.results_dir ?: params.outdir).toAbsolutePath().toString()
    def selectedMaskSource = resolveFeatureMaskSource('FEATURES_ONLY')
    def selectedModels = selectPreferredFeatureModels(results_dir, models_to_run, 'FEATURES_ONLY')
    params.feature_extraction.output_dir = normalizeOptionalPath(params.feature_extraction.output_dir) ?: resolveFeatureOutputDir(results_dir, selectedMaskSource)

    if (params.eta.enabled as boolean) {
        def etaUnits = Math.max(1, countSamplesheetRows(samplesheet_path) * estimateRequestedModelCount(models_to_run))
        startEtaTimer('FEATURES_ONLY', etaUnits)
    }

    log.info "FEATURES_ONLY mode: reading segmentations from ${results_dir}"
    log.info "FEATURES_ONLY mode: outputting features to ${params.feature_extraction.output_dir}"
    log.info "FEATURES_ONLY mode: selected mask source = ${selectedMaskSource}"
    if (selectedModels) {
        log.info "FEATURES_ONLY mode: feature extraction model filter = ${selectedModels.sort().join(', ')}"
    }

    ch_samplesheet = Channel.fromPath(samplesheet_path, checkIfExists: true)

    DISCOVER_POSTPROCESS_INPUTS(
        ch_samplesheet,
        results_dir,
        params.models
    )

    ch_features_inputs = DISCOVER_POSTPROCESS_INPUTS.out.inputs
        .splitCsv(header: true)
        .map { row ->
            def frameTag = row.frame_tag ?: 'ED'
            def frameIdx = row.frame_idx ? row.frame_idx as int : 0
            def segPath = row.seg

            if (selectedMaskSource == 'postprocess') {
                def ppSeg = file("${results_dir}/postprocess/${row.model}/segmentations/${row.patient_id}_${frameTag}_${row.model}_pp.nii.gz")
                if (!ppSeg.exists()) {
                    log.warn "Skipping ${row.patient_id} / ${row.model} / ${frameTag}: postprocess mask not found in ${results_dir}/postprocess/${row.model}/segmentations"
                    return null
                }
                segPath = ppSeg.toAbsolutePath().toString()
            }

            [
                row.patient_id,
                row.model,
                (row.architecture ?: 'unknown').toLowerCase(),
                frameTag,
                frameIdx,
                segPath,
                row.image,
                row.info_cfg ?: ''
            ]
        }
        .filter { it != null }
        .filter { patient_id, model, architecture, frame_tag, frame_idx, seg, image, info_cfg ->
            architecture != 'atrial_nnunet'
        }
        .filter { patient_id, model, architecture, frame_tag, frame_idx, seg, image, info_cfg ->
            selectedModels == null || selectedModels.contains(model.toString())
        }
        .map { patient_id, model, architecture, frame_tag, frame_idx, seg, image, info_cfg ->
            def imageFile = file(image, checkIfExists: true)
            def segFile = file(seg, checkIfExists: true)
            def safeModel = model.toString().replaceAll('[^A-Za-z0-9_.-]', '_')
            [ "${patient_id}_${safeModel}_${frame_tag}", frame_tag, frame_idx, imageFile, segFile, info_cfg ?: '' ]
        }

    EXTRACT_FEATURES(ch_features_inputs)
}

workflow DEBUG_ONLY {
    assertSlurmAccountForProfile('DEBUG_ONLY')
    if (params.eta.enabled as boolean) {
        startEtaTimer('DEBUG_ONLY', 1)
    }

    def models_to_run = parseModels(params.models)
    def debug_source_outdir = normalizeOptionalPath(params.debug_source_outdir)
    if (!debug_source_outdir) {
        exit 1, "ERROR: DEBUG_ONLY requires --debug_source_outdir pointing to an existing completed SORAT results directory. Use --outdir for the refresh run output location."
    }

    def refresh_outdir = file(params.outdir).toAbsolutePath().toString()
    def source_outdir = file(debug_source_outdir).toAbsolutePath().toString()
    if (refresh_outdir == source_outdir) {
        exit 1, "ERROR: DEBUG_ONLY requires --outdir to differ from --debug_source_outdir to avoid overwriting the source run pipeline_info/debug artifacts."
    }

    def debug_samplesheet = resolveOptionalSamplesheet(
        params.input,
        models_to_run,
        normalizeOptionalPath(params.default_inputs.sax) ?: normalizeOptionalPath(params.default_inputs.acdc),
        normalizeOptionalPath(params.sax_data_root) ?: normalizeOptionalPath(params.acdc_dir),
        normalizeOptionalPath(params.sax_data_split) ?: normalizeOptionalPath(params.acdc_dataset),
        normalizeOptionalPath(params.default_inputs.atrial) ?: normalizeOptionalPath(params.default_inputs.mbas),
        normalizeOptionalPath(params.atrial_nnunet.dataset_root) ?: normalizeOptionalPath(params.atrial_nnunet.mbas_root),
        'DEBUG_ONLY'
    )

    def debug_input_samplesheet = debug_samplesheet ?: file("${projectDir}/.sorat/missing_debug_samplesheet.csv").toAbsolutePath().toString()
    def debug_run_name = normalizeOptionalPath(params.debug_run_name) ?: "${workflow.runName}_debug_only"
    def debug_workflow_duration = normalizeOptionalPath(params.debug_workflow_duration) ?: ''
    def debug_workflow_start = normalizeOptionalPath(params.debug_workflow_start) ?: ''
    def debug_workflow_success = normalizeOptionalPath(params.debug_workflow_success) ?: 'true'

    log.info "DEBUG_ONLY mode: reading source results from ${source_outdir}"
    log.info "DEBUG_ONLY mode: writing refresh workflow artifacts to ${refresh_outdir}"
    if (!debug_samplesheet) {
        log.warn "DEBUG_ONLY mode: no input samplesheet was resolved; patient coverage will be estimated from existing metrics only."
    }

    GENERATE_DEBUG_REPORT(
        Channel.value('debug_only'),
        source_outdir,
        debug_input_samplesheet,
        params.models,
        debug_run_name,
        debug_workflow_duration,
        debug_workflow_start,
        debug_workflow_success
    )
}

/*
========================================================================================
    WORKFLOW COMPLETION
========================================================================================
*/

workflow.onComplete {
    if (params.eta.enabled as boolean) {
        try {
            finalizeEtaTimer(workflow.success)
        } catch (Exception ignored) {
            // Keep completion reporting resilient even if ETA state is unavailable.
        }
    }

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
