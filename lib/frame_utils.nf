/*
========================================================================================
    Frame Utilities
========================================================================================
    Shared Groovy helpers for the unified frame-list pipeline topology.
    Imported by main.nf and used in .flatMap / .filter operators.
---------------------------------------------------------------------------------------
*/

import groovy.json.JsonSlurper

/**
 * Parse a frame-manifest JSON file produced by build_frame_manifest().
 *
 * Returns the list of frame objects::
 *     [[tag:"ED", idx:0], [tag:"ES", idx:12]]
 *
 * Returns an empty list when the manifest is missing or unreadable.
 */
def parseManifest(manifest_file) {
    if (!manifest_file) {
        return []
    }
    def f = null
    try {
        if (manifest_file instanceof File) {
            f = manifest_file
        } else if (manifest_file instanceof java.nio.file.Path) {
            f = manifest_file.toFile()
        } else {
            f = file(manifest_file as String)
        }
    } catch (Exception e) {
        return []
    }
    if (!f || !f.exists()) {
        return []
    }
    try {
        def json = new JsonSlurper().parseText(f.text)
        return (json.frames ?: []) as List
    } catch (Exception e) {
        return []
    }
}

/**
 * Normalize a segmentation output emitted by a process glob into a List.
 *
 * Nextflow's ``path("${prefix}_*_${tag}.nii.gz")`` output returns a *single*
 * ``Path`` when exactly one file matches the glob, and a ``List<Path>`` only
 * when two or more files match.  Calling ``.collect`` on a bare ``Path``
 * iterates over its *name elements* (path components) instead of a one-element
 * list, which silently fans out one bogus task per path component.
 *
 * This helper guarantees a List so downstream ``.collect`` / ``.flatMap``
 * operators always iterate over file objects.
 */
def normalizeSegList(segList) {
    if (segList == null) {
        return []
    }
    if (segList instanceof List) {
        return segList
    }
    return [segList]
}

/**
 * Extract the frame tag ("ED", "ES", or "frameNN") from a segmentation filename.
 *
 * Expects a name like::
 *     patient101_ED_cinema__acdc_seed0.nii.gz
 *     A1K2P5_frame03_nnformer__acdc_ensemble.nii.gz
 */
def frameTagFromFilename(filename) {
    def name = null
    if (filename instanceof File) {
        name = filename.name
    } else if (filename instanceof java.nio.file.Path) {
        name = filename.getFileName().toString()
    } else {
        name = filename.toString()
    }
    def matcher = (name =~ /_(ED|ES|frame\d{2,})_/)
    if (matcher) {
        return matcher[0][1]
    }
    return null
}

/**
 * Return the unified frame-discovery regex.
 *
 * Captures three groups::
 *     1 – patient_id prefix
 *     2 – frame tag  ("ED" | "ES" | "frameNN")
 *     3 – model tag suffix
 */
def discoverFramePattern() {
    return ~/^(.*)_(ED|ES|frame\d{2,})_(.+)\.nii\.gz$/
}

/**
 * Infer an architecture family from a model tag, matching the heuristic
 * used in main.nf's inferArchitectureFromModel().
 */
def inferArchitecture(modelTag) {
    def tag = modelTag.toString().toLowerCase()
    if (tag.startsWith('cinema'))                { return 'cinema' }
    if (tag.startsWith('nnformer'))              { return 'nnformer' }
    if (tag.startsWith('nnunet'))                { return 'nnunet' }
    if (tag.startsWith('atrial_nnunet'))         { return 'atrial_nnunet' }
    if (tag.startsWith('vsa3l') || tag.startsWith('ventricular_short_axis_3label')) { return 'vsa3l' }
    if (tag.contains('atrial_nnunet'))           { return 'atrial_nnunet' }
    if (tag.contains('cinema'))                  { return 'cinema' }
    if (tag.contains('nnformer') || tag.contains('nnunet')) { return 'nnformer' }
    return 'unknown'
}
