/*
========================================================================================
    SORAT Utility Functions
========================================================================================
    Common utility functions for the pipeline
----------------------------------------------------------------------------------------
*/

import groovy.json.JsonSlurper

/**
 * Validate input samplesheet format and content
 * @param samplesheet Path to the input CSV file
 * @return Boolean indicating validation success
 */
def validateInput(samplesheet) {
    def required_columns = ['patient_id', 'image']
    def file_content = file(samplesheet).text
    def lines = file_content.split('\n')
    
    if (lines.size() < 2) {
        exit 1, "ERROR: Samplesheet appears to be empty or has no data rows"
    }
    
    def header = lines[0].split(',').collect { it.trim() }
    
    required_columns.each { col ->
        if (!(col in header)) {
            exit 1, "ERROR: Required column '${col}' not found in samplesheet header"
        }
    }
    
    log.info "Samplesheet validation passed: ${lines.size() - 1} samples found"
    return true
}

/**
 * Load and parse the model registry JSON file
 */
def loadModelRegistry(registry_path) {
    def json_file = file(registry_path)
    if (!json_file.exists()) {
        log.warn "Model registry file not found at ${registry_path}"
        return [:]
    }
    
    def slurper = new JsonSlurper()
    def registry = slurper.parseText(json_file.text)
    
    return registry.models.findAll { k, v -> v.enabled }
}

/**
 * Get model configuration from registry
 */
def getModelConfig(registry, model_name) {
    if (!registry.containsKey(model_name)) {
        exit 1, "ERROR: Model '${model_name}' not found in registry"
    }
    return registry[model_name]
}

/**
 * Generate sample metadata map
 */
def createMeta(patient_id, model_name, additional = [:]) {
    def meta = [
        patient_id: patient_id,
        model: model_name,
        timestamp: new Date().format('yyyy-MM-dd_HH-mm-ss')
    ]
    meta.putAll(additional)
    return meta
}

/**
 * Parse frame indices from Info.cfg file
 */
def parseInfoCfg(info_cfg_path) {
    def info = [ed_frame: 0, es_frame: null]
    
    if (info_cfg_path && file(info_cfg_path).exists()) {
        file(info_cfg_path).eachLine { line ->
            if (line.startsWith('ED:')) {
                info.ed_frame = line.split(':')[1].trim() as int
            } else if (line.startsWith('ES:')) {
                info.es_frame = line.split(':')[1].trim() as int
            }
        }
    }
    
    return info
}
