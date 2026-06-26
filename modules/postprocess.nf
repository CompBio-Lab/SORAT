/*
========================================================================================
	Postprocess Module
========================================================================================
	Optional intensity-aware LV -> MYO relabeling and visual diagnostics.
----------------------------------------------------------------------------------------
*/

process DISCOVER_POSTPROCESS_INPUTS {
	label 'process_low'

	publishDir "${params.outdir}/postprocess", mode: params.publish_dir_mode

	input:
	path samplesheet
	val results_dir
	val models

	output:
	path "postprocess_inputs.csv", emit: inputs
	path "versions.yml", emit: versions

	script:
	"""
	python /app/bin/discover_postprocess_inputs.py \
		--samplesheet ${samplesheet} \
		--results_dir ${results_dir} \
		--models "${models}" \
		--output_csv postprocess_inputs.csv

	cat <<-END_VERSIONS > versions.yml
	"${task.process}":
		python: \$(python --version | sed 's/Python //')
	END_VERSIONS
	"""
}


process POSTPROCESS_LV_MYO {
	tag "$patient_id - $model"
	label 'process_medium'

	publishDir "${params.outdir}/postprocess/${model}/segmentations", mode: params.publish_dir_mode
	publishDir "${params.outdir}/postprocess/${model}/delta", mode: params.publish_dir_mode, pattern: "*_delta_*.nii.gz"
	publishDir "${params.outdir}/postprocess/${model}/summaries", mode: params.publish_dir_mode, pattern: "*_postprocess_summary.json"

	input:
	tuple val(patient_id), val(model), val(frame_tag), val(frame_idx), path(seg), val(meta), val(image_path), val(info_cfg)

	output:
	tuple val(patient_id), val(model), val(frame_tag), val(frame_idx), path("${patient_id}_${frame_tag}_${model}_pp.nii.gz"), val(meta_pp), val(image_path), val(info_cfg), emit: segmentations
	tuple val(patient_id), val(model), val(frame_tag), path(seg), path("${patient_id}_${frame_tag}_${model}_pp.nii.gz"), path("${patient_id}_delta_${frame_tag}_${model}.nii.gz"), val(image_path), val(info_cfg), emit: before_after
	path "${patient_id}_${model}_postprocess_summary.json", emit: summaries
	path "versions.yml", emit: versions

	script:
	meta_pp = meta + [postprocessed: true, postprocess_module: 'lv_myo_otsu']

    def infoArg = info_cfg ? "--info_cfg '${info_cfg}'" : ""
    """
    cp ${projectDir}/bin/frame_manifest.py . 2>/dev/null || true

    python /app/bin/postprocess_lv_myo.py \
        --patient_id ${patient_id} \
        --model ${model} \
        --seg ${seg} \
        --frame_tag ${frame_tag} \
        --frame_idx ${frame_idx} \
        --image '${image_path}' \
        ${infoArg} \
		--output ${patient_id}_${frame_tag}_${model}_pp.nii.gz \
		--delta ${patient_id}_delta_${frame_tag}_${model}.nii.gz \
		--summary_json ${patient_id}_${model}_postprocess_summary.json \
		--method otsu \
		--strength ${params.postprocess.strength}

	cat <<-END_VERSIONS > versions.yml
	"${task.process}":
		python: \$(python --version | sed 's/Python //')
		simpleitk: \$(python -c "import SimpleITK; print(SimpleITK.__version__)")
		numpy: \$(python -c "import numpy; print(numpy.__version__)")
	END_VERSIONS
	"""
}


process VISUALIZE_POSTPROCESS_DELTA {
	tag "$patient_id - $model"
	label 'process_low'

	publishDir "${params.outdir}/postprocess/${model}/figures", mode: params.publish_dir_mode

	input:
	tuple val(patient_id), val(model), val(frame_tag), path(seg_before), path(seg_after), path(delta), val(image_path), val(info_cfg)

	output:
	path "${patient_id}_${model}_${frame_tag}_postprocess_comparison.png", emit: figures
	path "versions.yml", emit: versions

	script:
    def infoArg = info_cfg ? "--info_cfg '${info_cfg}'" : ""
    """
    cp ${projectDir}/bin/frame_manifest.py . 2>/dev/null || true

    python /app/bin/generate_postprocess_comparison.py \
        --patient_id ${patient_id} \
        --model ${model} \
        --frame_tag ${frame_tag} \
        --image '${image_path}' \
        ${infoArg} \
		--seg_before ${seg_before} \
		--seg_after ${seg_after} \
		--delta ${delta} \
		--output_png ${patient_id}_${model}_${frame_tag}_postprocess_comparison.png

	cat <<-END_VERSIONS > versions.yml
	"${task.process}":
		python: \$(python --version | sed 's/Python //')
		matplotlib: \$(python -c "import matplotlib; print(matplotlib.__version__)")
	END_VERSIONS
	"""
}
