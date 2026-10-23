# Outputs

```text
results/
├── <model>/
│   ├── preprocessed/              model-specific inputs (also cached in .cache/preprocess)
│   └── segmentations/             predictions in the original image space: <patient>_<frame>_<model_tag>.nii.gz
├── inputs/                        NIfTI converted from other formats, with provenance JSON
├── metrics/<model_tag>/           per-frame DSC and HD95 for each structure
├── comparison/
│   ├── aggregated_metrics.csv     every model, patient, frame and structure
│   ├── model_comparison.csv       summary statistics per model
│   ├── per_patient_summary.csv
│   ├── comparison_report.html     interactive report
│   └── figures/
├── postprocess/<model_tag>/       post-processed masks and before/after views (optional)
├── previews/<model_tag>/          ED/ES overlays with per-structure DSC and difference maps
├── features/raw/                  volumes, mass, wall thickness, radiomics (optional; features/postprocessed/ for post-processed masks)
├── debug/                         run-time, GPU and success analytics (optional)
└── pipeline_info/                 Nextflow execution report, timeline, trace and DAG (.dot)
```

## Model tags

Each prediction is named by its model tag: architecture, training data and variant. For example, `cinema__acdc_seed0` is one seed, `cinema__acdc_ensemble` is the majority vote of the three seeds, `nnformer__fold0` is nnFormer, and `vsa3l__model` is VSA-3L. When post-processed masks are scored, they appear as `<model_tag>_pp`.

## Provenance

Nextflow's execution report in `pipeline_info/` records the command line, the container image path assigned to each process and every task's run time; each process also writes a `versions.yml` with its Python and library versions. Every converted input has a `<patient_id>_input_provenance.json` with source paths, SHA-256 checksums and the output geometry.
