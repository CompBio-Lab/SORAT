# Outputs

```text
results/
├── <model>/
│   ├── preprocessed/              model-specific inputs (also cached in .cache/preprocess)
│   └── segmentations/             predictions in the original image space: <patient>_<frame>_<model_tag>.nii.gz
├── inputs/                        NIfTI converted from other formats, with provenance JSON
├── metrics/<model>/               per-frame DSC and HD95 for each structure
├── comparison/
│   ├── aggregated_metrics.csv     every model, patient, frame and structure
│   ├── model_comparison.csv       summary statistics per model
│   ├── per_patient_summary.csv
│   ├── comparison_report.html     interactive report
│   └── figures/
├── postprocess/                   post-processed masks and before/after views (optional)
├── previews/<model_tag>/          ED/ES overlays with per-structure DSC and difference maps
├── features/                      volumes, EF, mass, wall thickness, radiomics (optional)
├── debug/                         run-time, GPU and success analytics (optional)
└── pipeline_info/                 Nextflow execution report, timeline, trace and DAG
```

## Model tags

Each prediction is named by its model tag: architecture, training data and variant. For example, `cinema__acdc_seed0` is one seed, `cinema__acdc_ensemble` is the majority vote of the three seeds, `nnformer__fold0` is nnFormer, and `vsa3l__model` is VSA-3L. When post-processed masks are scored, they appear as `<model_tag>_pp`.

## Provenance

`pipeline_info/` records the SORAT commit, the parameters, each task's container and its run time. Every converted input has a `<patient_id>_input_provenance.json` with source paths, SHA-256 checksums and the output geometry.
