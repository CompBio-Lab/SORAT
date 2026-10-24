# Adding a model

Each model in SORAT has a container with its code and weights, a preprocessing script and a segmentation script in `bin/`, and a Nextflow module in `modules/` that runs them. Whatever the model does internally, its masks must come out in the original image space and in SORAT's label set.

## 1. Generate the templates

```bash
python3 bin/add_model.py mymodel
```

This writes `modules/mymodel.nf`, `bin/mymodel_preprocess.py`, `bin/mymodel_segment.py`, a Dockerfile template, and an entry to add to `conf/model_registry.json`.

## 2. Implement preprocessing and segmentation

The preprocessing script reads the image named in the samplesheet, selects frames with `frame_manifest.py` and records the original geometry (spacing, origin, direction) in `metadata.json`. The segmentation script maps the model's labels to 1 RV, 2 myocardium, 3 LV (or 1 wall, 2 RA, 3 LA for an atrial model), resamples each prediction to the original geometry with `geometry_utils.py`, and writes `<patient>_<frame>_<model_tag>.nii.gz`.

The existing models are worked examples: `bin/nnformer_*.py` (3-D) and `bin/vsa3l_*.py` (2-D, slice by slice).

## 3. Wire it into the pipeline

These steps are still manual:
- `main.nf`: add the discover, preprocess and segment block. Add the name to `parseModels` and `inferArchitectureFromModel`. Give the preprocessing a cache key with a version suffix, and bump that suffix whenever preprocessing output changes.
- `nextflow.config`: add a `withName` block for the container and resource labels; GPU processes use a `process_gpu*` label.
- `modules/discover_models.nf`: add a discovery process if the container holds several checkpoints.

## 4. Test

Add unit tests to `tests/`, then run the model on a few subjects with ground truth. Compare its masks with the model's own inference code, as described in [Reproducing the paper](../paper/README.md#3-standalone-runs).
