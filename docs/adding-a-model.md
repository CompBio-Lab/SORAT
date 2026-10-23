# Adding a model

A model in SORAT is three things:
- a container with the model's code and weights;
- a preprocessing script and a segmentation script in `bin/`;
- a Nextflow module in `modules/` that runs them.

The segmentation script must write masks in the original image space, using SORAT's label set.

## 1. Generate the templates

```bash
python3 bin/add_model.py mymodel
```

This writes:
- `modules/mymodel.nf`;
- `bin/mymodel_preprocess.py` and `bin/mymodel_segment.py`;
- a Dockerfile template;
- an entry to add to `conf/model_registry.json`.

## 2. Implement preprocessing and segmentation

- **Preprocessing** reads the image named in the samplesheet and selects frames with `frame_manifest.py`. It must record the original geometry (spacing, origin, direction) in `metadata.json`.
- **Segmentation** maps its labels to 1 RV, 2 MYO, 3 LV (or 1 wall, 2 RA, 3 LA for atrial models). It resamples each prediction to the original geometry with `geometry_utils.py` and writes `<patient>_<frame>_<model_tag>.nii.gz`.

The existing models are worked examples: `bin/nnformer_*.py` (3-D) and `bin/vsa3l_*.py` (2-D, slice by slice).

## 3. Wire it into the pipeline

These steps are still manual:
- `main.nf`: add the discover, preprocess and segment block. Add the name to `parseModels` and `inferArchitectureFromModel`. Give the preprocessing a cache key with a version suffix, and bump that suffix whenever preprocessing output changes.
- `nextflow.config`: add a `withName` block for the container and resource labels; GPU processes use a `process_gpu*` label.
- `modules/discover_models.nf`: add a discovery process if the container holds several checkpoints.

## 4. Test

Add unit tests to `tests/`, then run the model on a few subjects with ground truth. Compare its masks with the model's own inference code, as described in [Reproducing the paper](../paper/README.md#3-standalone-runs).
