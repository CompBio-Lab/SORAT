# Features

With `--feature_extraction.enabled true`, SORAT writes one CSV per mask and phase to `features/raw/<patient>_<model_tag>_<ED|ES>_features.csv` (`features/postprocessed/` when the features come from post-processed masks).

Each row gives the LV, RV and myocardial volumes in ml and the myocardial mass in g (volume × 1.05 g/ml); ejection fractions follow from the ED and ES volumes. Wall thickness (`wall_thickness_mean_mm`, `_max_mm`, `_p95_mm`) is measured in-plane, slice by slice, because short-axis stacks are strongly anisotropic. The samples are the myocardium pixels 4-connected to the LV cavity, and the thickness at each sample is its in-plane distance to the nearest pixel that is neither myocardium nor LV, so the septum is measured to its RV border. Radiomics adds shape, first-order and GLCM features of the myocardium from PyRadiomics.

The same functions are applied to reference masks, so automatic and manual measurements share one definition.

## Python environment for radiomics

PyRadiomics is not in the model containers. Build a virtual environment with the container's own Python and pass it to SORAT, which adds its `site-packages` to `PYTHONPATH`:

```bash
apptainer exec containers/sorat-cinema.sif python -m venv /path/to/venvs/sorat-features
apptainer exec containers/sorat-cinema.sif /path/to/venvs/sorat-features/bin/python -m pip install pyradiomics

nextflow run main.nf ... --feature_extraction.enabled true --feature_extraction.mask_source predictions \
    --feature_extraction.virtualenv_path /path/to/venvs/sorat-features --feature_extraction.require_virtualenv true
```

The environment must use the same Python major and minor version as the container. On SLURM, always set `--feature_extraction.mask_source` (`predictions` or `postprocess`); the default, `prompt`, asks interactively.

## Features from an earlier run

```bash
nextflow run main.nf -entry FEATURES_ONLY -profile slurm --slurm_account <account> \
    --models all --input samplesheet.csv \
    --feature_extraction.enabled true --feature_extraction.mask_source predictions \
    --feature_extraction.results_dir /path/to/results --outdir /path/to/results_features
```

`--feature_extraction.results_dir` is read and never written. The features go to this run's `--outdir`, or to `--feature_extraction.output_dir`.

## Radiomics settings

The defaults use PyRadiomics' settings: no normalization, bin width 25, 3-D and native spacing. For comparisons across scanners, a harmonized setup could be:

```bash
--feature_extraction.radiomics.normalize true --feature_extraction.radiomics.normalize_scale 100 \
--feature_extraction.radiomics.bin_count 32 --feature_extraction.radiomics.resample_spacing '1.25,1.25,0' \
--feature_extraction.radiomics.force2d true --feature_extraction.output_dir /path/to/features_harmonized
```

`normalize` z-scores against the whole image, whose content differs between scanners. `intensity_reference lv_bloodpool` instead scales each image by its own LV blood pool. Check either choice on your data before relying on texture features across vendors.

| Parameter | Default | Description |
|---|---|---|
| `--feature_extraction.radiomics.normalize` | `false` | Z-score the image before radiomics (PyRadiomics `normalize`; uses the whole image, not just the mask) |
| `--feature_extraction.radiomics.normalize_scale` | `null` | Scale applied after normalization (PyRadiomics `normalizeScale`, e.g. `100`) |
| `--feature_extraction.radiomics.remove_outliers` | `null` | Clip normalized intensities beyond N standard deviations (`removeOutliers`) |
| `--feature_extraction.radiomics.bin_count` | `null` | Fixed number of grey-level bins (`binCount`); mutually exclusive with `bin_width` |
| `--feature_extraction.radiomics.bin_width` | `null` | Grey-level bin width (`binWidth`; PyRadiomics default 25) |
| `--feature_extraction.radiomics.resample_spacing` | `null` | Resample to `'x,y,z'` mm before extraction; `0` keeps that axis (e.g. `'1.25,1.25,0'`) |
| `--feature_extraction.radiomics.force2d` | `false` | Compute texture per slice in 2-D (`force2D`) |
| `--feature_extraction.radiomics.force2d_dimension` | `0` | Axis treated as the slice axis when `force2d` is on (`force2Ddimension`) |
| `--feature_extraction.radiomics.intensity_reference` | `null` | `lv_bloodpool`: divide each image by the mean intensity of its LV blood pool (× `intensity_reference_scale`, default 100) before radiomics; an alternative to `normalize`, which uses the whole image |
| `--feature_extraction.radiomics.intensity_reference_scale` | `null` | Value given to the reference tissue's mean intensity (default 100) |
