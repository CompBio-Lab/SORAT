# Model Weights Directory

This directory contains model weights for the segmentation models.

## Directory Structure

```
models/
├── cinema/
│   └── finetuned/
│       └── segmentation/
│           └── acdc_sax/
│               ├── acdc_sax_0.safetensors
│               ├── acdc_sax_1.safetensors
│               ├── acdc_sax_2.safetensors
│               └── config.yaml
├── nnformer/
│   └── nnFormer_trained_models/
│       └── nnFormer/
│           └── 3d_fullres/
│               └── Task001_ACDC/
│                   └── nnFormerTrainerV2_nnformer_acdc__nnFormerPlansv2.1/
│                       └── fold_0/
│                           └── model_best.model.pkl
└── vsa3l/
    ├── model.pt
    └── configs/
        └── inference.json
```

## Downloading Models

### CineMA
Model weights should be placed in `models/cinema/finetuned/segmentation/`.

### nnFormer
Model weights should be placed in `models/nnformer/nnFormer_trained_models/`.

### VSA-3L (MONAI)
Download from MONAI Model Zoo and place in `models/vsa3l/`.

## Note

Model weights are not included in this repository due to their size.
Please download them separately from the respective sources.
