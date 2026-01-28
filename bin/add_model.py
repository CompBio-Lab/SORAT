#!/usr/bin/env python3
"""
Model Template Generator for CASC Pipeline

Generates boilerplate code for adding a new segmentation model to the pipeline.
"""

import argparse
import json
from pathlib import Path
from string import Template


def create_module_template(model_name: str, model_name_upper: str) -> str:
    """Generate Nextflow module template."""
    return Template("""/*
========================================================================================
    ${model_name_upper} Module
========================================================================================
    Processes for ${model_name_upper} cardiac segmentation model
----------------------------------------------------------------------------------------
*/

process ${model_name_upper}_PREPROCESS {
    tag "$$patient_id"
    label 'process_medium'
    
    publishDir "$${params.outdir}/${model_name}/preprocessed", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(image), path(ground_truth), path(info_cfg)
    
    output:
    tuple val(patient_id), path("$${patient_id}_preprocessed"), path(ground_truth), path(info_cfg), emit: preprocessed
    path "versions.yml", emit: versions
    
    script:
    def gt_arg = ground_truth ? "--ground_truth $${ground_truth}" : ""
    def info_arg = info_cfg ? "--info_cfg $${info_cfg}" : ""
    \"\"\"
    ${model_name}_preprocess.py \\\\
        --input $${image} \\\\
        --patient_id $${patient_id} \\\\
        --output_dir $${patient_id}_preprocessed \\\\
        $${gt_arg} \\\\
        $${info_arg}
    
    cat <<-END_VERSIONS > versions.yml
    "$${task.process}":
        python: \\$$(python --version | sed 's/Python //')
    END_VERSIONS
    \"\"\"
}

process ${model_name_upper}_SEGMENT {
    tag "$$patient_id"
    label 'process_gpu'
    
    publishDir "$${params.outdir}/${model_name}/segmentations", mode: params.publish_dir_mode
    
    input:
    tuple val(patient_id), path(preprocessed_dir), path(ground_truth), path(info_cfg)
    
    output:
    tuple val(patient_id), path("$${patient_id}_ED_${model_name}.nii.gz"), path("$${patient_id}_ES_${model_name}.nii.gz"), val(meta), emit: segmentation
    path "versions.yml", emit: versions
    
    script:
    meta = [model: '${model_name}']
    \"\"\"
    ${model_name}_segment.py \\\\
        --input_dir $${preprocessed_dir} \\\\
        --patient_id $${patient_id} \\\\
        --output_prefix $${patient_id} \\\\
        --model_dir /models/${model_name}
    
    cat <<-END_VERSIONS > versions.yml
    "$${task.process}":
        python: \\$$(python --version | sed 's/Python //')
        torch: \\$$(python -c "import torch; print(torch.__version__)")
    END_VERSIONS
    \"\"\"
}
""").substitute(model_name=model_name, model_name_upper=model_name_upper)


def create_preprocess_script(model_name: str) -> str:
    """Generate preprocessing script template."""
    return f'''#!/usr/bin/env python3
"""
{model_name.upper()} Preprocessing Script for CASC Pipeline

Preprocesses cardiac MRI data for {model_name.upper()} model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_info_cfg(info_path: Path) -> dict:
    """Parse ACDC Info.cfg file to get ED/ES frame indices."""
    info = {{'ed_frame': 0, 'es_frame': None}}
    
    if info_path and info_path.exists():
        with open(info_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'ed':
                        info['ed_frame'] = int(value)
                    elif key == 'es':
                        info['es_frame'] = int(value)
    
    return info


def preprocess_patient(
    input_path: Path,
    output_dir: Path,
    patient_id: str,
    info_cfg: Path = None,
    ground_truth: Path = None
) -> dict:
    """
    Preprocess a single patient's data for {model_name.upper()}.
    
    Args:
        input_path: Path to input NIfTI file
        output_dir: Directory to save preprocessed data
        patient_id: Patient identifier
        info_cfg: Path to Info.cfg file (optional)
        ground_truth: Path to ground truth (optional)
    
    Returns:
        Dictionary with preprocessing metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse info config
    info = parse_info_cfg(info_cfg) if info_cfg else {{'ed_frame': 0, 'es_frame': None}}
    
    # Load image
    image = sitk.ReadImage(str(input_path))
    array = sitk.GetArrayFromImage(image)
    
    # TODO: Add your preprocessing logic here
    # Example: resampling, normalization, cropping, etc.
    
    # Save preprocessed data
    output_path = output_dir / f"{{patient_id}}_preprocessed.nii.gz"
    sitk.WriteImage(image, str(output_path), useCompression=True)
    
    # Save metadata
    metadata = {{
        'patient_id': patient_id,
        'input_path': str(input_path),
        'ed_frame': info['ed_frame'],
        'es_frame': info['es_frame']
    }}
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='{model_name.upper()} preprocessing for CASC pipeline')
    parser.add_argument('--input', required=True, help='Input NIfTI file')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--info_cfg', help='Path to Info.cfg file')
    parser.add_argument('--ground_truth', help='Path to ground truth')
    
    args = parser.parse_args()
    
    metadata = preprocess_patient(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        patient_id=args.patient_id,
        info_cfg=Path(args.info_cfg) if args.info_cfg else None,
        ground_truth=Path(args.ground_truth) if args.ground_truth else None
    )
    
    print(f"Preprocessing complete for {{args.patient_id}}")


if __name__ == '__main__':
    main()
'''


def create_segment_script(model_name: str) -> str:
    """Generate segmentation script template."""
    return f'''#!/usr/bin/env python3
"""
{model_name.upper()} Segmentation Script for CASC Pipeline

Runs cardiac segmentation inference using the {model_name.upper()} model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def load_model(model_dir: Path, device: torch.device):
    """Load {model_name.upper()} model."""
    # TODO: Implement model loading
    # model = YourModel()
    # model.load_state_dict(torch.load(model_dir / 'model.pt'))
    # model.to(device)
    # model.eval()
    # return model
    raise NotImplementedError("Model loading not implemented")


def run_inference(model, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Run segmentation inference."""
    # TODO: Implement inference
    # with torch.no_grad():
    #     input_tensor = torch.from_numpy(image).float().to(device)
    #     output = model(input_tensor)
    #     segmentation = output.argmax(dim=1).cpu().numpy()
    # return segmentation
    raise NotImplementedError("Inference not implemented")


def segment_patient(
    input_dir: Path,
    patient_id: str,
    output_prefix: str,
    model_dir: Path,
    device: torch.device = None
) -> dict:
    """
    Segment a single patient's cardiac MRI.
    
    Args:
        input_dir: Directory with preprocessed data
        patient_id: Patient identifier
        output_prefix: Prefix for output files
        model_dir: Directory containing model weights
        device: Torch device
    
    Returns:
        Dictionary with segmentation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load metadata
    with open(input_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model = load_model(model_dir, device)
    
    # Load preprocessed image
    image_path = input_dir / f"{{patient_id}}_preprocessed.nii.gz"
    image_sitk = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image_sitk)
    
    # Run inference
    segmentation = run_inference(model, image_array, device)
    
    # Save ED segmentation
    ed_seg = segmentation[metadata['ed_frame']] if len(segmentation.shape) > 3 else segmentation
    ed_sitk = sitk.GetImageFromArray(ed_seg.astype(np.uint8))
    ed_sitk.SetSpacing(image_sitk.GetSpacing()[:3])
    sitk.WriteImage(ed_sitk, f"{{output_prefix}}_ED_{model_name}.nii.gz", useCompression=True)
    
    # Save ES segmentation
    es_frame = metadata['es_frame'] or 0
    es_seg = segmentation[es_frame] if len(segmentation.shape) > 3 else segmentation
    es_sitk = sitk.GetImageFromArray(es_seg.astype(np.uint8))
    es_sitk.SetSpacing(image_sitk.GetSpacing()[:3])
    sitk.WriteImage(es_sitk, f"{{output_prefix}}_ES_{model_name}.nii.gz", useCompression=True)
    
    return {{
        'patient_id': patient_id,
        'model': '{model_name}',
        'ed_output': f"{{output_prefix}}_ED_{model_name}.nii.gz",
        'es_output': f"{{output_prefix}}_ES_{model_name}.nii.gz"
    }}


def main():
    parser = argparse.ArgumentParser(description='{model_name.upper()} segmentation for CASC pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory with preprocessed data')
    parser.add_argument('--patient_id', required=True, help='Patient identifier')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--model_dir', required=True, help='Directory containing model')
    
    args = parser.parse_args()
    
    results = segment_patient(
        input_dir=Path(args.input_dir),
        patient_id=args.patient_id,
        output_prefix=args.output_prefix,
        model_dir=Path(args.model_dir)
    )
    
    print(f"Segmentation complete for {{args.patient_id}}")


if __name__ == '__main__':
    main()
'''


def create_dockerfile(model_name: str) -> str:
    """Generate Dockerfile template."""
    return f'''# {model_name.upper()} Docker image
FROM ghcr.io/your-org/casc-base:latest

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="{model_name.upper()} cardiac segmentation model for CASC pipeline"

# Install model-specific dependencies
RUN pip install --no-cache-dir \\
    # Add your model's dependencies here
    your-model-package

# Copy model source code
COPY {model_name}/ /app/{model_name}/
ENV PYTHONPATH="/app:${{PYTHONPATH}}"

# Copy model-specific scripts
COPY bin/{model_name}_*.py /app/bin/

# Model weights structure:
# /models/{model_name}/model.pt

ENTRYPOINT ["python"]
'''


def create_registry_entry(model_name: str, model_name_upper: str) -> dict:
    """Generate model registry entry."""
    return {
        "name": model_name_upper,
        "version": "1.0.0",
        "description": f"Description of {model_name_upper} model",
        "container": f"ghcr.io/your-org/casc-{model_name}:latest",
        "module_path": f"modules/{model_name}.nf",
        "input_format": "nifti_4d",
        "output_labels": {
            "0": "background",
            "1": "right_ventricle",
            "2": "myocardium",
            "3": "left_ventricle"
        },
        "scripts": {
            "preprocess": f"bin/{model_name}_preprocess.py",
            "segment": f"bin/{model_name}_segment.py"
        },
        "model_weights": {
            "path": f"models/{model_name}",
            "format": "pytorch"
        },
        "enabled": True
    }


def generate_model_template(model_name: str, output_dir: Path):
    """Generate all template files for a new model."""
    model_name = model_name.lower().replace('-', '_').replace(' ', '_')
    model_name_upper = model_name.upper()
    
    output_dir = Path(output_dir)
    
    # Create directories
    (output_dir / 'modules').mkdir(parents=True, exist_ok=True)
    (output_dir / 'bin').mkdir(parents=True, exist_ok=True)
    (output_dir / 'containers' / model_name).mkdir(parents=True, exist_ok=True)
    
    # Generate module
    module_path = output_dir / 'modules' / f'{model_name}.nf'
    with open(module_path, 'w') as f:
        f.write(create_module_template(model_name, model_name_upper))
    print(f"Created: {module_path}")
    
    # Generate preprocessing script
    preprocess_path = output_dir / 'bin' / f'{model_name}_preprocess.py'
    with open(preprocess_path, 'w') as f:
        f.write(create_preprocess_script(model_name))
    preprocess_path.chmod(0o755)
    print(f"Created: {preprocess_path}")
    
    # Generate segmentation script
    segment_path = output_dir / 'bin' / f'{model_name}_segment.py'
    with open(segment_path, 'w') as f:
        f.write(create_segment_script(model_name))
    segment_path.chmod(0o755)
    print(f"Created: {segment_path}")
    
    # Generate Dockerfile
    dockerfile_path = output_dir / 'containers' / model_name / 'Dockerfile'
    with open(dockerfile_path, 'w') as f:
        f.write(create_dockerfile(model_name))
    print(f"Created: {dockerfile_path}")
    
    # Generate registry entry
    registry_entry = create_registry_entry(model_name, model_name_upper)
    registry_path = output_dir / 'conf' / f'{model_name}_registry_entry.json'
    (output_dir / 'conf').mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry_entry, f, indent=4)
    print(f"Created: {registry_path}")
    
    print(f"\n✓ Template files generated for model '{model_name}'")
    print("\nNext steps:")
    print(f"1. Implement preprocessing logic in bin/{model_name}_preprocess.py")
    print(f"2. Implement segmentation logic in bin/{model_name}_segment.py")
    print(f"3. Update containers/{model_name}/Dockerfile with dependencies")
    print(f"4. Add the model entry from conf/{model_name}_registry_entry.json to conf/model_registry.json")
    print(f"5. Import and add the module to main.nf workflow")


def main():
    parser = argparse.ArgumentParser(description='Generate template files for a new CASC model')
    parser.add_argument('model_name', help='Name of the new model')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    
    args = parser.parse_args()
    
    generate_model_template(args.model_name, Path(args.output_dir))


if __name__ == '__main__':
    main()
