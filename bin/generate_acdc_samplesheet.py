#!/usr/bin/env python3
"""
Generate ACDC Samplesheet for SORAT Pipeline

Creates a CSV samplesheet from ACDC dataset directory structure.
"""

import argparse
from pathlib import Path


def generate_samplesheet(acdc_dir: Path, output_path: Path, dataset: str = "testing"):
    """
    Generate a samplesheet CSV from ACDC dataset.
    
    Args:
        acdc_dir: Path to ACDC database directory
        output_path: Path for output CSV
        dataset: 'testing' or 'training'
    """
    data_dir = acdc_dir / dataset
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Find all patient directories
    patient_dirs = sorted([d for d in data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('patient')])
    
    rows = ["patient_id,image,ground_truth,info_cfg"]
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        
        # Look for 4D image
        image_4d = patient_dir / f"{patient_id}_4d.nii.gz"
        if not image_4d.exists():
            print(f"Warning: No 4D image found for {patient_id}, skipping")
            continue
        
        # Info.cfg path
        info_cfg = patient_dir / "Info.cfg"
        info_cfg_str = str(info_cfg) if info_cfg.exists() else ""
        
        # Ground truth directory (the patient folder contains GT files)
        gt_dir = str(patient_dir)
        
        rows.append(f"{patient_id},{image_4d},{gt_dir},{info_cfg_str}")
    
    # Write samplesheet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(rows))
    
    print(f"Generated samplesheet with {len(rows) - 1} patients: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate ACDC samplesheet')
    parser.add_argument('--acdc_dir', required=True, help='Path to ACDC database directory')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--dataset', default='testing', choices=['testing', 'training'],
                        help='Which dataset to use')
    
    args = parser.parse_args()
    
    generate_samplesheet(
        acdc_dir=Path(args.acdc_dir),
        output_path=Path(args.output),
        dataset=args.dataset
    )


if __name__ == '__main__':
    main()
