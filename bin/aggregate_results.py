#!/usr/bin/env python3
"""
Aggregate Results Script for CASC Pipeline

Aggregates metrics across all models and patients for comparison.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_metrics(input_files: list, output_aggregated: Path, output_comparison: Path, output_per_patient: Path):
    """
    Aggregate metrics from multiple CSV files.
    
    Args:
        input_files: List of input CSV file paths
        output_aggregated: Path for aggregated metrics CSV
        output_comparison: Path for model comparison CSV
        output_per_patient: Path for per-patient summary CSV
    """
    # Load all metrics files
    dfs = []
    for f in input_files:
        if Path(f).exists():
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
    
    if not dfs:
        print("No metrics files found")
        pd.DataFrame().to_csv(output_aggregated, index=False)
        pd.DataFrame().to_csv(output_comparison, index=False)
        pd.DataFrame().to_csv(output_per_patient, index=False)
        return
    
    # Concatenate all metrics
    all_metrics = pd.concat(dfs, ignore_index=True)
    
    # Save aggregated metrics
    all_metrics.to_csv(output_aggregated, index=False)
    print(f"Aggregated {len(all_metrics)} metric records")
    
    # Create model comparison summary
    metric_cols = [col for col in all_metrics.columns if 'dice' in col or 'hd95' in col]
    
    comparison_data = []
    for model in all_metrics['model'].unique():
        model_df = all_metrics[all_metrics['model'] == model]

        architecture = model_df['architecture'].iloc[0] if 'architecture' in model_df.columns else ''
        model_stats = {
            'model': model,
            'architecture': architecture,
            'n_patients': len(model_df)
        }
        
        for col in metric_cols:
            if col in model_df.columns:
                values = model_df[col].dropna()
                # Filter out infinite values for HD95
                if 'hd95' in col:
                    values = values[np.isfinite(values)]
                
                if len(values) > 0:
                    model_stats[f'{col}_mean'] = values.mean()
                    model_stats[f'{col}_std'] = values.std()
                    model_stats[f'{col}_median'] = values.median()
        
        comparison_data.append(model_stats)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_comparison, index=False)
    print(f"Created comparison summary for {len(comparison_df)} models")
    
    # Create per-patient summary (pivoted by model)
    patient_summary = all_metrics.pivot_table(
        index='patient_id',
        columns='model',
        values='overall_dice_mean' if 'overall_dice_mean' in all_metrics.columns else metric_cols[0],
        aggfunc='first'
    ).reset_index()
    
    patient_summary.to_csv(output_per_patient, index=False)
    print(f"Created per-patient summary for {len(patient_summary)} patients")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    for _, row in comparison_df.iterrows():
        print(f"\n{row['model'].upper()}")
        print(f"  Patients: {row['n_patients']}")
        if 'overall_dice_mean_mean' in row:
            print(f"  Overall Dice: {row['overall_dice_mean_mean']:.4f} ± {row.get('overall_dice_mean_std', 0):.4f}")
        if 'ed_dice_mean_mean' in row:
            print(f"  ED Dice: {row['ed_dice_mean_mean']:.4f} ± {row.get('ed_dice_mean_std', 0):.4f}")
        if 'es_dice_mean_mean' in row:
            print(f"  ES Dice: {row['es_dice_mean_mean']:.4f} ± {row.get('es_dice_mean_std', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate segmentation metrics')
    parser.add_argument('--input_files', nargs='+', required=True, help='Input CSV files')
    parser.add_argument('--output_aggregated', required=True, help='Output aggregated CSV')
    parser.add_argument('--output_comparison', required=True, help='Output comparison CSV')
    parser.add_argument('--output_per_patient', required=True, help='Output per-patient CSV')
    
    args = parser.parse_args()
    
    aggregate_metrics(
        input_files=args.input_files,
        output_aggregated=Path(args.output_aggregated),
        output_comparison=Path(args.output_comparison),
        output_per_patient=Path(args.output_per_patient)
    )


if __name__ == '__main__':
    main()
