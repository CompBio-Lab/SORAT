#!/usr/bin/env python3
"""
Generate Report Script for CASC Pipeline

Generates HTML report with visualizations comparing segmentation models.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_report(
    summary_file: Path,
    segmentations_dir: Path,
    models: str,
    output_report: Path,
    output_figures: Path
):
    """
    Generate comprehensive comparison report.
    
    Args:
        summary_file: Path to aggregated metrics CSV
        segmentations_dir: Directory containing segmentation files
        models: Comma-separated list of models
        output_report: Path for output HTML report
        output_figures: Directory for output figures
    """
    output_figures = Path(output_figures)
    output_figures.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    df = pd.read_csv(summary_file)
    
    model_list = [m.strip() for m in models.split(',')]
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    figures = []
    
    # 1. Overall Dice comparison bar plot
    if 'overall_dice_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_means = df.groupby('model')['overall_dice_mean'].agg(['mean', 'std']).reset_index()
        
        bars = ax.bar(model_means['model'], model_means['mean'], yerr=model_means['std'],
                      capsize=5, color=sns.color_palette("husl", len(model_means)))
        
        ax.set_ylabel('Dice Score')
        ax.set_xlabel('Model')
        ax.set_title('Overall Dice Score Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, mean in zip(bars, model_means['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        fig_path = output_figures / 'overall_dice_comparison.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        figures.append(('Overall Dice Comparison', fig_path.name))
        plt.close()
    
    # 2. Per-structure Dice comparison
    structure_cols = ['dice_rv', 'dice_myo', 'dice_lv']
    available_cols = [col for col in df.columns if any(s in col for s in structure_cols)]
    
    if available_cols:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, structure in enumerate(['rv', 'myo', 'lv']):
            ax = axes[idx]
            
            # Find ED and ES columns for this structure
            ed_col = f'ed_dice_{structure}'
            es_col = f'es_dice_{structure}'
            
            plot_data = []
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                
                if ed_col in model_df.columns:
                    for val in model_df[ed_col].dropna():
                        plot_data.append({'model': model, 'phase': 'ED', 'dice': val})
                
                if es_col in model_df.columns:
                    for val in model_df[es_col].dropna():
                        plot_data.append({'model': model, 'phase': 'ES', 'dice': val})
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.boxplot(data=plot_df, x='model', y='dice', hue='phase', ax=ax)
                ax.set_title(f'{structure.upper()} Dice Score')
                ax.set_ylabel('Dice Score')
                ax.set_xlabel('')
                ax.set_ylim(0, 1)
                ax.legend(title='Phase')
        
        fig.suptitle('Per-Structure Dice Score Comparison', y=1.02)
        plt.tight_layout()
        
        fig_path = output_figures / 'structure_dice_comparison.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        figures.append(('Per-Structure Dice Comparison', fig_path.name))
        plt.close()
    
    # 3. HD95 comparison
    if any('hd95' in col for col in df.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        hd95_col = 'overall_hd95_mean' if 'overall_hd95_mean' in df.columns else 'ed_hd95_mean'
        
        if hd95_col in df.columns:
            plot_df = df[df[hd95_col] < 100]  # Filter extreme values
            sns.boxplot(data=plot_df, x='model', y=hd95_col, ax=ax)
            ax.set_ylabel('HD95 (mm)')
            ax.set_xlabel('Model')
            ax.set_title('Hausdorff Distance (95th percentile) Comparison')
            
            fig_path = output_figures / 'hd95_comparison.png'
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            figures.append(('HD95 Comparison', fig_path.name))
            plt.close()
    
    # 4. Per-patient comparison heatmap
    if len(df['model'].unique()) > 1 and 'overall_dice_mean' in df.columns:
        pivot_df = df.pivot_table(
            index='patient_id',
            columns='model',
            values='overall_dice_mean',
            aggfunc='first'
        )
        
        if not pivot_df.empty:
            fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_df) * 0.3)))
            
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn',
                        vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Dice Score'})
            ax.set_title('Per-Patient Dice Score Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel('Patient')
            
            plt.tight_layout()
            fig_path = output_figures / 'patient_heatmap.png'
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            figures.append(('Per-Patient Heatmap', fig_path.name))
            plt.close()
    
    # Generate HTML report
    html_content = generate_html_report(df, figures, model_list)
    
    with open(output_report, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {output_report}")
    print(f"Figures saved to: {output_figures}")


def generate_html_report(df: pd.DataFrame, figures: list, models: list) -> str:
    """Generate HTML content for the report."""
    
    # Compute summary statistics
    summary_stats = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        stats = {
            'Model': model,
            'N': len(model_df)
        }
        
        if 'overall_dice_mean' in model_df.columns:
            stats['Dice (mean±std)'] = f"{model_df['overall_dice_mean'].mean():.4f} ± {model_df['overall_dice_mean'].std():.4f}"
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CASC - Cardiac Segmentation Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .figure-caption {{
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .models-list {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #95a5a6;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🫀 CASC - Cardiac Segmentation Comparison Report</h1>
        
        <div class="models-list">
            <strong>Models Compared:</strong> {', '.join(models)}
        </div>
        
        <h2>Summary Statistics</h2>
        {summary_df.to_html(index=False, classes='summary-table')}
        
        <h2>Visualizations</h2>
"""
    
    for title, fig_name in figures:
        html += f"""
        <div class="figure">
            <img src="figures/{fig_name}" alt="{title}">
            <div class="figure-caption">{title}</div>
        </div>
"""
    
    html += f"""
        <h2>Detailed Results</h2>
        {df.to_html(index=False, classes='results-table')}
        
        <div class="footer">
            <p>Generated by CASC (Cardiac Automated Segmentation Comparison) Pipeline</p>
            <p>Report generated automatically. For more information, see the pipeline documentation.</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Generate comparison report')
    parser.add_argument('--summary', required=True, help='Aggregated metrics CSV')
    parser.add_argument('--segmentations', required=True, help='Segmentations directory')
    parser.add_argument('--models', required=True, help='Comma-separated model list')
    parser.add_argument('--output_report', required=True, help='Output HTML report')
    parser.add_argument('--output_figures', required=True, help='Output figures directory')
    
    args = parser.parse_args()
    
    generate_report(
        summary_file=Path(args.summary),
        segmentations_dir=Path(args.segmentations),
        models=args.models,
        output_report=Path(args.output_report),
        output_figures=Path(args.output_figures)
    )


if __name__ == '__main__':
    main()
