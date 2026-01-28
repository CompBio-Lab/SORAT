#!/usr/bin/env python3
"""
Generate Report Script for CASC Pipeline

Generates HTML report with visualizations comparing segmentation models.
"""

import argparse
import base64
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False


ARCH_LABELS = {
    'cinema': 'CineMA',
    'nnformer': 'nnFormer',
    'vsa3l': 'VSA-3L'
}


def infer_architecture(model: str) -> str:
    if '__' in model:
        return model.split('__', 1)[0]
    for prefix in ['cinema', 'nnformer', 'vsa3l']:
        if model.startswith(prefix):
            return prefix
    return 'unknown'


def build_model_display(model: str, architecture: str) -> str:
    if '__' in model:
        variant = model.split('__', 1)[1].replace('_', ' ')
    else:
        variant = model.replace('_', ' ')
    arch_label = ARCH_LABELS.get(architecture, architecture.title())
    return f"{arch_label} • {variant}" if variant else arch_label


def build_model_variant(model: str) -> str:
    if '__' in model:
        return model.split('__', 1)[1].replace('_', ' ')
    for prefix in ['cinema', 'nnformer', 'vsa3l']:
        if model.startswith(prefix):
            return model[len(prefix):].lstrip('_').replace('_', ' ') or model
    return model.replace('_', ' ')


def encode_image(path: Path) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


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

    model_list = [m.strip() for m in models.split(',') if m.strip()]

    if df.empty:
        html_content = generate_html_report(df, [], model_list)
        with open(output_report, 'w') as f:
            f.write(html_content)
        print(f"Report generated: {output_report}")
        print(f"Figures saved to: {output_figures}")
        return

    if 'architecture' not in df.columns:
        df['architecture'] = df['model'].apply(infer_architecture)
    df['model_display'] = df.apply(lambda r: build_model_display(r['model'], r['architecture']), axis=1)
    df['model_variant'] = df['model'].apply(build_model_variant)
    if df['model_variant'].duplicated().any():
        df['model_variant'] = df.apply(
            lambda r: f"{r['model_variant']} ({ARCH_LABELS.get(r['architecture'], r['architecture'].title())})",
            axis=1
        )

    model_variants = sorted(df['model_variant'].unique())
    if _HAS_SEABORN:
        model_palette = dict(zip(model_variants, sns.color_palette('tab20', n_colors=len(model_variants))))
    else:
        model_palette = dict(zip(model_variants, plt.cm.tab20(np.linspace(0, 1, len(model_variants)))))
    def abbreviate_model_label(label: str) -> str:
        if '•' in label:
            arch, variant = [part.strip() for part in label.split('•', 1)]
            arch_abbrev = ''.join([token[0] for token in arch.split() if token]).upper()
            variant_tokens = [token for token in variant.split() if token]
            if variant_tokens:
                variant_abbrev = '-'.join([token[:3].lower() for token in variant_tokens])
                return f"{arch_abbrev}-{variant_abbrev}"
            return arch_abbrev
        tokens = [token for token in label.split() if token]
        return ''.join([token[0] for token in tokens]).upper() if tokens else label

    model_legend_handles = [
        plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=model_palette[m],
                   markersize=9, label=m)
        for m in model_variants
    ]

    # Set style
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['figure.figsize'] = (12, 6)

    figures = []

    def add_figure(title: str, fig_path: Path):
        figures.append({
            'title': title,
            'filename': fig_path.name,
            'data_uri': f"data:image/png;base64,{encode_image(fig_path)}"
        })

    # 1. Overall Dice comparison (distribution)
    if 'overall_dice_mean' in df.columns:
        plot_df = df.dropna(subset=['overall_dice_mean'])
        order = model_variants
        fig_width = max(10, len(order) * 1.2)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        if _HAS_SEABORN:
            sns.boxplot(
                data=plot_df,
                x='model_variant',
                y='overall_dice_mean',
                order=order,
                palette=model_palette,
                showfliers=False,
                linewidth=1.6,
                width=0.6,
                ax=ax
            )
            sns.stripplot(
                data=plot_df,
                x='model_variant',
                y='overall_dice_mean',
                order=order,
                color='black',
                size=3.5,
                alpha=0.35,
                jitter=0.22,
                ax=ax
            )
            ax.legend(handles=model_legend_handles, title='Model', frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        else:
            data = [plot_df[plot_df['model_variant'] == m]['overall_dice_mean'].values for m in order]
            box = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
            for patch, model in zip(box['boxes'], order):
                patch.set_facecolor(model_palette[model])
                patch.set_alpha(0.6)
            ax.legend(handles=model_legend_handles, title='Model', frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1.0))

        ymin = max(0.0, plot_df['overall_dice_mean'].min() - 0.05)
        ymax = min(1.0, plot_df['overall_dice_mean'].max() + 0.02)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('Dice Score')
        ax.set_xlabel('')
        ax.set_title('Overall Dice Score Distribution')
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', length=0)
        plt.tight_layout()

        fig_path = output_figures / 'overall_dice_comparison.png'
        fig.savefig(fig_path, dpi=170, bbox_inches='tight')
        add_figure('Overall Dice Distribution', fig_path)
        plt.close()

    # 2. Per-structure Dice comparison
    structure_cols = ['dice_rv', 'dice_myo', 'dice_lv']
    available_cols = [col for col in df.columns if any(s in col for s in structure_cols)]

    if available_cols:
        plot_data = []
        for structure in ['rv', 'myo', 'lv']:
            ed_col = f'ed_dice_{structure}'
            es_col = f'es_dice_{structure}'
            if ed_col in df.columns:
                for _, row in df[['model', 'model_display', 'architecture', ed_col]].dropna().iterrows():
                    plot_data.append({
                        'model': row['model'],
                        'model_display': row['model_display'],
                        'architecture': row['architecture'],
                        'phase': 'ED',
                        'structure': structure.upper(),
                        'dice': row[ed_col]
                    })
            if es_col in df.columns:
                for _, row in df[['model', 'model_display', 'architecture', es_col]].dropna().iterrows():
                    plot_data.append({
                        'model': row['model'],
                        'model_display': row['model_display'],
                        'architecture': row['architecture'],
                        'phase': 'ES',
                        'structure': structure.upper(),
                        'dice': row[es_col]
                    })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            plot_df['model_variant'] = plot_df['model'].apply(build_model_variant)
            if plot_df['model_variant'].duplicated().any():
                plot_df['model_variant'] = plot_df.apply(
                    lambda r: f"{r['model_variant']} ({ARCH_LABELS.get(r['architecture'], r['architecture'].title())})",
                    axis=1
                )
            order = model_variants
            fig, axes = plt.subplots(2, 3, figsize=(max(18, len(order) * 1.6), 9), sharey=True)

            for row_idx, phase in enumerate(['ED', 'ES']):
                for col_idx, structure in enumerate(['RV', 'MYO', 'LV']):
                    ax = axes[row_idx][col_idx]
                    struct_df = plot_df[(plot_df['structure'] == structure) & (plot_df['phase'] == phase)]
                    if _HAS_SEABORN:
                        sns.boxplot(
                            data=struct_df,
                            x='model_variant',
                            y='dice',
                            order=order,
                            palette=model_palette,
                            showfliers=False,
                            linewidth=1.2,
                            width=0.6,
                            ax=ax
                        )
                    else:
                        data = [struct_df[struct_df['model_variant'] == model]['dice'].values for model in order]
                        box = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
                        for patch, model in zip(box['boxes'], order):
                            patch.set_facecolor(model_palette[model])
                            patch.set_alpha(0.6)
                    ax.set_title(f'{structure} Dice ({phase})')
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', which='both', length=0)
                    if col_idx == 0:
                        ax.set_ylabel('Dice Score')
                    if ax.get_legend():
                        ax.legend_.remove()

            fig.legend(handles=model_legend_handles, loc='upper left', title='Model', bbox_to_anchor=(1.02, 1.0))
            fig.suptitle('Per-Structure Dice Score Comparison', y=1.02)
            plt.subplots_adjust(right=0.8)
            plt.tight_layout()

            fig_path = output_figures / 'structure_dice_comparison.png'
            fig.savefig(fig_path, dpi=170, bbox_inches='tight')
            add_figure('Per-Structure Dice Comparison', fig_path)
            plt.close()

    # 3. HD95 comparison
    if any('hd95' in col for col in df.columns):
        hd95_col = 'overall_hd95_mean' if 'overall_hd95_mean' in df.columns else 'ed_hd95_mean'
        if hd95_col in df.columns:
            plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[hd95_col])
            if not plot_df.empty:
                order = model_variants
                fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.2), 6))
                if _HAS_SEABORN:
                    sns.boxplot(
                        data=plot_df,
                        x='model_variant',
                        y=hd95_col,
                        order=order,
                        palette=model_palette,
                        showfliers=False,
                        linewidth=1.3,
                        width=0.6,
                        ax=ax
                    )
                    sns.stripplot(
                        data=plot_df,
                        x='model_variant',
                        y=hd95_col,
                        order=order,
                        color='black',
                        size=3.2,
                        alpha=0.35,
                        jitter=0.22,
                        ax=ax
                    )
                    ax.legend(handles=model_legend_handles, title='Model', frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1.0))
                else:
                    data = [plot_df[plot_df['model_variant'] == m][hd95_col].values for m in order]
                    box = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
                    for patch, model in zip(box['boxes'], order):
                        patch.set_facecolor(model_palette[model])
                        patch.set_alpha(0.6)
                    ax.legend(handles=model_legend_handles, title='Model', frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1.0))

                upper = np.quantile(plot_df[hd95_col], 0.95) if len(plot_df) > 5 else plot_df[hd95_col].max()
                ax.set_ylim(0, max(1.0, upper * 1.2))
                ax.set_ylabel('HD95 (mm)')
                ax.set_xlabel('')
                ax.set_title('Hausdorff Distance (95th percentile)')
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
                plt.tight_layout()

                fig_path = output_figures / 'hd95_comparison.png'
                fig.savefig(fig_path, dpi=170, bbox_inches='tight')
                add_figure('HD95 Comparison', fig_path)
                plt.close()

    # 4. Per-patient comparison heatmap
    if len(df['model'].unique()) > 1 and 'overall_dice_mean' in df.columns:
        pivot_df = df.pivot_table(
            index='patient_id',
            columns='model_display',
            values='overall_dice_mean',
            aggfunc='first'
        )

        if not pivot_df.empty:
            order = sorted(pivot_df.columns)
            pivot_df = pivot_df[order]
            fig_height = max(6, len(pivot_df) * 0.25)
            fig_width = max(14, len(order) * 1.4)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            if _HAS_SEABORN:
                sns.heatmap(pivot_df, annot=False, cmap='RdYlGn',
                            vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Dice Score'})
            else:
                im = ax.imshow(pivot_df.values, cmap='RdYlGn', vmin=0.5, vmax=1.0)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Dice Score')

            ax.set_title('Per-Patient Dice Score Heatmap')
            ax.set_xlabel('Model')
            ax.set_ylabel('Patient')
            ax.tick_params(axis='x', rotation=0)

            model_labels = list(pivot_df.columns)
            abbreviations = {label: abbreviate_model_label(label) for label in model_labels}
            ax.set_xticks(np.arange(len(model_labels)) + 0.5)
            ax.set_xticklabels([str(i + 1) for i in range(len(model_labels))], fontsize=10)
            for tick_label, model in zip(ax.get_xticklabels(), model_labels):
                tick_label.set_color(model_palette.get(model, '#333333'))

            heatmap_legend = [
                plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=model_palette.get(model, '#333333'),
                           markersize=8, label=f"{idx + 1}  –  {model}")
                for idx, model in enumerate(model_labels)
            ]
            ax.legend(handles=heatmap_legend, title='Model Legend',
                      loc='upper center', bbox_to_anchor=(0.5, -0.08),
                      ncol=2, borderaxespad=0.0)

            if len(pivot_df) > 40:
                step = max(2, len(pivot_df) // 30)
                yticks = np.arange(0, len(pivot_df), step)
                ax.set_yticks(yticks + 0.5)
                ax.set_yticklabels(pivot_df.index[::step], fontsize=8)
            else:
                ax.set_yticklabels(pivot_df.index, fontsize=8)

            plt.subplots_adjust(bottom=0.22)
            plt.tight_layout()
            fig_path = output_figures / 'patient_heatmap.png'
            fig.savefig(fig_path, dpi=170, bbox_inches='tight')
            add_figure('Per-Patient Heatmap', fig_path)
            plt.close()

    # Generate HTML report
    html_content = generate_html_report(df, figures, model_list)
    
    with open(output_report, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {output_report}")
    print(f"Figures saved to: {output_figures}")


def generate_html_report(df: pd.DataFrame, figures: list, models: list) -> str:
    """Generate HTML content for the report."""

    if df.empty:
        summary_df = pd.DataFrame()
    else:
        if 'architecture' not in df.columns:
            df['architecture'] = df['model'].apply(infer_architecture)
        if 'model_display' not in df.columns:
            df['model_display'] = df.apply(lambda r: build_model_display(r['model'], r['architecture']), axis=1)

        summary_stats = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            architecture = model_df['architecture'].iloc[0] if 'architecture' in model_df.columns else ''
            display_name = model_df['model_display'].iloc[0] if 'model_display' in model_df.columns else model

            stats = {
                'Model': display_name,
                'Architecture': ARCH_LABELS.get(architecture, architecture.title()) if architecture else '',
                'N': len(model_df)
            }

            if 'overall_dice_mean' in model_df.columns:
                stats['Dice (mean±std)'] = f"{model_df['overall_dice_mean'].mean():.4f} ± {model_df['overall_dice_mean'].std():.4f}"
            if 'overall_hd95_mean' in model_df.columns:
                valid_hd95 = model_df['overall_hd95_mean'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_hd95) > 0:
                    stats['HD95 (mean±std)'] = f"{valid_hd95.mean():.2f} ± {valid_hd95.std():.2f}"

            summary_stats.append(stats)

        summary_df = pd.DataFrame(summary_stats)

    def format_value(val):
        if pd.isna(val):
            return ""
        if isinstance(val, (float, np.floating)):
            return f"{val:.4f}" if abs(val) <= 1.5 else f"{val:.2f}"
        return str(val)

    if not df.empty:
        display_df = df.copy()
        if 'model_display' in display_df.columns:
            display_df = display_df.rename(columns={'model_display': 'model_display_label'})

        preferred_cols = [
            'patient_id', 'architecture', 'model_display_label', 'model',
            'overall_dice_mean', 'ed_dice_mean', 'es_dice_mean',
            'overall_hd95_mean', 'ed_hd95_mean', 'es_hd95_mean'
        ]
        preferred_cols = [c for c in preferred_cols if c in display_df.columns]
        other_cols = [c for c in display_df.columns if c not in preferred_cols]
        display_df = display_df[preferred_cols + other_cols]
        display_df = display_df.rename(columns={'model_display_label': 'model_display'})
    else:
        display_df = df

    core_cols = ['patient_id', 'architecture', 'model_display', 'overall_dice_mean', 'ed_dice_mean', 'es_dice_mean']
    core_cols = [c for c in core_cols if c in display_df.columns]

    def build_table_html(table_df: pd.DataFrame, table_id: str) -> str:
        cols = list(table_df.columns)
        header_cells = []
        for col in cols:
            cls = 'advanced-col' if col not in core_cols else ''
            header_cells.append(f"<th class='{cls}'>{escape(col)}</th>")

        rows_html = []
        for _, row in table_df.iterrows():
            row_arch = row.get('architecture', '') if isinstance(row, pd.Series) else ''
            row_cells = []
            for col in cols:
                cls = 'advanced-col' if col not in core_cols else ''
                row_cells.append(f"<td class='{cls}'>{escape(format_value(row[col]))}</td>")
            rows_html.append(
                f"<tr data-arch='{escape(str(row_arch))}'>" + "".join(row_cells) + "</tr>"
            )

        return (
            f"<table id='{table_id}' class='data-table'>"
            "<thead><tr>" + "".join(header_cells) + "</tr></thead>"
            "<tbody>" + "".join(rows_html) + "</tbody>"
            "</table>"
        )

    summary_table_html = summary_df.to_html(index=False, classes='summary-table') if not summary_df.empty else "<p>No summary available.</p>"
    results_table_html = build_table_html(display_df, 'results-table') if not display_df.empty else "<p>No detailed results available.</p>"

    arch_options = sorted(set(display_df['architecture'].unique())) if not display_df.empty and 'architecture' in display_df.columns else []
    arch_options_html = "".join([f"<option value='{escape(a)}'>{escape(ARCH_LABELS.get(a, a.title()))}</option>" for a in arch_options])

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CASC - Cardiac Segmentation Comparison Report</title>
    <style>
        :root {{
            color-scheme: light;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 24px;
            background: #f6f7fb;
            color: #1f2a44;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 32px 40px;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(31, 42, 68, 0.08);
        }}
        h1 {{
            color: #13213c;
            border-bottom: 3px solid #5b8cff;
            padding-bottom: 12px;
        }}
        h2 {{
            color: #1f2a44;
            margin-top: 32px;
        }}
        .models-list {{
            background: linear-gradient(120deg, #e8f0ff 0%, #f8fbff 100%);
            padding: 16px;
            border-radius: 12px;
            margin: 20px 0;
            font-size: 0.95rem;
        }}
        .figure {{
            margin: 24px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(16, 24, 40, 0.12);
        }}
        .figure-caption {{
            font-style: italic;
            color: #5f6b85;
            margin-top: 8px;
        }}
        details {{
            background: #f9fafc;
            border-radius: 12px;
            padding: 12px 16px;
            margin: 16px 0;
            border: 1px solid #e4e8f1;
        }}
        details summary {{
            cursor: pointer;
            font-weight: 600;
            color: #1f2a44;
        }}
        .table-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            margin: 12px 0;
        }}
        .table-controls input,
        .table-controls select {{
            padding: 8px 10px;
            border: 1px solid #ccd3e0;
            border-radius: 8px;
            font-size: 0.9rem;
        }}
        .table-controls label {{
            font-size: 0.9rem;
            color: #445070;
        }}
        .table-container {{
            overflow: auto;
            max-height: 520px;
            border: 1px solid #e4e8f1;
            border-radius: 12px;
            background: white;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            font-size: 0.85rem;
        }}
        th, td {{
            border-bottom: 1px solid #edf0f6;
            padding: 10px 12px;
            text-align: left;
            word-break: break-word;
        }}
        th {{
            position: sticky;
            top: 0;
            background-color: #5b8cff;
            color: white;
            z-index: 1;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fd;
        }}
        tr:hover {{
            background-color: #eef3ff;
        }}
        .advanced-col {{
            display: none;
        }}
        .footer {{
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #e4e8f1;
            color: #7a869c;
            font-size: 0.9rem;
        }}
        .chip {{
            display: inline-block;
            padding: 6px 12px;
            background: #e9f0ff;
            color: #3c4d77;
            border-radius: 999px;
            font-size: 0.85rem;
            margin: 4px 6px 4px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🫀 CASC - Cardiac Segmentation Comparison Report</h1>

        <div class="models-list">
            <strong>Models Requested:</strong> {', '.join(models) if models else 'auto'}
            <div style="margin-top:8px;">
                {' '.join([f"<span class='chip'>{escape(ARCH_LABELS.get(a, a.title()))}</span>" for a in arch_options])}
            </div>
        </div>

        <details open>
            <summary>Summary Statistics</summary>
            {summary_table_html}
        </details>

        <details open>
            <summary>Visualizations</summary>
"""

    for fig in figures:
        src = fig.get('data_uri') or f"figures/{fig.get('filename', '')}"
        html += f"""
            <div class="figure">
                <img src="{src}" alt="{escape(fig.get('title', 'Figure'))}">
                <div class="figure-caption">{escape(fig.get('title', ''))}</div>
            </div>
        """

    html += f"""
        </details>

        <details>
            <summary>Detailed Results (interactive)</summary>
            <div class="table-controls">
                <label>Search: <input type="text" id="table-search" placeholder="Filter rows"></label>
                <label>Architecture:
                    <select id="arch-filter">
                        <option value="all">All</option>
                        {arch_options_html}
                    </select>
                </label>
                <label>Show rows:
                    <select id="row-limit">
                        <option value="50">50</option>
                        <option value="100" selected>100</option>
                        <option value="250">250</option>
                        <option value="all">All</option>
                    </select>
                </label>
                <label><input type="checkbox" id="toggle-advanced"> Show advanced metrics</label>
                <span style="margin-left:auto; font-size:0.85rem; color:#6b778c;">Visible rows: <span id="row-count">0</span></span>
            </div>
            <div class="table-container">
                {results_table_html}
            </div>
        </details>

        <div class="footer">
            <p>Generated by CASC (Cardiac Automated Segmentation Comparison) Pipeline</p>
            <p>Report generated automatically. Open this file directly in your browser or VS Code to view offline.</p>
        </div>
    </div>

    <script>
        const table = document.getElementById('results-table');
        const searchInput = document.getElementById('table-search');
        const archFilter = document.getElementById('arch-filter');
        const rowLimit = document.getElementById('row-limit');
        const rowCount = document.getElementById('row-count');
        const toggleAdvanced = document.getElementById('toggle-advanced');

        function applyFilters() {{
            if (!table) return;
            const query = (searchInput?.value || '').toLowerCase();
            const arch = archFilter?.value || 'all';
            const limitValue = rowLimit?.value || 'all';
            const limit = limitValue === 'all' ? Infinity : parseInt(limitValue, 10);
            const rows = table.querySelectorAll('tbody tr');

            let visible = 0;
            rows.forEach(row => {{
                const rowText = row.textContent.toLowerCase();
                const rowArch = row.dataset.arch || '';
                const matches = (arch === 'all' || rowArch === arch) && rowText.includes(query);
                if (matches && visible < limit) {{
                    row.style.display = '';
                    visible += 1;
                }} else {{
                    row.style.display = 'none';
                }}
            }});

            if (rowCount) rowCount.textContent = visible.toString();
        }}

        function toggleAdvancedCols(show) {{
            document.querySelectorAll('.advanced-col').forEach(el => {{
                el.style.display = show ? '' : 'none';
            }});
        }}

        if (searchInput) searchInput.addEventListener('input', applyFilters);
        if (archFilter) archFilter.addEventListener('change', applyFilters);
        if (rowLimit) rowLimit.addEventListener('change', applyFilters);
        if (toggleAdvanced) toggleAdvanced.addEventListener('change', (e) => {{
            toggleAdvancedCols(e.target.checked);
        }});

        toggleAdvancedCols(false);
        applyFilters();
    </script>
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
