#!/usr/bin/env python3
"""
Generate debug report for SORAT pipeline.

Produces execution, scalability, GPU consumption, robustness, and segmentation-quality
metrics designed to support method validation in manuscript preparation.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECTION3_DIRNAME = 'section3_1_proof_of_concept'
SECTION3_PALETTE = {
    'SORAT': '#16324f',
    'Standalone': '#c8553d',
}

SECTION3_MODEL_PATHS = {
    'CineMA': Path('debug/CineMa/text/debug_metrics_summary.json'),
    'nnFormer': Path('debug/nnFormer/text/debug_metrics_summary.json'),
    'VSA-3L': Path('debug/VSA3L/text/debug_metrics_summary.json'),
}


def infer_architecture(model_name: str) -> str:
    if '__' in model_name:
        return model_name.split('__', 1)[0]
    for prefix in ('cinema', 'nnformer', 'vsa3l'):
        if model_name.startswith(prefix):
            return prefix
    return 'unknown'


def parse_duration_to_seconds(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    if text.replace('.', '', 1).isdigit():
        return float(text)

    total = 0.0
    number = ''
    unit_map = {
        'ms': 0.001,
        's': 1.0,
        'm': 60.0,
        'h': 3600.0,
        'd': 86400.0,
    }

    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch.isdigit() or ch == '.':
            number += ch
            idx += 1
            continue

        if ch.isspace():
            idx += 1
            continue

        if not number:
            idx += 1
            continue

        if text[idx:idx + 2] == 'ms':
            unit = 'ms'
            idx += 2
        else:
            unit = ch
            idx += 1

        if unit in unit_map:
            total += float(number) * unit_map[unit]
        number = ''

    if number:
        total += float(number)

    return total if total > 0 else np.nan


def parse_memory_to_gb(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower().replace(' ', '')
    if not text:
        return np.nan

    try:
        return float(text)
    except ValueError:
        pass

    units = {
        'kb': 1 / (1024**2),
        'kib': 1 / (1024**2),
        'mb': 1 / 1024,
        'mib': 1 / 1024,
        'gb': 1.0,
        'gib': 1.0,
        'tb': 1024.0,
        'tib': 1024.0,
        'b': 1 / (1024**3),
    }

    for unit, factor in units.items():
        if text.endswith(unit):
            number = text[:-len(unit)]
            try:
                return float(number) * factor
            except ValueError:
                return np.nan

    return np.nan


def read_trace_file(trace_path: Path) -> pd.DataFrame:
    if not trace_path.exists():
        return pd.DataFrame()

    for sep in ('\t', ',', None):
        try:
            if sep is None:
                df = pd.read_csv(trace_path, sep=None, engine='python')
            else:
                df = pd.read_csv(trace_path, sep=sep)
            if not df.empty:
                return df
        except Exception:
            continue

    return pd.DataFrame()


def resolve_path_from_launch_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    launch_dir = os.environ.get('NXF_LAUNCHDIR')
    if launch_dir:
        return (Path(launch_dir) / path).resolve()

    return path.resolve()


def clean_number(value):
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    return value


def infer_reference_label(path: Path, summary: dict) -> str:
    run_name = str(summary.get('run_name') or '').lower()
    path_text = str(path).lower()
    if 'cinema' in run_name or '/cinema/' in path_text:
        return 'Standalone CineMA'
    if 'nnformer' in run_name or '/nnformer/' in path_text:
        return 'Standalone nnFormer'
    if 'vsa3l' in run_name or '/vsa-3l/' in path_text or '/vsa3l/' in path_text:
        return 'Standalone VSA-3L'
    return f'Reference: {path.parent.parent.parent.name}'


def infer_model_family(text: str) -> str:
    text = str(text).lower()
    if 'cinema' in text:
        return 'CineMA'
    if 'nnformer' in text:
        return 'nnFormer'
    if 'vsa3l' in text or 'vsa-3l' in text:
        return 'VSA-3L'
    return 'Unknown'


def parse_reference_summaries(raw_value: Optional[str]) -> list[dict]:
    if not raw_value:
        return []

    entries = []
    for chunk in str(raw_value).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        label = None
        path_text = chunk
        if '=' in chunk:
            label, path_text = chunk.split('=', 1)
            label = label.strip() or None
        path = resolve_path_from_launch_dir(path_text.strip())
        if path.exists():
            entries.append({'label': label, 'path': path})
    return entries


def flatten_summary_record(label: str, mode: str, path: Path, summary: dict) -> dict:
    execution = summary.get('execution', {})
    gpu = summary.get('gpu', {})
    scientific = summary.get('scientific', {})
    duration_seconds = clean_number(execution.get('workflow_duration_seconds'))
    patient_count = clean_number(scientific.get('n_patients_input'))
    derived_throughput = np.nan
    if pd.notna(duration_seconds) and duration_seconds > 0 and pd.notna(patient_count):
        derived_throughput = float((patient_count * 3600.0) / duration_seconds)

    return {
        'pipeline': label,
        'mode': mode,
        'model_family': infer_model_family(label),
        'source_json': str(path),
        'run_name': summary.get('run_name'),
        'workflow_start': summary.get('workflow_start'),
        'models_requested': summary.get('models_requested'),
        'workflow_success': bool(execution.get('workflow_success', False)),
        'workflow_duration_seconds': duration_seconds,
        'workflow_duration_hours': duration_seconds / 3600.0 if pd.notna(duration_seconds) else np.nan,
        'total_tasks': clean_number(execution.get('total_tasks')),
        'task_success_rate': clean_number(execution.get('task_success_rate')),
        'memory_gb_hours_est': clean_number(execution.get('memory_gb_hours_est')),
        'gpu_task_count': clean_number(gpu.get('gpu_task_count')),
        'gpu_walltime_hours_est': clean_number(gpu.get('gpu_walltime_hours_est')),
        'avg_gpu_concurrency_est': clean_number(gpu.get('avg_gpu_concurrency_est')),
        'patient_throughput_per_hour': clean_number(execution.get('patient_throughput_per_hour')),
        'patient_throughput_per_hour_derived': derived_throughput,
        'n_patients_input': patient_count,
        'n_patients_with_metrics': clean_number(scientific.get('n_patients_with_metrics')),
        'coverage_rate': clean_number(scientific.get('coverage_rate')),
        'n_models_evaluated': clean_number(scientific.get('n_models_evaluated')),
    }


def add_bar_labels(ax, decimals=2):
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height + max(ax.get_ylim()[1] * 0.015, 0.01),
            f'{height:.{decimals}f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )


def infer_gpu_family(process_name: str) -> str:
    upper = str(process_name).upper()
    if upper.startswith('CINEMA_'):
        return 'CineMA'
    if upper.startswith('NNFORMER_'):
        return 'nnFormer'
    if upper.startswith('VSA3L_'):
        return 'VSA-3L'
    return re.sub(r'\s*\(.*\)', '', str(process_name))


def load_source_summary(source_summary_path: Optional[Path]) -> Optional[dict]:
    if not source_summary_path:
        return None
    if not source_summary_path.exists():
        return None
    with open(source_summary_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def merge_execution_blocks(preferred: dict, fallback: dict) -> dict:
    merged = dict(fallback or {})
    for key, value in (preferred or {}).items():
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        merged[key] = value
    return merged


def merge_summary_with_source(current_summary: dict, source_summary: Optional[dict]) -> dict:
    if not source_summary:
        return current_summary

    merged = dict(current_summary)
    merged['run_name'] = source_summary.get('run_name') or current_summary.get('run_name')
    merged['workflow_start'] = source_summary.get('workflow_start') or current_summary.get('workflow_start')
    merged['models_requested'] = source_summary.get('models_requested') or current_summary.get('models_requested')
    merged['execution'] = merge_execution_blocks(source_summary.get('execution', {}), current_summary.get('execution', {}))
    merged['gpu'] = merge_execution_blocks(source_summary.get('gpu', {}), current_summary.get('gpu', {}))
    merged['scientific'] = merge_execution_blocks(source_summary.get('scientific', {}), current_summary.get('scientific', {}))
    merged['source_summary_path'] = source_summary.get('source_summary_path') or current_summary.get('source_summary_path')
    return merged


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def collect_section3_records(source_outdir: Path, reference_specs: list[dict]) -> list[dict]:
    records = []

    for family, relative_path in SECTION3_MODEL_PATHS.items():
        path = source_outdir / relative_path
        if path.exists():
            summary = load_json(path)
            records.append(flatten_summary_record(f'SORAT {family}', 'SORAT', path, summary))

    for spec in reference_specs:
        summary = load_json(spec['path'])
        label = spec['label'] or infer_reference_label(spec['path'], summary)
        records.append(flatten_summary_record(label, 'Standalone', spec['path'], summary))

    return records


def generate_proof_of_concept_artifacts(
    source_outdir: Path,
    section3_outdir: Path,
    sorat_summary: dict,
    sorat_gpu_profile: Dict,
    reference_specs: list[dict],
):
    del sorat_summary
    del sorat_gpu_profile

    if not reference_specs:
        return

    comparison_dir = section3_outdir / 'metrics' / SECTION3_DIRNAME
    comparison_dir.mkdir(parents=True, exist_ok=True)

    records = collect_section3_records(source_outdir, reference_specs)
    if not records:
        return

    frame = pd.DataFrame(records)
    family_order = ['CineMA', 'nnFormer', 'VSA-3L']
    mode_order = ['SORAT', 'Standalone']
    frame['model_family'] = pd.Categorical(frame['model_family'], categories=family_order, ordered=True)
    frame['mode'] = pd.Categorical(frame['mode'], categories=mode_order, ordered=True)
    frame = frame.sort_values(['model_family', 'mode']).reset_index(drop=True)
    frame.to_csv(comparison_dir / 'section3_1_run_comparison.csv', index=False)

    compact_cols = [
        'model_family', 'pipeline', 'mode', 'workflow_duration_hours', 'gpu_walltime_hours_est', 'memory_gb_hours_est', 'gpu_task_count',
        'patient_throughput_per_hour_derived', 'task_success_rate', 'coverage_rate',
        'n_patients_with_metrics', 'n_models_evaluated', 'source_json'
    ]
    frame[compact_cols].to_csv(comparison_dir / 'section3_1_run_comparison_compact.csv', index=False)

    pivot_runtime = frame.pivot(index='model_family', columns='mode', values='workflow_duration_hours').reindex(family_order)
    pivot_throughput = frame.pivot(index='model_family', columns='mode', values='patient_throughput_per_hour_derived').reindex(family_order)
    pivot_gpu = frame.pivot(index='model_family', columns='mode', values='gpu_walltime_hours_est').reindex(family_order)
    pivot_tasks = frame.pivot(index='model_family', columns='mode', values='gpu_task_count').reindex(family_order)
    pivot_memory = frame.pivot(index='model_family', columns='mode', values='memory_gb_hours_est').reindex(family_order)
    pivot_reliability = frame.pivot(index='model_family', columns='mode', values='task_success_rate').reindex(family_order)
    pivot_coverage = frame.pivot(index='model_family', columns='mode', values='coverage_rate').reindex(family_order)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    pivot_runtime.plot(kind='bar', ax=axes[0], color=[SECTION3_PALETTE['SORAT'], SECTION3_PALETTE['Standalone']], width=0.72)
    axes[0].set_title('Wall-clock Runtime by Model Family')
    axes[0].set_ylabel('Hours')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].legend(frameon=False, title='Implementation')

    pivot_throughput.plot(kind='bar', ax=axes[1], color=[SECTION3_PALETTE['SORAT'], SECTION3_PALETTE['Standalone']], width=0.72)
    axes[1].set_title('Patient Throughput by Model Family')
    axes[1].set_ylabel('Patients per hour')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].legend(frameon=False, title='Implementation')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(comparison_dir / 'section3_1_execution_overview.png', dpi=300, bbox_inches='tight')
    fig.savefig(comparison_dir / 'section3_1_execution_overview.svg', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    pivot_gpu.plot(kind='bar', ax=axes[0], color=[SECTION3_PALETTE['SORAT'], SECTION3_PALETTE['Standalone']], width=0.72)
    axes[0].set_title('Estimated GPU Walltime by Model Family')
    axes[0].set_ylabel('GPU-hours')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].legend(frameon=False, title='Implementation')

    pivot_tasks.plot(kind='bar', ax=axes[1], color=[SECTION3_PALETTE['SORAT'], SECTION3_PALETTE['Standalone']], width=0.72)
    axes[1].set_title('GPU-associated Task Count by Model Family')
    axes[1].set_ylabel('Tasks')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].legend(frameon=False, title='Implementation')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(comparison_dir / 'section3_1_resource_usage.png', dpi=300, bbox_inches='tight')
    fig.savefig(comparison_dir / 'section3_1_resource_usage.svg', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    (pivot_reliability * 100.0).plot(kind='bar', ax=axes[0], color=[SECTION3_PALETTE['SORAT'], SECTION3_PALETTE['Standalone']], width=0.72)
    axes[0].set_title('Task Success Rate by Model Family')
    axes[0].set_ylabel('Percent')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].legend(frameon=False, title='Implementation')

    (pivot_coverage * 100.0).plot(kind='bar', ax=axes[1], color=[SECTION3_PALETTE['SORAT'], SECTION3_PALETTE['Standalone']], width=0.72)
    axes[1].set_title('Patient Coverage by Model Family')
    axes[1].set_ylabel('Percent')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].legend(frameon=False, title='Implementation')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linewidth=0.8)
        ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(comparison_dir / 'section3_1_reliability_coverage.png', dpi=300, bbox_inches='tight')
    fig.savefig(comparison_dir / 'section3_1_reliability_coverage.svg', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.2))
    pivot_memory.plot(kind='bar', ax=ax, color=[SECTION3_PALETTE['SORAT'], SECTION3_PALETTE['Standalone']], width=0.72)
    ax.set_title('Estimated Memory Pressure by Model Family')
    ax.set_ylabel('GB-hours from peak RSS')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(frameon=False, title='Implementation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(comparison_dir / 'section3_1_memory_pressure.png', dpi=300, bbox_inches='tight')
    fig.savefig(comparison_dir / 'section3_1_memory_pressure.svg', bbox_inches='tight')
    plt.close(fig)

    lines = [
        '# Section 3.1 Summary Artifacts',
        '',
        'These values are derived directly from the per-model SORAT debug summaries and the standalone pipeline debug summaries.',
        'Section 3.1 focuses on workflow efficiency and resource use, not segmentation Dice comparisons.',
        'Standalone pipeline numbers are contextual cross-run comparisons rather than a synchronized benchmark campaign.',
        '',
    ]
    for family in family_order:
        sub = frame[frame['model_family'] == family]
        if sub.empty:
            continue
        lines.append(f'- {family}:')
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['mode']}: runtime {row['workflow_duration_hours']:.2f} h, "
                f"GPU walltime {row['gpu_walltime_hours_est']:.2f} GPU-hours, "
                f"GPU task count {int(row['gpu_task_count']) if pd.notna(row['gpu_task_count']) else 'NA'}, "
                f"throughput {row['patient_throughput_per_hour_derived']:.2f} patients/hour."
            )
    (comparison_dir / 'section3_1_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def derive_makespan_seconds(trace_df: pd.DataFrame) -> float:
    if trace_df.empty:
        return np.nan

    df = trace_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if 'submit' not in df.columns:
        return np.nan

    duration_col = 'realtime' if 'realtime' in df.columns else ('duration' if 'duration' in df.columns else None)
    if duration_col is None:
        return np.nan

    submit_ts = pd.to_datetime(df['submit'], errors='coerce')
    runtime_s = df[duration_col].map(parse_duration_to_seconds)
    finish_ts = submit_ts + pd.to_timedelta(runtime_s, unit='s')

    if submit_ts.isna().all() or finish_ts.isna().all():
        return np.nan

    start = submit_ts.min()
    end = finish_ts.max()
    if pd.isna(start) or pd.isna(end):
        return np.nan

    makespan = (end - start).total_seconds()
    return float(makespan) if makespan > 0 else np.nan


def safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors='coerce').dropna()
    return float(clean.mean()) if len(clean) else np.nan


def safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors='coerce').dropna()
    return float(clean.median()) if len(clean) else np.nan


def safe_p95(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors='coerce').dropna()
    return float(np.percentile(clean, 95)) if len(clean) else np.nan


def compute_execution_metrics(trace_df: pd.DataFrame, workflow_duration: str, workflow_success: str) -> Dict:
    if trace_df.empty:
        return {
            'workflow_success': str(workflow_success).lower() == 'true',
            'workflow_duration_seconds': parse_duration_to_seconds(workflow_duration),
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'task_success_rate': np.nan,
            'retry_rate': np.nan,
            'cpu_time_hours_est': np.nan,
            'memory_gb_hours_est': np.nan,
            'makespan_seconds_trace': np.nan,
            'mean_parallelism_est': np.nan,
            'patient_throughput_per_hour': np.nan,
        }

    df = trace_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    duration_col = 'realtime' if 'realtime' in df.columns else ('duration' if 'duration' in df.columns else None)
    if duration_col:
        df['runtime_seconds'] = df[duration_col].map(parse_duration_to_seconds)
    else:
        df['runtime_seconds'] = np.nan

    if 'cpus' in df.columns:
        df['cpus_num'] = pd.to_numeric(df['cpus'], errors='coerce')
    else:
        df['cpus_num'] = np.nan

    if 'peak_rss' in df.columns:
        df['peak_rss_gb'] = df['peak_rss'].map(parse_memory_to_gb)
    else:
        df['peak_rss_gb'] = np.nan

    total_tasks = len(df)
    completed_tasks = int((df['status'].astype(str).str.upper() == 'COMPLETED').sum()) if 'status' in df.columns else 0
    failed_tasks = int((df['status'].astype(str).str.upper() == 'FAILED').sum()) if 'status' in df.columns else 0

    attempts = pd.to_numeric(df['attempt'], errors='coerce') if 'attempt' in df.columns else pd.Series(dtype=float)
    retry_count = int((attempts > 1).sum()) if len(attempts) else 0

    workflow_duration_seconds = parse_duration_to_seconds(workflow_duration)
    if np.isnan(workflow_duration_seconds):
        workflow_duration_seconds = derive_makespan_seconds(df)
    runtime_seconds_sum = pd.to_numeric(df['runtime_seconds'], errors='coerce').fillna(0).sum()

    cpu_time_hours = np.nan
    if not df['cpus_num'].isna().all():
        cpu_time_hours = float((df['runtime_seconds'].fillna(0) * df['cpus_num'].fillna(1)).sum() / 3600.0)

    memory_gb_hours = np.nan
    if not df['peak_rss_gb'].isna().all():
        memory_gb_hours = float((df['runtime_seconds'].fillna(0) / 3600.0 * df['peak_rss_gb'].fillna(0)).sum())

    mean_parallelism = np.nan
    if workflow_duration_seconds and workflow_duration_seconds > 0:
        mean_parallelism = float(runtime_seconds_sum / workflow_duration_seconds)

    derived_success = (failed_tasks == 0) if total_tasks > 0 else (str(workflow_success).lower() == 'true')

    return {
        'workflow_success': derived_success,
        'workflow_duration_seconds': workflow_duration_seconds,
        'total_tasks': int(total_tasks),
        'completed_tasks': int(completed_tasks),
        'failed_tasks': int(failed_tasks),
        'task_success_rate': float(completed_tasks / total_tasks) if total_tasks else np.nan,
        'retry_rate': float(retry_count / total_tasks) if total_tasks else np.nan,
        'cpu_time_hours_est': cpu_time_hours,
        'memory_gb_hours_est': memory_gb_hours,
        'makespan_seconds_trace': workflow_duration_seconds,
        'mean_parallelism_est': mean_parallelism,
        'patient_throughput_per_hour': np.nan,
    }


def build_task_profile(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()

    df = trace_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df['process_name'] = df['name'] if 'name' in df.columns else 'unknown'

    duration_col = 'realtime' if 'realtime' in df.columns else ('duration' if 'duration' in df.columns else None)
    if duration_col:
        df['runtime_seconds'] = df[duration_col].map(parse_duration_to_seconds)
    else:
        df['runtime_seconds'] = np.nan

    if '%cpu' in df.columns:
        df['cpu_percent'] = pd.to_numeric(df['%cpu'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    else:
        df['cpu_percent'] = np.nan

    if 'peak_rss' in df.columns:
        df['peak_rss_gb'] = df['peak_rss'].map(parse_memory_to_gb)
    else:
        df['peak_rss_gb'] = np.nan

    profile_rows = []
    for process_name, process_df in df.groupby('process_name'):
        statuses = process_df['status'].astype(str).str.upper() if 'status' in process_df.columns else pd.Series(dtype=str)
        completed = int((statuses == 'COMPLETED').sum()) if len(statuses) else 0
        total = len(process_df)

        profile_rows.append({
            'process': process_name,
            'tasks': int(total),
            'success_rate': float(completed / total) if total else np.nan,
            'runtime_mean_s': safe_mean(process_df['runtime_seconds']),
            'runtime_median_s': safe_median(process_df['runtime_seconds']),
            'runtime_p95_s': safe_p95(process_df['runtime_seconds']),
            'cpu_percent_mean': safe_mean(process_df['cpu_percent']),
            'peak_rss_gb_mean': safe_mean(process_df['peak_rss_gb']),
            'peak_rss_gb_max': float(pd.to_numeric(process_df['peak_rss_gb'], errors='coerce').max()) if len(process_df) else np.nan,
            'runtime_hours_total': float(pd.to_numeric(process_df['runtime_seconds'], errors='coerce').fillna(0).sum() / 3600.0),
        })

    profile = pd.DataFrame(profile_rows)
    if not profile.empty:
        profile = profile.sort_values(by='runtime_hours_total', ascending=False)
    return profile


def build_gpu_profile(trace_df: pd.DataFrame, workflow_duration_seconds: float) -> Dict:
    if trace_df.empty:
        return {
            'gpu_task_count': 0,
            'gpu_task_fraction': np.nan,
            'gpu_walltime_hours_est': 0.0,
            'avg_gpu_concurrency_est': np.nan,
            'gpu_process_breakdown': [],
        }

    df = trace_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    duration_col = 'realtime' if 'realtime' in df.columns else ('duration' if 'duration' in df.columns else None)
    if duration_col:
        df['runtime_seconds'] = df[duration_col].map(parse_duration_to_seconds)
    else:
        df['runtime_seconds'] = np.nan

    label_series = df['label'].astype(str).str.lower() if 'label' in df.columns else pd.Series([''] * len(df))
    name_series = df['name'].astype(str).str.lower() if 'name' in df.columns else pd.Series([''] * len(df))

    gpu_mask = label_series.str.contains('process_gpu', na=False) | name_series.str.contains('segment', na=False)
    gpu_df = df[gpu_mask].copy()

    gpu_task_count = len(gpu_df)
    total_tasks = len(df)
    gpu_walltime_hours = float(pd.to_numeric(gpu_df['runtime_seconds'], errors='coerce').fillna(0).sum() / 3600.0)

    avg_concurrency = np.nan
    if workflow_duration_seconds and workflow_duration_seconds > 0:
        avg_concurrency = float((gpu_walltime_hours * 3600.0) / workflow_duration_seconds)

    breakdown = []
    if gpu_task_count:
        for process_name, process_df in gpu_df.groupby('name' if 'name' in gpu_df.columns else 'status'):
            breakdown.append({
                'process': str(process_name),
                'gpu_tasks': int(len(process_df)),
                'gpu_walltime_hours': float(pd.to_numeric(process_df['runtime_seconds'], errors='coerce').fillna(0).sum() / 3600.0),
                'runtime_median_s': safe_median(process_df['runtime_seconds']),
            })

    return {
        'gpu_task_count': int(gpu_task_count),
        'gpu_task_fraction': float(gpu_task_count / total_tasks) if total_tasks else np.nan,
        'gpu_walltime_hours_est': gpu_walltime_hours,
        'avg_gpu_concurrency_est': avg_concurrency,
        'gpu_process_breakdown': sorted(breakdown, key=lambda x: x['gpu_walltime_hours'], reverse=True),
    }


def load_metrics_data(outdir: Path) -> pd.DataFrame:
    comparison_agg = outdir / 'comparison' / 'aggregated_metrics.csv'
    if comparison_agg.exists():
        try:
            return pd.read_csv(comparison_agg)
        except Exception:
            pass

    metrics_files = list((outdir / 'metrics').rglob('*_metrics.csv'))
    dfs = []
    for fp in metrics_files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def compute_scientific_metrics(metrics_df: pd.DataFrame, input_samplesheet: Path) -> Dict:
    summary = {
        'n_patients_input': np.nan,
        'n_patients_with_metrics': 0,
        'coverage_rate': np.nan,
        'n_models_evaluated': 0,
        'best_model_overall_dice': None,
        'best_model_dice_value': np.nan,
        'cross_model_dice_spread': np.nan,
        'architecture_consistency_cv_mean': np.nan,
        'rv_dice_mean': np.nan,
        'myo_dice_mean': np.nan,
        'lv_dice_mean': np.nan,
    }

    if input_samplesheet.exists():
        try:
            input_df = pd.read_csv(input_samplesheet)
            summary['n_patients_input'] = int(input_df['patient_id'].nunique()) if 'patient_id' in input_df.columns else len(input_df)
        except Exception:
            pass

    if metrics_df.empty:
        return summary

    df = metrics_df.copy()
    if 'architecture' not in df.columns and 'model' in df.columns:
        df['architecture'] = df['model'].astype(str).map(infer_architecture)

    if 'patient_id' in df.columns:
        summary['n_patients_with_metrics'] = int(df['patient_id'].nunique())

    if isinstance(summary['n_patients_input'], (int, np.integer)) and summary['n_patients_input'] > 0:
        summary['coverage_rate'] = float(summary['n_patients_with_metrics'] / summary['n_patients_input'])

    if 'model' in df.columns:
        summary['n_models_evaluated'] = int(df['model'].nunique())

    if 'overall_dice_mean' in df.columns and 'model' in df.columns:
        model_perf = df.groupby('model', as_index=False)['overall_dice_mean'].mean().sort_values('overall_dice_mean', ascending=False)
        if not model_perf.empty:
            best_row = model_perf.iloc[0]
            summary['best_model_overall_dice'] = str(best_row['model'])
            summary['best_model_dice_value'] = float(best_row['overall_dice_mean'])
            if len(model_perf) > 1:
                summary['cross_model_dice_spread'] = float(model_perf['overall_dice_mean'].max() - model_perf['overall_dice_mean'].min())

    if 'overall_dice_mean' in df.columns and 'architecture' in df.columns:
        cvs = []
        for arch, sub in df.groupby('architecture'):
            values = pd.to_numeric(sub['overall_dice_mean'], errors='coerce').dropna()
            if len(values) >= 2 and values.mean() > 0:
                cvs.append(float(values.std() / values.mean()))
        if cvs:
            summary['architecture_consistency_cv_mean'] = float(np.mean(cvs))

    for structure in ('rv', 'myo', 'lv'):
        col_candidates = [f'ed_dice_{structure}', f'es_dice_{structure}', f'dice_{structure}']
        vals = []
        for col in col_candidates:
            if col in df.columns:
                vals.extend(pd.to_numeric(df[col], errors='coerce').dropna().tolist())
        if vals:
            summary[f'{structure}_dice_mean'] = float(np.mean(vals))

    return summary


def make_figures(
    task_profile: pd.DataFrame,
    gpu_profile: Dict,
    metrics_df: pd.DataFrame,
    figures_dir: Path,
):
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not task_profile.empty:
        top = task_profile.head(12)
        plt.figure(figsize=(12, 6))
        plt.bar(top['process'], top['runtime_hours_total'], color='#4e79a7')
        plt.title('Top Process Runtime Contribution')
        plt.ylabel('Runtime (hours)')
        plt.xlabel('Process')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figures_dir / 'runtime_by_process.png', dpi=160)
        plt.close()

    breakdown = gpu_profile.get('gpu_process_breakdown', [])
    if breakdown:
        gpu_df = pd.DataFrame(breakdown)
        plt.figure(figsize=(10, 5))
        plt.bar(gpu_df['process'], gpu_df['gpu_walltime_hours'], color='#f28e2b')
        plt.title('Estimated GPU Walltime by Process')
        plt.ylabel('GPU walltime (hours)')
        plt.xlabel('Process')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figures_dir / 'gpu_walltime_by_process.png', dpi=160)
        plt.close()

    if not metrics_df.empty and 'overall_dice_mean' in metrics_df.columns and 'model' in metrics_df.columns:
        model_perf = metrics_df.groupby('model', as_index=False)['overall_dice_mean'].mean().sort_values('overall_dice_mean', ascending=False)
        plt.figure(figsize=(10, 5))
        plt.bar(model_perf['model'], model_perf['overall_dice_mean'], color='#59a14f')
        plt.title('Mean Overall Dice by Model')
        plt.ylabel('Overall Dice')
        plt.xlabel('Model')
        plt.ylim(0, 1)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(figures_dir / 'overall_dice_by_model.png', dpi=160)
        plt.close()


def write_markdown_report(
    output_path: Path,
    run_name: str,
    model_selection: str,
    workflow_start: str,
    execution: Dict,
    gpu_profile: Dict,
    scientific: Dict,
):
    lines = []
    lines.append('# SORAT Debug Report')
    lines.append('')
    lines.append('## Run Context')
    lines.append(f'- Run name: {run_name}')
    lines.append(f'- Models requested: {model_selection}')
    lines.append(f'- Workflow start: {workflow_start}')
    lines.append(f'- Workflow success: {execution.get("workflow_success")}')
    lines.append('')

    lines.append('## Execution & Reliability')
    lines.append(f'- Workflow duration (s): {execution.get("workflow_duration_seconds")}')
    lines.append(f'- Total tasks: {execution.get("total_tasks")}')
    lines.append(f'- Task success rate: {execution.get("task_success_rate"):.4f}' if not math.isnan(execution.get('task_success_rate', np.nan)) else '- Task success rate: NA')
    lines.append(f'- Retry rate: {execution.get("retry_rate"):.4f}' if not math.isnan(execution.get('retry_rate', np.nan)) else '- Retry rate: NA')
    lines.append(f'- Estimated CPU time (core-hours): {execution.get("cpu_time_hours_est"):.3f}' if not math.isnan(execution.get('cpu_time_hours_est', np.nan)) else '- Estimated CPU time (core-hours): NA')
    lines.append(f'- Estimated memory pressure (GB-hours from peak RSS): {execution.get("memory_gb_hours_est"):.3f}' if not math.isnan(execution.get('memory_gb_hours_est', np.nan)) else '- Estimated memory pressure (GB-hours from peak RSS): NA')
    lines.append(f'- Mean parallelism estimate: {execution.get("mean_parallelism_est"):.3f}' if not math.isnan(execution.get('mean_parallelism_est', np.nan)) else '- Mean parallelism estimate: NA')
    throughput = execution.get('patient_throughput_per_hour', np.nan)
    lines.append(f'- Patient throughput (patients/hour): {throughput:.3f}' if not math.isnan(throughput) else '- Patient throughput (patients/hour): NA')
    lines.append('')

    lines.append('## GPU Scalability')
    lines.append(f'- GPU task count: {gpu_profile.get("gpu_task_count")}')
    gpu_fraction = gpu_profile.get('gpu_task_fraction', np.nan)
    lines.append(f'- GPU task fraction: {gpu_fraction:.4f}' if not math.isnan(gpu_fraction) else '- GPU task fraction: NA')
    lines.append(f'- Estimated GPU walltime (hours): {gpu_profile.get("gpu_walltime_hours_est"):.3f}')
    avg_gpu = gpu_profile.get('avg_gpu_concurrency_est', np.nan)
    lines.append(f'- Average GPU concurrency estimate: {avg_gpu:.3f}' if not math.isnan(avg_gpu) else '- Average GPU concurrency estimate: NA')
    lines.append('')

    lines.append('## Scientific Utility')
    lines.append(f'- Input patient count: {scientific.get("n_patients_input")}')
    lines.append(f'- Evaluated patient count: {scientific.get("n_patients_with_metrics")}')
    coverage = scientific.get('coverage_rate', np.nan)
    lines.append(f'- Coverage rate: {coverage:.4f}' if not math.isnan(coverage) else '- Coverage rate: NA')
    lines.append(f'- Models evaluated: {scientific.get("n_models_evaluated")}')
    lines.append(f'- Best model by mean overall Dice: {scientific.get("best_model_overall_dice")} ({scientific.get("best_model_dice_value"):.4f})' if not math.isnan(scientific.get('best_model_dice_value', np.nan)) else '- Best model by mean overall Dice: NA')
    spread = scientific.get('cross_model_dice_spread', np.nan)
    lines.append(f'- Cross-model Dice spread: {spread:.4f}' if not math.isnan(spread) else '- Cross-model Dice spread: NA')
    consistency = scientific.get('architecture_consistency_cv_mean', np.nan)
    lines.append(f'- Architecture consistency CV (lower is better): {consistency:.4f}' if not math.isnan(consistency) else '- Architecture consistency CV (lower is better): NA')
    for structure in ('rv', 'myo', 'lv'):
        val = scientific.get(f'{structure}_dice_mean', np.nan)
        lines.append(f'- Mean {structure.upper()} Dice: {val:.4f}' if not math.isnan(val) else f'- Mean {structure.upper()} Dice: NA')

    output_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate SORAT debug analytics report')
    parser.add_argument('--source_outdir', required=True, help='Results directory to read metrics, pipeline_info, and existing debug artifacts from')
    parser.add_argument('--input_samplesheet', required=True, help='Input samplesheet used for run')
    parser.add_argument('--model_selection', required=True, help='Models selected for the run')
    parser.add_argument('--run_name', required=True, help='Nextflow run name')
    parser.add_argument('--workflow_duration', required=True, help='Workflow duration string from Nextflow')
    parser.add_argument('--workflow_start', required=True, help='Workflow start timestamp from Nextflow')
    parser.add_argument('--workflow_success', required=True, help='Workflow success boolean from Nextflow')
    parser.add_argument('--text_dir', required=True, help='Directory to write textual debug outputs')
    parser.add_argument('--figures_dir', required=True, help='Directory to write figure outputs')
    parser.add_argument('--source_summary', default='', help='Optional existing SORAT debug summary JSON to use as the authoritative integrated execution source')
    parser.add_argument('--section3_outdir', default='', help='Optional output directory root for Section 3.1 comparison artifacts; defaults to source_outdir')
    parser.add_argument('--reference_summaries', default='', help='Optional comma-separated list of label=path or path debug summary JSON references for Section 3.1 comparison artifacts')
    args = parser.parse_args()

    source_outdir = resolve_path_from_launch_dir(args.source_outdir)
    trace_path = source_outdir / 'pipeline_info' / 'execution_trace.txt'
    input_samplesheet = resolve_path_from_launch_dir(args.input_samplesheet)
    source_summary_path = resolve_path_from_launch_dir(args.source_summary) if args.source_summary else None
    section3_outdir = resolve_path_from_launch_dir(args.section3_outdir) if args.section3_outdir else source_outdir

    text_dir = Path(args.text_dir)
    figures_dir = Path(args.figures_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    trace_df = read_trace_file(trace_path)
    execution = compute_execution_metrics(trace_df, args.workflow_duration, args.workflow_success)
    task_profile = build_task_profile(trace_df)
    gpu_profile = build_gpu_profile(trace_df, execution.get('workflow_duration_seconds', np.nan))

    metrics_df = load_metrics_data(source_outdir)
    scientific = compute_scientific_metrics(metrics_df, input_samplesheet)

    n_patients_input = scientific.get('n_patients_input', np.nan)
    duration_seconds = execution.get('workflow_duration_seconds', np.nan)
    if isinstance(n_patients_input, (int, np.integer)) and duration_seconds and duration_seconds > 0:
        execution['patient_throughput_per_hour'] = float((n_patients_input * 3600.0) / duration_seconds)

    summary = {
        'run_name': args.run_name,
        'workflow_start': args.workflow_start,
        'models_requested': args.model_selection,
        'source_outdir': str(source_outdir),
        'source_summary_path': str(source_summary_path) if source_summary_path else None,
        'execution': execution,
        'gpu': {
            k: v for k, v in gpu_profile.items() if k != 'gpu_process_breakdown'
        },
        'scientific': scientific,
    }

    source_summary = load_source_summary(source_summary_path)
    summary = merge_summary_with_source(summary, source_summary)

    merged_execution = summary.get('execution', {})
    merged_gpu = summary.get('gpu', {})
    merged_scientific = summary.get('scientific', {})
    if math.isnan(merged_execution.get('patient_throughput_per_hour', np.nan)):
        n_patients_input = merged_scientific.get('n_patients_input', np.nan)
        duration_seconds = merged_execution.get('workflow_duration_seconds', np.nan)
        if isinstance(n_patients_input, (int, np.integer)) and duration_seconds and duration_seconds > 0:
            merged_execution['patient_throughput_per_hour'] = float((n_patients_input * 3600.0) / duration_seconds)
            summary['execution'] = merged_execution

    with open(text_dir / 'debug_metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if not task_profile.empty:
        task_profile.to_csv(text_dir / 'task_profile.csv', index=False)

    gpu_breakdown = pd.DataFrame(gpu_profile.get('gpu_process_breakdown', []))
    if not gpu_breakdown.empty:
        gpu_breakdown.to_csv(text_dir / 'gpu_profile.csv', index=False)

    if not metrics_df.empty:
        metrics_df.to_csv(text_dir / 'metrics_snapshot.csv', index=False)

    write_markdown_report(
        output_path=text_dir / 'debug_report.md',
        run_name=summary.get('run_name', args.run_name),
        model_selection=summary.get('models_requested', args.model_selection),
        workflow_start=summary.get('workflow_start', args.workflow_start),
        execution=summary.get('execution', {}),
        gpu_profile=summary.get('gpu', {}),
        scientific=summary.get('scientific', {}),
    )

    make_figures(
        task_profile=task_profile,
        gpu_profile=gpu_profile,
        metrics_df=metrics_df,
        figures_dir=figures_dir,
    )

    reference_specs = parse_reference_summaries(args.reference_summaries)
    generate_proof_of_concept_artifacts(
        source_outdir=source_outdir,
        section3_outdir=section3_outdir,
        sorat_summary=summary,
        sorat_gpu_profile=merged_gpu,
        reference_specs=reference_specs,
    )

    print(f'Debug summary written to: {text_dir / "debug_metrics_summary.json"}')
    print(f'Debug markdown report written to: {text_dir / "debug_report.md"}')
    print(f'Figures directory: {figures_dir}')


if __name__ == '__main__':
    main()
