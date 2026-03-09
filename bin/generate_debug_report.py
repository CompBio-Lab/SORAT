#!/usr/bin/env python3
"""
Generate debug report for CASC pipeline.

Produces execution, scalability, GPU consumption, robustness, and segmentation-quality
metrics designed to support method validation in manuscript preparation.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    lines.append('# CASC Debug Report')
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
    parser = argparse.ArgumentParser(description='Generate CASC debug analytics report')
    parser.add_argument('--outdir', required=True, help='Pipeline output directory')
    parser.add_argument('--input_samplesheet', required=True, help='Input samplesheet used for run')
    parser.add_argument('--model_selection', required=True, help='Models selected for the run')
    parser.add_argument('--run_name', required=True, help='Nextflow run name')
    parser.add_argument('--workflow_duration', required=True, help='Workflow duration string from Nextflow')
    parser.add_argument('--workflow_start', required=True, help='Workflow start timestamp from Nextflow')
    parser.add_argument('--workflow_success', required=True, help='Workflow success boolean from Nextflow')
    parser.add_argument('--text_dir', required=True, help='Directory to write textual debug outputs')
    parser.add_argument('--figures_dir', required=True, help='Directory to write figure outputs')
    args = parser.parse_args()

    outdir = resolve_path_from_launch_dir(args.outdir)
    trace_path = outdir / 'pipeline_info' / 'execution_trace.txt'
    input_samplesheet = resolve_path_from_launch_dir(args.input_samplesheet)

    text_dir = Path(args.text_dir)
    figures_dir = Path(args.figures_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    trace_df = read_trace_file(trace_path)
    execution = compute_execution_metrics(trace_df, args.workflow_duration, args.workflow_success)
    task_profile = build_task_profile(trace_df)
    gpu_profile = build_gpu_profile(trace_df, execution.get('workflow_duration_seconds', np.nan))

    metrics_df = load_metrics_data(outdir)
    scientific = compute_scientific_metrics(metrics_df, input_samplesheet)

    n_patients_input = scientific.get('n_patients_input', np.nan)
    duration_seconds = execution.get('workflow_duration_seconds', np.nan)
    if isinstance(n_patients_input, (int, np.integer)) and duration_seconds and duration_seconds > 0:
        execution['patient_throughput_per_hour'] = float((n_patients_input * 3600.0) / duration_seconds)

    summary = {
        'run_name': args.run_name,
        'workflow_start': args.workflow_start,
        'models_requested': args.model_selection,
        'execution': execution,
        'gpu': {
            k: v for k, v in gpu_profile.items() if k != 'gpu_process_breakdown'
        },
        'scientific': scientific,
    }

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
        run_name=args.run_name,
        model_selection=args.model_selection,
        workflow_start=args.workflow_start,
        execution=execution,
        gpu_profile=gpu_profile,
        scientific=scientific,
    )

    make_figures(
        task_profile=task_profile,
        gpu_profile=gpu_profile,
        metrics_df=metrics_df,
        figures_dir=figures_dir,
    )

    print(f'Debug summary written to: {text_dir / "debug_metrics_summary.json"}')
    print(f'Debug markdown report written to: {text_dir / "debug_report.md"}')
    print(f'Figures directory: {figures_dir}')


if __name__ == '__main__':
    main()
