"""Feature tables (T21): per-phase SORAT CSVs -> one row per subject with families.

Per-phase CSVs are named ``<pid>_<source>_<ED|ES>_features.csv`` where ``source``
is a SORAT model tag (e.g. ``nnformer__fold0``) or ``gt``. Each holds one row with
columns prefixed ``ed_``/``es_`` plus a trailing unprefixed ``myocardial_mass_g``.
The ``patient_id`` column is not a bare ID, so IDs are taken from filenames.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import output_dir, repo_path

PHASES = ("ED", "ES")
META_SUFFIXES = (
    "patient_id", "mask_file", "mask_source", "voxel_volume_ml", "radiomics_settings",
    "radiomics_error",
)
CLINICAL_BASE = (
    "lv_volume_ml", "rv_volume_ml", "myo_volume_ml", "myocardial_mass_g",
    "wall_thickness_mean_mm", "wall_thickness_max_mm",
)
SENSITIVITY_BASE = ("wall_thickness_p95_mm",)
DERIVED = ("lv_sv_ml", "rv_sv_ml", "lvef_pct", "rvef_pct", "mass_to_volume_g_per_ml")
FAMILY_PATTERNS = {
    "shape": re.compile(r"^(ed|es)_radiomics_original_shape_"),
    "texture": re.compile(r"^(ed|es)_radiomics_original_(firstorder|glcm)_"),
}


def parse_feature_filename(name: str):
    """Split ``<pid>_<source>_<PHASE>_features.csv`` into (pid, source, phase)."""
    if not name.endswith("_features.csv"):
        raise ValueError(f"Not a feature CSV: {name}")
    stem = name[: -len("_features.csv")]
    rest, phase = stem.rsplit("_", 1)
    pid, source = rest.split("_", 1)  # study IDs (001, patient101) contain no underscores
    if phase not in PHASES:
        raise ValueError(f"Unexpected phase {phase!r} in {name}")
    return pid, source, phase


def read_phase_csvs(directory, source: str) -> pd.DataFrame:
    """Read all ``*_<source>_{ED,ES}_features.csv`` into a long table.

    Returns one row per (source_id, phase) with unprefixed feature columns and a
    ``radiomics_error`` column (NaN when extraction succeeded).
    """
    rows = []
    for path in sorted(Path(directory).glob(f"*_{source}_*_features.csv")):
        pid, file_source, phase = parse_feature_filename(path.name)
        if file_source != source:
            continue
        if path.stat().st_size == 0:
            raise ValueError(f"Empty feature file: {path}")
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"Expected one row in {path}, found {len(frame)}")
        prefix = f"{phase.lower()}_"
        record = {
            col[len(prefix):]: frame.iloc[0][col] for col in frame.columns if col.startswith(prefix)
        }
        record.update({"source_id": pid, "phase": phase, "file": str(path)})
        rows.append(record)
    if not rows:
        raise FileNotFoundError(f"No *_{source}_*_features.csv files in {directory}")
    long = pd.DataFrame(rows)
    if "radiomics_error" not in long.columns:
        long["radiomics_error"] = np.nan
    return long


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Merge ED and ES rows into one row per subject with ed_/es_ prefixes."""
    counts = long.groupby("source_id")["phase"].nunique()
    incomplete = counts[counts != len(PHASES)]
    if len(incomplete):
        raise ValueError(f"Subjects missing a phase: {sorted(incomplete.index)[:10]}")
    feature_cols = [c for c in long.columns if c not in ("source_id", "phase", "file")]
    parts = []
    for phase in PHASES:
        part = long[long["phase"] == phase].set_index("source_id")[feature_cols]
        parts.append(part.add_prefix(f"{phase.lower()}_"))
    return pd.concat(parts, axis=1).sort_index()


def add_derived(wide: pd.DataFrame) -> pd.DataFrame:
    """Stroke volumes, ejection fractions and ED mass-to-volume ratio."""
    out = wide.copy()

    def ratio(num, den):
        den = den.astype(float)
        return (num.astype(float) / den).where(den > 0)

    out["lv_sv_ml"] = out["ed_lv_volume_ml"] - out["es_lv_volume_ml"]
    out["rv_sv_ml"] = out["ed_rv_volume_ml"] - out["es_rv_volume_ml"]
    out["lvef_pct"] = 100.0 * ratio(out["lv_sv_ml"], out["ed_lv_volume_ml"])
    out["rvef_pct"] = 100.0 * ratio(out["rv_sv_ml"], out["ed_rv_volume_ml"])
    out["mass_to_volume_g_per_ml"] = ratio(out["ed_myocardial_mass_g"], out["ed_lv_volume_ml"])
    return out


def feature_family(column: str):
    """Family of a model feature column, or None for non-model columns."""
    for phase in ("ed_", "es_"):
        if column.startswith(phase):
            base = column[len(phase):]
            if base in CLINICAL_BASE:
                return "clinical"
            if base in SENSITIVITY_BASE:
                return "sensitivity"
    if column in DERIVED:
        return "clinical"
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.match(column):
            return family
    return None


def prune_columns(wide: pd.DataFrame):
    """Drop meta, constant and exact-duplicate feature columns. Returns (table, log)."""
    log = []
    drop = [c for c in wide.columns if c.split("_", 1)[-1] in META_SUFFIXES or c in META_SUFFIXES]
    log += [(c, "meta") for c in drop]
    table = wide.drop(columns=drop)

    model_cols = [c for c in table.columns if feature_family(c) is not None]
    constant = [c for c in model_cols if table[c].nunique(dropna=True) <= 1]
    log += [(c, "constant") for c in constant]
    table = table.drop(columns=constant)

    seen = {}
    duplicates = []
    for col in [c for c in table.columns if feature_family(c) is not None]:
        key = tuple(np.round(table[col].to_numpy(dtype=float), 10))
        if key in seen:
            duplicates.append(col)
            log.append((col, f"duplicate_of:{seen[key]}"))
        else:
            seen[key] = col
    table = table.drop(columns=duplicates)
    return table, pd.DataFrame(log, columns=["column", "reason"])


def data_dictionary(columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        family = feature_family(col)
        if family is None:
            continue
        phase = col[:2].upper() if col[:3] in ("ed_", "es_") else "ED+ES"
        unit = next((u for u in ("_ml", "_mm", "_g", "_pct", "_g_per_ml") if col.endswith(u)), "")
        rows.append({"feature": col, "family": family, "phase": phase, "unit": unit.lstrip("_")})
    return pd.DataFrame(rows)


def build_feature_table(directory, source: str, cohort: pd.DataFrame, dataset: str):
    """Build the modelling table for one dataset/source/config directory.

    Returns (table, prune_log). ``table`` is indexed by ``subject_id`` and holds
    cohort columns followed by model features; subjects not in the cohort are
    dropped, and cohort subjects without features raise.
    """
    long = read_phase_csvs(directory, source)
    errors = long[long["radiomics_error"].notna()]
    if len(errors):
        raise ValueError(f"radiomics_error in {len(errors)} rows, e.g. {errors['file'].iloc[0]}")

    wide = add_derived(to_wide(long.drop(columns=["radiomics_error"])))
    wide = wide.drop(columns=[c for c in ("myocardial_mass_g",) if c in wide.columns])
    table, log = prune_columns(wide)

    subset = cohort[cohort["dataset"] == dataset].set_index("source_id")
    missing = sorted(set(subset.index) - set(table.index))
    if missing:
        raise ValueError(f"{len(missing)} cohort subjects lack features, e.g. {missing[:5]}")
    table = subset.join(table, how="left").reset_index().set_index("subject_id")
    return table, log


def feature_columns(table: pd.DataFrame, families) -> list:
    """Model feature columns belonging to any of ``families``."""
    families = set(families)
    return [c for c in table.columns if feature_family(c) in families]


def write_feature_tables(config: dict, cohort: pd.DataFrame, source: str, root: str,
                         configs=("norm", "raw"), datasets=("mms2", "acdc")) -> dict:
    """Build and save tables for every dataset x feature config under ``root``."""
    out = output_dir(config, "tables")
    written = {}
    for dataset in datasets:
        for cfg in configs:
            directory = repo_path(config, root) / dataset / cfg
            table, log = build_feature_table(directory, source, cohort, dataset)
            name = f"features_{dataset}_{source.replace('__', '-')}_{cfg}"
            table.to_parquet(out / f"{name}.parquet")
            log.to_csv(out / f"{name}_pruned.csv", index=False)
            data_dictionary(table.columns).to_csv(out / f"{name}_dictionary.csv", index=False)
            written[(dataset, cfg)] = out / f"{name}.parquet"
    return written
