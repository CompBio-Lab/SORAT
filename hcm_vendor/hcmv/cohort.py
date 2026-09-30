"""Study cohort (T20) and pipeline samplesheets (T13).

The cohort table has one row per subject with disease label, vendor and the
subject's role in the experiments:

- ``train_pool``: M&Ms-2 Siemens + Philips NOR/HCM (E1-E4)
- ``ge_check``: M&Ms-2 GE NOR/HCM (specificity check, E3 probe)
- ``external``: ACDC test-split NOR/HCM (E5)
"""

from pathlib import Path

import pandas as pd

from .config import output_dir, repo_path

COHORT_COLUMNS = [
    "subject_id", "dataset", "source_id", "disease", "y", "vendor", "scanner",
    "field_T", "role", "mms2_challenge_split",
]
SAMPLESHEET_COLUMNS = ["patient_id", "image", "ground_truth", "info_cfg"]


def mms2_challenge_split(code: int) -> str:
    """M&Ms-2 readme: subjects 1-160 train, 161-200 validation, 201-360 test."""
    if code <= 160:
        return "train"
    if code <= 200:
        return "val"
    return "test"


def load_mms2_info(info_csv) -> pd.DataFrame:
    """Read dataset_information.csv, dropping the ~1M empty Excel padding rows."""
    info = pd.read_csv(info_csv, low_memory=False).dropna(how="all")
    info = info.dropna(subset=["SUBJECT_CODE"]).copy()
    info["SUBJECT_CODE"] = info["SUBJECT_CODE"].astype(int)
    if info["SUBJECT_CODE"].duplicated().any():
        raise ValueError("Duplicate SUBJECT_CODE values in M&Ms-2 dataset information")
    return info


def build_mms2_cohort(info: pd.DataFrame, config: dict) -> pd.DataFrame:
    cohort_cfg = config["cohort"]
    vendor_map = cohort_cfg["vendor_map"]
    subset = info[info["DISEASE"].isin(cohort_cfg["diseases"])].copy()

    unknown = sorted(set(subset["VENDOR"]) - set(vendor_map))
    if unknown:
        raise ValueError(f"Unmapped M&Ms-2 vendor names: {unknown}")

    vendor = subset["VENDOR"].map(vendor_map)
    source_id = subset["SUBJECT_CODE"].map("{:03d}".format)
    return pd.DataFrame({
        "subject_id": "mms2_" + source_id,
        "dataset": "mms2",
        "source_id": source_id,
        "disease": subset["DISEASE"].values,
        "y": (subset["DISEASE"] == cohort_cfg["positive_class"]).astype(int).values,
        "vendor": vendor.values,
        "scanner": subset["SCANNER"].values,
        "field_T": subset["FIELD"].astype(float).values,
        "role": ["train_pool" if v in cohort_cfg["train_vendors"] else "ge_check" for v in vendor],
        "mms2_challenge_split": subset["SUBJECT_CODE"].map(mms2_challenge_split).values,
    })


def read_info_cfg(path) -> dict:
    """Parse an ACDC-style ``Key: value`` Info.cfg."""
    values = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def build_acdc_cohort(testing_root, config: dict) -> pd.DataFrame:
    cohort_cfg = config["cohort"]
    rows = []
    for patient_dir in sorted(Path(testing_root).glob("patient*")):
        info_path = patient_dir / "Info.cfg"
        if not patient_dir.is_dir() or not info_path.exists():
            continue
        group = read_info_cfg(info_path).get("Group")
        if group not in cohort_cfg["diseases"]:
            continue
        rows.append({
            "subject_id": f"acdc_{patient_dir.name}",
            "dataset": "acdc",
            "source_id": patient_dir.name,
            "disease": group,
            "y": int(group == cohort_cfg["positive_class"]),
            "vendor": "Siemens",
            "scanner": None,
            "field_T": float("nan"),  # ACDC mixes 1.5 T and 3 T; not recorded per subject
            "role": "external",
            "mms2_challenge_split": None,
        })
    return pd.DataFrame(rows, columns=COHORT_COLUMNS)


def check_expected_counts(cohort: pd.DataFrame, expected: dict) -> None:
    """Raise if disease x vendor counts differ from ``cohort.expected_counts``."""
    problems = []
    for dataset, by_vendor in expected.items():
        subset = cohort[cohort["dataset"] == dataset]
        observed = subset.groupby(["vendor", "disease"]).size().to_dict()
        wanted = {(v, d): n for v, diseases in by_vendor.items() for d, n in diseases.items()}
        for key in sorted(set(observed) | set(wanted)):
            if observed.get(key, 0) != wanted.get(key, 0):
                problems.append(f"{dataset} {key}: expected {wanted.get(key, 0)}, got {observed.get(key, 0)}")
    if problems:
        raise ValueError("Cohort counts do not match configuration:\n  " + "\n  ".join(problems))


def build_cohort(config: dict) -> pd.DataFrame:
    paths = config["paths"]
    mms2 = build_mms2_cohort(load_mms2_info(repo_path(config, paths["mms2_info_csv"])), config)
    acdc = build_acdc_cohort(repo_path(config, paths["acdc_testing_root"]), config)
    cohort = pd.concat([mms2, acdc], ignore_index=True)[COHORT_COLUMNS]
    if cohort["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject_id values in cohort")
    check_expected_counts(cohort, config["cohort"]["expected_counts"])
    return cohort


def samplesheet_rows(cohort: pd.DataFrame, config: dict) -> dict:
    """SORAT samplesheet rows per dataset, built from the dataset folder layouts."""
    paths = config["paths"]
    mms2_root = repo_path(config, paths["mms2_sa_root"])
    acdc_root = repo_path(config, paths["acdc_testing_root"])
    sheets = {}
    for dataset, group in cohort.groupby("dataset"):
        rows = []
        for source_id in sorted(group["source_id"]):
            if dataset == "mms2":
                folder = mms2_root / source_id
                image = folder / f"{source_id}_SA_CINE.nii.gz"
            else:
                folder = acdc_root / source_id
                image = folder / f"{source_id}_4d.nii.gz"
            rows.append({
                "patient_id": source_id,
                "image": str(image),
                "ground_truth": str(folder),
                "info_cfg": str(folder / "Info.cfg"),
            })
        sheets[dataset] = pd.DataFrame(rows, columns=SAMPLESHEET_COLUMNS)
    return sheets


def check_samplesheet_files(sheet: pd.DataFrame) -> list:
    """Return paths in the samplesheet that do not exist."""
    missing = []
    for column in ("image", "ground_truth", "info_cfg"):
        missing.extend(p for p in sheet[column] if not Path(p).exists())
    return missing


def cross_check_reference(sheet: pd.DataFrame, reference_csv) -> list:
    """Compare generated rows with an existing SORAT samplesheet, if present."""
    reference_csv = Path(reference_csv)
    if not reference_csv.exists():
        return [f"reference samplesheet not found: {reference_csv}"]
    reference = pd.read_csv(reference_csv, dtype=str).set_index("patient_id")
    problems = []
    for row in sheet.itertuples(index=False):
        if row.patient_id not in reference.index:
            problems.append(f"{row.patient_id} missing from {reference_csv.name}")
            continue
        ref = reference.loc[row.patient_id]
        for column in ("image", "ground_truth", "info_cfg"):
            if str(ref[column]) != getattr(row, column):
                problems.append(f"{row.patient_id} {column}: {ref[column]} != {getattr(row, column)}")
    return problems


def write_cohort(config: dict) -> pd.DataFrame:
    cohort = build_cohort(config)
    out = output_dir(config, "tables")
    cohort.to_csv(out / "cohort.csv", index=False)
    cohort.to_parquet(out / "cohort.parquet", index=False)
    return cohort


def write_samplesheets(config: dict, cohort: pd.DataFrame) -> dict:
    out = output_dir(config, "inputs")
    references = {"mms2": config["paths"]["mms2_samplesheet"], "acdc": config["paths"]["acdc_samplesheet"]}
    written = {}
    for dataset, sheet in samplesheet_rows(cohort, config).items():
        missing = check_samplesheet_files(sheet)
        if missing:
            raise FileNotFoundError(f"{dataset}: {len(missing)} missing input paths, e.g. {missing[:3]}")
        problems = cross_check_reference(sheet, repo_path(config, references[dataset]))
        if problems:
            raise ValueError(f"{dataset} samplesheet disagrees with reference:\n  " + "\n  ".join(problems[:10]))
        path = out / f"{dataset}_nor_hcm.csv"
        sheet.to_csv(path, index=False)
        written[dataset] = path
    ids = cohort[["subject_id", "dataset", "source_id", "disease", "vendor", "role"]]
    ids.to_csv(out / "cohort_ids.csv", index=False)
    return written
