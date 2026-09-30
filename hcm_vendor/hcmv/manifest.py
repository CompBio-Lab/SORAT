"""Run manifests: record code version, config and environment next to every output."""

import datetime as _dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .config import config_hash

# Distribution names (not import names). Versions are read from package metadata
# rather than by importing, because importing torch/sklearn from /arc takes minutes.
TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "torch", "SimpleITK",
)


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def package_versions(names=TRACKED_PACKAGES) -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(config: dict, command: str, extra: dict = None) -> dict:
    repo_root = Path(config["paths"]["repo_root"])
    return {
        "command": command,
        "argv": sys.argv,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git(repo_root, "status", "--porcelain", "--", "hcm_vendor", "bin")),
        "config_path": config.get("_config_path"),
        "config_hash": config_hash(config),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": package_versions(),
        **(extra or {}),
    }


def write_manifest(out_dir: Path, config: dict, command: str, extra: dict = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(build_manifest(config, command, extra), indent=2, default=str))
    return path
