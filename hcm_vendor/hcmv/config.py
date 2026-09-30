"""Study configuration: YAML loading, dotted-key overrides, stable hashing."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "study.yaml"


def _parse_scalar(text: str) -> Any:
    """Parse an override value with YAML rules (numbers, bools, lists, null)."""
    return yaml.safe_load(text)


def apply_overrides(config: dict, overrides: Iterable[str]) -> dict:
    """Return a copy of ``config`` with ``key.sub=value`` overrides applied."""
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        dotted, raw_value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = result
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = _parse_scalar(raw_value)
    return result


def load_config(path: Optional[Path] = None, overrides: Iterable[str] = ()) -> dict:
    """Load the study YAML (default ``configs/study.yaml``) and apply overrides."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}
    config = apply_overrides(config, overrides)
    config["_config_path"] = str(config_path)
    return config


def config_hash(config: dict, length: int = 12) -> str:
    """Stable short hash of the config, ignoring private (underscore) keys."""
    public = {k: v for k, v in config.items() if not str(k).startswith("_")}
    canonical = json.dumps(public, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def repo_path(config: dict, value: str) -> Path:
    """Resolve a config path relative to ``paths.repo_root`` unless absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["paths"]["repo_root"]) / path


def output_dir(config: dict, *parts: str) -> Path:
    """Directory under the study output root, created on demand."""
    path = repo_path(config, config["paths"]["output_root"]).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
