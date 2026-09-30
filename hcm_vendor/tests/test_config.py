import json

import pytest

from hcmv.config import apply_overrides, config_hash, load_config
from hcmv.manifest import write_manifest


def test_default_config_loads_expected_sections():
    config = load_config()
    assert config["cohort"]["train_vendors"] == ["Siemens", "Philips"]
    assert config["cv"] == {"outer_splits": 5, "outer_repeats": 5, "inner_splits": 5}


def test_overrides_parse_yaml_scalars_and_create_nested_keys():
    config = apply_overrides({"cv": {"outer_repeats": 5}}, ["cv.outer_repeats=1", "new.flag=true"])
    assert config["cv"]["outer_repeats"] == 1
    assert config["new"]["flag"] is True


def test_override_without_equals_is_rejected():
    with pytest.raises(ValueError):
        apply_overrides({}, ["cv.outer_repeats"])


def test_config_hash_ignores_private_keys_and_tracks_changes():
    base = {"a": 1, "_config_path": "/x"}
    assert config_hash(base) == config_hash({"a": 1, "_config_path": "/y"})
    assert config_hash(base) != config_hash({"a": 2})


def test_manifest_records_config_hash(tmp_path):
    config = load_config(overrides=[f"paths.output_root={tmp_path}"])
    path = write_manifest(tmp_path / "run", config, "unit-test")
    manifest = json.loads(path.read_text())
    assert manifest["command"] == "unit-test"
    assert manifest["config_hash"] == config_hash(config)
    assert manifest["packages"]["python"]


def test_radiomics_cli_flags_from_study_config():
    from hcmv.__main__ import radiomics_cli_flags

    config = load_config()
    assert radiomics_cli_flags(config["feature_configs"]["raw"]) == []
    flags = radiomics_cli_flags(config["feature_configs"]["norm"])
    assert "--feature_extraction.radiomics.normalize true" in flags
    assert "--feature_extraction.radiomics.bin_count 32" in flags
    assert "--feature_extraction.radiomics.resample_spacing 1.25,1.25,0" in flags
    assert "--feature_extraction.radiomics.force2d true" in flags
