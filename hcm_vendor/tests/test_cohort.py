import pandas as pd
import pytest

from hcmv.cohort import (
    build_acdc_cohort,
    build_mms2_cohort,
    check_expected_counts,
    cross_check_reference,
    load_mms2_info,
    mms2_challenge_split,
    samplesheet_rows,
)

CONFIG = {
    "cohort": {
        "diseases": ["NOR", "HCM"],
        "positive_class": "HCM",
        "vendor_map": {
            "SIEMENS": "Siemens",
            "Philips Medical Systems": "Philips",
            "GE MEDICAL SYSTEMS": "GE",
        },
        "train_vendors": ["Siemens", "Philips"],
    },
}


@pytest.fixture
def info_csv(tmp_path):
    rows = [
        "SUBJECT_CODE,DISEASE,VENDOR,SCANNER,FIELD",
        "1,NOR,GE MEDICAL SYSTEMS,SIGNA EXCITE,1.5",
        "41,LV,SIEMENS,SymphonyTim,1.5",
        "71,HCM,SIEMENS,SymphonyTim,1.5",
        "172,HCM,Philips Medical Systems,Achieva,1.5",
        "230,NOR,Philips Medical Systems,Achieva,1.5",
        ",,,,",  # Excel padding rows
        ",,,,",
    ]
    path = tmp_path / "dataset_information.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_load_mms2_info_drops_padding_rows(info_csv):
    info = load_mms2_info(info_csv)
    assert len(info) == 5
    assert info["SUBJECT_CODE"].dtype.kind == "i"


def test_mms2_cohort_filters_diseases_and_assigns_roles(info_csv):
    cohort = build_mms2_cohort(load_mms2_info(info_csv), CONFIG).set_index("source_id")
    assert list(cohort.index) == ["001", "071", "172", "230"]  # LV excluded, IDs zero-padded
    assert cohort.loc["001", "role"] == "ge_check"
    assert cohort.loc["071", "role"] == "train_pool"
    assert cohort.loc["071", "y"] == 1 and cohort.loc["230", "y"] == 0
    assert cohort.loc["172", "vendor"] == "Philips"
    assert cohort.loc["172", "mms2_challenge_split"] == "val"
    assert cohort.loc["071", "subject_id"] == "mms2_071"


def test_unmapped_vendor_is_rejected(info_csv):
    info = load_mms2_info(info_csv)
    info.loc[info.index[0], "VENDOR"] = "CANON"
    with pytest.raises(ValueError, match="Unmapped"):
        build_mms2_cohort(info, CONFIG)


def test_challenge_split_boundaries():
    assert [mms2_challenge_split(c) for c in (160, 161, 200, 201)] == ["train", "val", "val", "test"]


def test_acdc_cohort_reads_group(tmp_path):
    for name, group in (("patient101", "DCM"), ("patient102", "HCM"), ("patient103", "NOR")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "Info.cfg").write_text(f"ED: 1\nES: 12\nGroup: {group}\nHeight: 170\n")
    cohort = build_acdc_cohort(tmp_path, CONFIG)
    assert list(cohort["source_id"]) == ["patient102", "patient103"]
    assert list(cohort["y"]) == [1, 0]
    assert set(cohort["role"]) == {"external"}


def test_expected_counts_mismatch_raises(info_csv):
    cohort = build_mms2_cohort(load_mms2_info(info_csv), CONFIG)
    check_expected_counts(cohort, {"mms2": {"Siemens": {"HCM": 1}, "Philips": {"HCM": 1, "NOR": 1}, "GE": {"NOR": 1}}})
    with pytest.raises(ValueError, match="expected 2, got 1"):
        check_expected_counts(cohort, {"mms2": {"Siemens": {"HCM": 2}, "Philips": {"HCM": 1, "NOR": 1}, "GE": {"NOR": 1}}})


def test_samplesheet_rows_follow_sorat_layout(tmp_path):
    cohort = pd.DataFrame({"dataset": ["mms2", "acdc"], "source_id": ["071", "patient102"]})
    config = {"paths": {"repo_root": str(tmp_path), "mms2_sa_root": "/m/SA", "acdc_testing_root": "/a/testing"}}
    sheets = samplesheet_rows(cohort, config)
    assert sheets["mms2"].iloc[0].to_dict() == {
        "patient_id": "071",
        "image": "/m/SA/071/071_SA_CINE.nii.gz",
        "ground_truth": "/m/SA/071",
        "info_cfg": "/m/SA/071/Info.cfg",
    }
    assert sheets["acdc"].iloc[0]["image"] == "/a/testing/patient102/patient102_4d.nii.gz"


def test_cross_check_reference_detects_differences(tmp_path):
    sheet = pd.DataFrame([{"patient_id": "071", "image": "/x.nii.gz", "ground_truth": "/g", "info_cfg": "/i"}])
    ref = tmp_path / "ref.csv"
    ref.write_text("patient_id,image,ground_truth,info_cfg\n071,/x.nii.gz,/g,/other\n")
    assert cross_check_reference(sheet, ref) == ["071 info_cfg: /other != /i"]
