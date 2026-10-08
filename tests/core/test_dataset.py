import json
from pathlib import Path

import pytest

from deskrag.core.dataset import DatasetError, dataset_sha256, load_cases

REPO = Path(__file__).resolve().parents[2]
SAMPLE_CASES = REPO / "data" / "samples" / "cases.jsonl"


def case_row(case_id: str, split: str = "dev", **overrides) -> str:
    row = {
        "case_id": case_id,
        "question": f"Question {case_id}?",
        "answerable": False,
        "case_type": "out_of_domain",
        "split": split,
        "author": "omar",
    }
    row.update(overrides)
    return json.dumps(row)


def write_rows(path: Path, *rows: str) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_sample_cases_cover_every_case_type():
    cases = load_cases(SAMPLE_CASES)
    assert len(cases) == 10
    assert {c.case_type for c in cases} == {
        "single_doc",
        "multi_doc",
        "paraphrase",
        "out_of_domain",
        "missing_info",
        "false_premise",
        "personal",
    }


def test_split_filter_keeps_only_that_split():
    dev = load_cases(SAMPLE_CASES, split="dev")
    assert dev
    assert all(case.split == "dev" for case in dev)


def test_directory_loads_every_jsonl_file_and_skips_blank_lines(tmp_path):
    write_rows(tmp_path / "omar.jsonl", case_row("om-001"))
    write_rows(tmp_path / "sam.jsonl", case_row("sa-001"), "", case_row("sa-002"))
    assert [c.case_id for c in load_cases(tmp_path)] == ["om-001", "sa-001", "sa-002"]


def test_duplicate_case_ids_across_files_are_an_error(tmp_path):
    write_rows(tmp_path / "a.jsonl", case_row("x-001"))
    write_rows(tmp_path / "b.jsonl", case_row("x-001"))
    with pytest.raises(DatasetError, match="x-001"):
        load_cases(tmp_path)


def test_invalid_row_error_names_the_file_and_line(tmp_path):
    path = write_rows(
        tmp_path / "omar.jsonl", case_row("om-001"), case_row("om-002", case_type="single_doc")
    )
    with pytest.raises(DatasetError, match=r"omar\.jsonl:2"):
        load_cases(path)


def test_malformed_json_error_names_the_file_and_line(tmp_path):
    path = write_rows(tmp_path / "omar.jsonl", "{not json")
    with pytest.raises(DatasetError, match=r"omar\.jsonl:1"):
        load_cases(path)


def test_dataset_sha256_ignores_cases_from_other_splits(tmp_path):
    write_rows(tmp_path / "a.jsonl", case_row("a-1"), case_row("a-2", split="test"))
    before = dataset_sha256(load_cases(tmp_path, split="dev"))
    write_rows(
        tmp_path / "a.jsonl", case_row("a-1"), case_row("a-2", split="test", question="Edited?")
    )
    assert dataset_sha256(load_cases(tmp_path, split="dev")) == before


def test_dataset_sha256_changes_when_a_case_changes(tmp_path):
    write_rows(tmp_path / "a.jsonl", case_row("a-1"))
    before = dataset_sha256(load_cases(tmp_path))
    write_rows(tmp_path / "a.jsonl", case_row("a-1", question="Edited?"))
    assert dataset_sha256(load_cases(tmp_path)) != before
