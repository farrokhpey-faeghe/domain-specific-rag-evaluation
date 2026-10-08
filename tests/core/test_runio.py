from datetime import UTC, datetime

from deskrag.core.runio import (
    RunPaths,
    append_jsonl,
    make_run_id,
    read_json,
    read_jsonl,
    write_json,
)
from deskrag.core.schemas import ErrorRow, RunCounts, RunSummary


def test_run_id_has_timestamp_slugged_name_and_short_hash():
    run_id = make_run_id(
        "Prompt v2!", "1a2b3c4d5e6f", now=datetime(2026, 10, 8, 15, 30, 12, tzinfo=UTC)
    )
    assert run_id == "20261008-153012-prompt-v2-1a2b3c4d"


def test_run_paths_layout(tmp_path):
    paths = RunPaths(tmp_path / "r1")
    assert paths.meta == tmp_path / "r1" / "meta.json"
    assert paths.results == tmp_path / "r1" / "results.jsonl"
    assert paths.errors == tmp_path / "r1" / "errors.jsonl"
    assert paths.summary == tmp_path / "r1" / "summary.json"
    assert paths.report == tmp_path / "r1" / "report.md"


def test_append_then_read_jsonl_round_trips(tmp_path):
    path = tmp_path / "run" / "errors.jsonl"
    rows = [
        ErrorRow(run_id="r", case_id="c1", error_class="timeout", message="took > 120 s"),
        ErrorRow(
            run_id="r", case_id="c2", error_class="infra", message="Ollama unreachable", attempts=3
        ),
    ]
    for row in rows:
        append_jsonl(path, row)
    assert read_jsonl(path, ErrorRow) == rows


def test_read_jsonl_of_a_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "missing.jsonl", ErrorRow) == []


def test_write_then_read_json_round_trips(tmp_path):
    summary = RunSummary(
        run_id="r",
        metrics={"e2e_pass": 0.5},
        counts=RunCounts(cases=2, ok=2, errors=0, truncated=0, parse_errors=0),
    )
    write_json(tmp_path / "run" / "summary.json", summary)
    assert read_json(tmp_path / "run" / "summary.json", RunSummary) == summary
