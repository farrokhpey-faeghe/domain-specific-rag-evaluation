"""Check integration of metrics, failure tags, and summaries."""

from pathlib import Path

import pytest

from deskrag.core.dataset import load_cases
from deskrag.core.fakes import FakeRAGSystem
from deskrag.core.schemas import ErrorRow
from deskrag.evaluation.runner import score_response, summarize

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def sample_data():
    cases = load_cases(SAMPLES / "cases.jsonl")
    system = FakeRAGSystem.from_corpus_dir(SAMPLES / "corpus")
    return cases, system


def test_response_combines_all_metric_modules(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s01")

    row = score_response(
        "test-run",
        case,
        system.answer(case.question),
        rep=2,
    )

    assert row.run_id == "test-run"
    assert row.case_id == "s01"
    assert row.rep == 2
    assert row.passed
    assert row.failure_tags == []
    assert row.scores["hit@1"] == 1.0
    assert row.scores["abstain_correct"] == 1.0
    assert row.scores["fact_recall"] == 1.0
    assert row.scores["format_error_rate"] == 0.0


def test_failed_response_has_scores_and_tags(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s03")

    row = score_response(
        "test-run",
        case,
        system.answer(case.question),
    )

    assert not row.passed
    assert row.scores["recall@5"] == 1.0
    assert row.scores["fact_recall"] == 0.5
    assert row.failure_tags == ["WRONG_ANSWER"]


def test_summary_matches_all_sample_outcomes(sample_data):
    cases, system = sample_data
    rows = [score_response("test-run", case, system.answer(case.question)) for case in cases]

    summary = summarize("test-run", rows, [])

    assert summary.metrics["e2e_pass"] == pytest.approx(4 / 10)
    assert summary.metrics["mrr"] == pytest.approx(0.75)
    assert summary.metrics["abstain_correct"] == pytest.approx(5 / 9)
    assert summary.metrics["format_error_rate"] == pytest.approx(0.1)

    assert summary.counts.cases == 10
    assert summary.counts.ok == 9
    assert summary.counts.parse_errors == 1
    assert summary.counts.truncated == 0
    assert summary.counts.errors == 0

    assert summary.failure_counts == {
        "WRONG_ANSWER": 1,
        "RETRIEVAL_MISS": 1,
        "FALSE_ABSTAIN_MODEL": 1,
        "FALSE_ABSTAIN_GATE": 1,
        "FORMAT_ERROR": 1,
        "MISSED_ABSTAIN": 2,
        "BAD_CITATION": 1,
    }


def test_infrastructure_error_is_not_scored(sample_data):
    cases, system = sample_data
    case = cases[0]
    row = score_response(
        "test-run",
        case,
        system.answer(case.question),
    )
    error = ErrorRow(
        run_id="test-run",
        case_id="failed-case",
        error_class="infra",
        message="Ollama is unavailable",
    )

    summary = summarize("test-run", [row], [error])

    assert summary.counts.cases == 2
    assert summary.counts.errors == 1
    assert summary.metrics["e2e_pass"] == 1.0


def test_repetitions_count_as_one_distinct_question(sample_data):
    cases, system = sample_data
    case = cases[0]
    rows = [
        score_response(
            "test-run",
            case,
            system.answer(case.question),
            rep=rep,
        )
        for rep in range(2)
    ]

    summary = summarize("test-run", rows, [])

    assert summary.counts.cases == 1
    assert summary.counts.ok == 2


def test_empty_summary():
    summary = summarize("test-run", [], [])

    assert summary.counts.cases == 0
    assert summary.failure_counts == {}
    assert all(value is None for value in summary.metrics.values())


def test_run_saves_sample_results(sample_data, tmp_path):
    from deskrag.core.runio import read_json, read_jsonl
    from deskrag.core.schemas import CaseResult, RunSummary
    from deskrag.evaluation.runner import run_cases

    cases, system = sample_data
    run_dir = tmp_path / "sample-run"

    summary = run_cases(
        "sample-run",
        cases,
        system,
        run_dir,
    )

    saved_rows = read_jsonl(
        run_dir / "results.jsonl",
        CaseResult,
    )
    saved_summary = read_json(
        run_dir / "summary.json",
        RunSummary,
    )

    assert len(saved_rows) == 10
    assert sum(row.passed for row in saved_rows) == 4
    assert saved_summary == summary
    assert summary.metrics["e2e_pass"] == pytest.approx(0.4)
    assert (run_dir / "errors.jsonl").read_text() == ""


def test_run_saves_each_repetition(sample_data, tmp_path):
    from deskrag.core.runio import read_jsonl
    from deskrag.core.schemas import CaseResult
    from deskrag.evaluation.runner import run_cases

    cases, system = sample_data
    run_dir = tmp_path / "repeated-run"

    run_cases(
        "repeated-run",
        cases[:1],
        system,
        run_dir,
        reps=2,
    )

    saved_rows = read_jsonl(
        run_dir / "results.jsonl",
        CaseResult,
    )

    assert len(saved_rows) == 2
    assert [row.rep for row in saved_rows] == [0, 1]


def test_run_logs_infra_error_and_continues(sample_data, tmp_path):
    from deskrag.core.interfaces import InfraError
    from deskrag.core.runio import read_jsonl
    from deskrag.core.schemas import CaseResult
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data

    class FailingSystem:
        def answer(self, question):
            if question == cases[0].question:
                raise InfraError("Ollama unavailable")

            return fake.answer(question)

    run_dir = tmp_path / "error-run"

    summary = run_cases(
        "error-run",
        cases[:2],
        FailingSystem(),
        run_dir,
    )

    saved_errors = read_jsonl(
        run_dir / "errors.jsonl",
        ErrorRow,
    )
    saved_rows = read_jsonl(
        run_dir / "results.jsonl",
        CaseResult,
    )

    assert len(saved_errors) == 1
    assert saved_errors[0].error_class == "infra"
    assert len(saved_rows) == 1
    assert saved_rows[0].case_id == cases[1].case_id
    assert summary.counts.errors == 1
    assert summary.metrics["e2e_pass"] == 1.0


def test_run_does_not_overwrite_existing_directory(
    sample_data,
    tmp_path,
):
    from deskrag.evaluation.runner import run_cases

    cases, system = sample_data
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    marker = run_dir / "keep.txt"
    marker.write_text("existing data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_cases("existing-run", cases, system, run_dir)

    assert marker.read_text(encoding="utf-8") == "existing data"


def test_run_saves_result_before_asking_next_question(
    sample_data,
    tmp_path,
):
    from deskrag.core.runio import read_jsonl
    from deskrag.core.schemas import CaseResult
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    run_dir = tmp_path / "incremental-run"

    class ObservingSystem:
        def answer(self, question):
            if question == cases[1].question:
                saved = read_jsonl(
                    run_dir / "results.jsonl",
                    CaseResult,
                )
                assert len(saved) == 1
                assert saved[0].case_id == cases[0].case_id

            return fake.answer(question)

    summary = run_cases(
        "incremental-run",
        cases[:2],
        ObservingSystem(),
        run_dir,
    )

    assert summary.counts.errors == 0
    assert summary.counts.ok == 2


@pytest.fixture
def fake_config():
    from deskrag.core.config import SystemConfig

    return SystemConfig(name="resume-test", system="fake", corpus_dir=str(SAMPLES / "corpus"))


def test_resume_skips_completed_repetitions(sample_data, fake_config, tmp_path):
    from deskrag.core.runio import read_jsonl
    from deskrag.core.schemas import CaseResult
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    target = tmp_path / "resume"
    run_cases("r", cases[:2], fake, target, reps=2, config=fake_config)

    class MustNotRun:
        def answer(self, question):
            raise AssertionError("Completed results must not run again")

    before = (target / "results.jsonl").read_bytes()
    summary = run_cases(
        "r", cases[:2], MustNotRun(), target, reps=2, config=fake_config, resume=True
    )
    assert (target / "results.jsonl").read_bytes() == before
    assert len(read_jsonl(target / "results.jsonl", CaseResult)) == 4
    assert summary.counts.errors == 0


def test_resume_retries_errors_and_removes_successes(sample_data, fake_config, tmp_path):
    from deskrag.core.interfaces import InfraError
    from deskrag.core.runio import read_jsonl
    from deskrag.core.schemas import CaseResult
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    target = tmp_path / "retry"

    class FailsFirst:
        def answer(self, question):
            if question == cases[0].question:
                raise InfraError("offline")
            return fake.answer(question)

    run_cases("r", cases[:2], FailsFirst(), target, config=fake_config)
    summary = run_cases("r", cases[:2], fake, target, config=fake_config, resume=True)
    assert summary.counts.errors == 0
    assert summary.metrics["e2e_pass"] == 1.0
    assert (target / "errors.jsonl").read_text() == ""
    assert len(read_jsonl(target / "results.jsonl", CaseResult)) == 2


def test_resume_preserves_errors_that_fail_again(sample_data, fake_config, tmp_path):
    from deskrag.core.interfaces import InfraError
    from deskrag.core.runio import read_jsonl
    from deskrag.evaluation.runner import run_cases

    cases, _ = sample_data
    target = tmp_path / "failed"

    class AlwaysFails:
        def answer(self, question):
            raise InfraError("offline")

    run_cases("r", cases[:1], AlwaysFails(), target, config=fake_config)
    summary = run_cases("r", cases[:1], AlwaysFails(), target, config=fake_config, resume=True)
    errors = read_jsonl(target / "errors.jsonl", ErrorRow)
    assert len(errors) == 1
    assert errors[0].attempts == 2
    assert summary.counts.errors == 1
    assert summary.metrics["e2e_pass"] is None


@pytest.mark.parametrize("change", ["config", "dataset", "reps"])
def test_resume_rejects_mismatches_without_writing(sample_data, fake_config, tmp_path, change):
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    target = tmp_path / "mismatch"
    run_cases("r", cases[:2], fake, target, config=fake_config)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    config = fake_config
    selected = cases[:2]
    reps = 1
    if change == "config":
        config = fake_config.model_copy(update={"prompt_version": "different"})
    elif change == "dataset":
        selected = cases[:1]
    else:
        reps = 2
    with pytest.raises(ValueError, match="Cannot resume"):
        run_cases("r", selected, fake, target, reps=reps, config=config, resume=True)
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}


def test_resume_continues_after_interruption(sample_data, fake_config, tmp_path):
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    target = tmp_path / "interrupted"

    class Interrupted:
        def answer(self, question):
            if question == cases[1].question:
                raise KeyboardInterrupt
            return fake.answer(question)

    with pytest.raises(KeyboardInterrupt):
        run_cases("r", cases[:2], Interrupted(), target, config=fake_config)
    summary = run_cases("r", cases[:2], fake, target, config=fake_config, resume=True)
    assert summary.counts.ok == 2
    assert summary.counts.errors == 0


def test_resume_refuses_legacy_directory_without_metadata(sample_data, fake_config, tmp_path):
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    target = tmp_path / "legacy"
    run_cases("r", cases[:1], fake, target)
    with pytest.raises(ValueError, match="without meta.json"):
        run_cases("r", cases[:1], fake, target, config=fake_config, resume=True)


@pytest.mark.parametrize("error_type", [KeyError, RuntimeError])
def test_scoring_bugs_propagate_without_system_error_rows(
    sample_data, tmp_path, monkeypatch, error_type
):
    from deskrag.evaluation.metrics import answer
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    target = tmp_path / "scorer-bug"
    original = answer.score_case
    calls = []

    def broken_score(case, response):
        calls.append(case.case_id)
        if len(calls) == 2:
            raise error_type("scorer bug")
        return original(case, response)

    monkeypatch.setattr(answer, "score_case", broken_score)
    with pytest.raises(error_type, match="scorer bug"):
        run_cases("r", cases, fake, target)

    assert calls == [cases[0].case_id, cases[1].case_id]
    assert (target / "errors.jsonl").read_text() == ""
    from deskrag.core.runio import read_jsonl
    from deskrag.core.schemas import CaseResult

    saved = read_jsonl(target / "results.jsonl", CaseResult)
    assert len(saved) == 1
    assert saved[0].case_id == cases[0].case_id


def test_unexpected_system_exception_is_still_logged(sample_data, tmp_path):
    from deskrag.core.runio import read_jsonl
    from deskrag.evaluation.runner import run_cases

    cases, _ = sample_data
    target = tmp_path / "system-bug"

    class BrokenSystem:
        def answer(self, question):
            raise RuntimeError("system failure")

    summary = run_cases("r", cases[:1], BrokenSystem(), target)
    errors = read_jsonl(target / "errors.jsonl", ErrorRow)
    assert summary.counts.errors == 1
    assert len(errors) == 1
    assert errors[0].error_class == "exception"
    assert "system failure" in errors[0].message


def test_metadata_and_case_type_breakdown(sample_data, fake_config, tmp_path):
    from deskrag.core.config import config_hash
    from deskrag.core.dataset import dataset_sha256
    from deskrag.core.runio import read_json
    from deskrag.core.schemas import RunMeta
    from deskrag.evaluation.runner import run_cases

    cases, fake = sample_data
    target = tmp_path / "metadata"
    summary = run_cases(
        "r", cases, fake, target, config=fake_config, dataset_path=str(SAMPLES / "cases.jsonl")
    )
    meta = read_json(target / "meta.json", RunMeta)
    assert meta.config_hash == config_hash(fake_config)
    assert meta.dataset.sha256 == dataset_sha256(cases)
    assert meta.dataset.n_cases == 10
    assert meta.corpus.n_docs == 5
    assert meta.env.python
    assert meta.env.model_digests == {}
    assert summary.by_case_type["single_doc"]["e2e_pass"] == pytest.approx(0.5)
    assert summary.by_case_type["multi_doc"]["fact_recall"] == 0.5
    assert summary.by_case_type["out_of_domain"]["e2e_pass"] == 1.0
