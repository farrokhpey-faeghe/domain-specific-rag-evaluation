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
