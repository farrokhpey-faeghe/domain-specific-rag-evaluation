"""Check operational metrics using known response measurements."""

from pathlib import Path

import pytest

from deskrag.core.dataset import load_cases
from deskrag.core.fakes import FakeRAGSystem
from deskrag.core.schemas import CaseResult, Usage
from deskrag.evaluation.metrics.ops import aggregate, score_case

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def sample_data():
    cases = load_cases(SAMPLES / "cases.jsonl")
    system = FakeRAGSystem.from_corpus_dir(SAMPLES / "corpus")
    return cases, system


def test_normal_response_measurements(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s01")

    scores = score_case(case, system.answer(case.question))

    assert scores == {
        "format_error_rate": 0.0,
        "latency_s": 0.6,
        "prompt_tokens": 420.0,
        "completion_tokens": 24.0,
    }


def test_gate_abstention_uses_no_llm_tokens(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s05")

    scores = score_case(case, system.answer(case.question))

    assert scores["format_error_rate"] == 0.0
    assert scores["latency_s"] == 0.02
    assert scores["prompt_tokens"] == 0.0
    assert scores["completion_tokens"] == 0.0


@pytest.mark.parametrize("status", ["parse_error", "truncated"])
def test_broken_output_counts_as_format_error(sample_data, status):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s06")
    answer = system.answer(case.question).model_copy(update={"status": status})

    scores = score_case(case, answer)

    assert scores["format_error_rate"] == 1.0
    assert scores["prompt_tokens"] == 420.0


def test_all_sample_operational_averages(sample_data):
    cases, system = sample_data
    rows = []

    for case in cases:
        answer = system.answer(case.question)

        rows.append(
            CaseResult(
                run_id="test-run",
                case_id=case.case_id,
                answer=answer,
                scores=score_case(case, answer),
                passed=False,
            )
        )

    metrics = aggregate(rows)

    assert metrics["format_error_rate"] == pytest.approx(0.1)
    assert metrics["prompt_tokens_mean"] == pytest.approx(336.0)
    assert metrics["completion_tokens_mean"] == pytest.approx(19.2)
    assert metrics["latency_p50_s"] == pytest.approx(0.6)
    assert metrics["latency_p95_s"] == pytest.approx(0.6)


def test_percentiles_with_different_response_times(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s01")
    original = system.answer(case.question)
    rows = []

    for rep, latency in enumerate([5.0, 1.0, 4.0, 2.0, 3.0]):
        answer = original.model_copy(update={"usage": Usage(latency_s=latency)})

        rows.append(
            CaseResult(
                run_id="test-run",
                case_id=case.case_id,
                rep=rep,
                answer=answer,
                scores=score_case(case, answer),
                passed=False,
            )
        )

    metrics = aggregate(rows)

    assert metrics["latency_p50_s"] == pytest.approx(3.0)
    assert metrics["latency_p95_s"] == pytest.approx(4.8)


def test_single_response_percentiles(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s01")
    answer = system.answer(case.question)

    row = CaseResult(
        run_id="test-run",
        case_id=case.case_id,
        answer=answer,
        scores=score_case(case, answer),
        passed=False,
    )

    metrics = aggregate([row])

    assert metrics["latency_p50_s"] == pytest.approx(0.6)
    assert metrics["latency_p95_s"] == pytest.approx(0.6)


def test_empty_results_have_no_metrics():
    assert all(value is None for value in aggregate([]).values())
