"""Check abstention metrics using known fake-system outcomes."""

from pathlib import Path

import pytest

from deskrag.core.dataset import load_cases
from deskrag.core.fakes import FakeRAGSystem
from deskrag.core.schemas import CaseResult
from deskrag.evaluation.metrics.abstention import aggregate, score_case

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def sample_data():
    cases = load_cases(SAMPLES / "cases.jsonl")
    system = FakeRAGSystem.from_corpus_dir(SAMPLES / "corpus")
    return cases, system


def test_answerable_question_incorrectly_declined(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s05")

    scores = score_case(case, system.answer(case.question))

    assert scores == {
        "abstain_correct": 0.0,
        "false_abstain_rate": 1.0,
        "missed_abstain_rate": None,
    }


def test_unanswerable_question_correctly_declined(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s07")

    scores = score_case(case, system.answer(case.question))

    assert scores == {
        "abstain_correct": 1.0,
        "false_abstain_rate": None,
        "missed_abstain_rate": 0.0,
    }


def test_unanswerable_question_incorrectly_answered(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s08")

    scores = score_case(case, system.answer(case.question))

    assert scores == {
        "abstain_correct": 0.0,
        "false_abstain_rate": None,
        "missed_abstain_rate": 1.0,
    }


@pytest.mark.parametrize("status", ["parse_error", "truncated"])
def test_broken_output_is_excluded(sample_data, status):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s06")
    answer = system.answer(case.question).model_copy(update={"status": status})

    scores = score_case(case, answer)

    assert all(value is None for value in scores.values())


def test_all_sample_abstention_averages(sample_data):
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

    assert metrics["abstain_correct"] == pytest.approx(5 / 9)
    assert metrics["false_abstain_rate"] == pytest.approx(2 / 5)
    assert metrics["missed_abstain_rate"] == pytest.approx(2 / 4)
    assert metrics["abstain_precision"] == pytest.approx(0.5)
    assert metrics["abstain_recall"] == pytest.approx(0.5)
    assert metrics["abstain_f1"] == pytest.approx(0.5)


def test_no_predicted_abstentions(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s08")
    answer = system.answer(case.question)

    row = CaseResult(
        run_id="test-run",
        case_id=case.case_id,
        answer=answer,
        scores=score_case(case, answer),
        passed=False,
    )

    metrics = aggregate([row])

    assert metrics["abstain_precision"] == 0.0
    assert metrics["abstain_recall"] == 0.0
    assert metrics["abstain_f1"] == 0.0


def test_empty_results_have_no_metrics():
    assert all(value is None for value in aggregate([]).values())
