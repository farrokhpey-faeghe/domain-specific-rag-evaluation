"""Check retrieval metrics against known sample outcomes."""

from pathlib import Path

import pytest

from deskrag.core.dataset import load_cases
from deskrag.core.fakes import FakeRAGSystem
from deskrag.core.schemas import CaseResult
from deskrag.evaluation.metrics.retrieval import aggregate, score_case

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def sample_data():
    cases = load_cases(SAMPLES / "cases.jsonl")
    system = FakeRAGSystem.from_corpus_dir(SAMPLES / "corpus")
    return cases, system


def test_s02_gold_document_is_second(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s02")

    scores = score_case(case, system.answer(case.question))

    assert scores == {
        "hit@1": 0.0,
        "hit@3": 1.0,
        "hit@5": 1.0,
        "recall@5": 1.0,
        "mrr": 0.5,
    }


def test_s03_both_gold_documents_are_retrieved(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s03")

    scores = score_case(case, system.answer(case.question))

    assert scores["hit@1"] == 1.0
    assert scores["recall@5"] == 1.0


def test_repeated_chunks_do_not_take_document_positions(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s02")
    answer = system.answer(case.question)

    # Repeat the first document before the gold document.
    first, second, third = answer.retrieved
    answer = answer.model_copy(update={"retrieved": [first, first, second, third]})

    scores = score_case(case, answer)

    assert scores["mrr"] == 0.5


def test_unanswerable_question_has_no_retrieval_scores(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s07")

    scores = score_case(case, system.answer(case.question))

    assert all(value is None for value in scores.values())


def test_all_sample_retrieval_averages(sample_data):
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

    scores = aggregate(rows)

    assert scores["hit@1"] == pytest.approx(4 / 6)
    assert scores["hit@3"] == pytest.approx(5 / 6)
    assert scores["hit@5"] == pytest.approx(5 / 6)
    assert scores["recall@5"] == pytest.approx(5 / 6)
    assert scores["mrr"] == pytest.approx(0.75)


def test_empty_results_have_no_averages():
    assert all(value is None for value in aggregate([]).values())
