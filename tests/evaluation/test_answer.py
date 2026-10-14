"""Check answer metrics against known sample outcomes."""

from pathlib import Path

import pytest

from deskrag.core.dataset import load_cases
from deskrag.core.fakes import FakeRAGSystem
from deskrag.core.schemas import CaseResult
from deskrag.evaluation.metrics.answer import aggregate, score_case

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def sample_data():
    cases = load_cases(SAMPLES / "cases.jsonl")
    system = FakeRAGSystem.from_corpus_dir(SAMPLES / "corpus")
    return cases, system


def test_correct_answer_passes(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s01")

    scores = score_case(case, system.answer(case.question))

    assert scores == {
        "fact_recall": 1.0,
        "answer_correct": 1.0,
        "citation_valid": 1.0,
        "citation_hit": 1.0,
        "e2e_pass": 1.0,
    }


def test_missing_fact_fails_overall(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s03")

    scores = score_case(case, system.answer(case.question))

    assert scores["fact_recall"] == 0.5
    assert scores["answer_correct"] == 0.0
    assert scores["e2e_pass"] == 0.0


def test_matching_handles_case_whitespace_and_alternatives(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s03")
    case = case.model_copy(
        update={
            "must_include": [
                "parking portal|permit website",
                "student portal",
            ]
        }
    )

    answer = system.answer(case.question).model_copy(
        update={
            "answer": (
                "Use the PERMIT   WEBSITE for parking and the STUDENT\nPORTAL for transcripts."
            )
        }
    )

    scores = score_case(case, answer)

    assert scores["fact_recall"] == 1.0
    assert scores["answer_correct"] == 1.0


def test_answerable_abstention_fails(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s05")

    scores = score_case(case, system.answer(case.question))

    assert scores["fact_recall"] is None
    assert scores["answer_correct"] is None
    assert scores["e2e_pass"] == 0.0


def test_unanswerable_abstention_passes(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s07")

    scores = score_case(case, system.answer(case.question))

    assert scores["fact_recall"] is None
    assert scores["citation_valid"] is None
    assert scores["e2e_pass"] == 1.0


def test_unanswerable_answer_and_bad_citation_fail(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s10")

    scores = score_case(case, system.answer(case.question))

    assert scores["fact_recall"] is None
    assert scores["citation_valid"] == 0.0
    assert scores["citation_hit"] == 0.0
    assert scores["e2e_pass"] == 0.0


@pytest.mark.parametrize("status", ["parse_error", "truncated"])
def test_broken_output_fails_overall(sample_data, status):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s06")
    answer = system.answer(case.question).model_copy(update={"status": status})

    scores = score_case(case, answer)

    assert scores["fact_recall"] is None
    assert scores["citation_valid"] is None
    assert scores["e2e_pass"] == 0.0


def test_empty_citations_are_valid_but_have_no_hit(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s01")
    answer = system.answer(case.question).model_copy(update={"cited_doc_ids": []})

    scores = score_case(case, answer)

    assert scores["citation_valid"] == 1.0
    assert scores["citation_hit"] == 0.0
    assert scores["e2e_pass"] == 1.0


@pytest.mark.parametrize(
    "split, expected",
    [(None, 4 / 10), ("dev", 4 / 7)],
)
def test_sample_overall_pass_rate(sample_data, split, expected):
    cases, system = sample_data
    rows = []

    for case in cases:
        if split is not None and case.split != split:
            continue

        answer = system.answer(case.question)
        scores = score_case(case, answer)

        rows.append(
            CaseResult(
                run_id="test-run",
                case_id=case.case_id,
                answer=answer,
                scores=scores,
                passed=scores["e2e_pass"] == 1.0,
            )
        )

    metrics = aggregate(rows)

    assert metrics["e2e_pass"] == pytest.approx(expected)


def test_empty_results_have_no_metrics():
    assert all(value is None for value in aggregate([]).values())
