"""Check failure tags against the designed sample outcomes."""

from pathlib import Path

import pytest

from deskrag.core.dataset import load_cases
from deskrag.core.fakes import FakeRAGSystem
from deskrag.evaluation.failures import tag_failures
from deskrag.evaluation.metrics.answer import score_case

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def sample_data():
    cases = load_cases(SAMPLES / "cases.jsonl")
    system = FakeRAGSystem.from_corpus_dir(SAMPLES / "corpus")
    return cases, system


@pytest.mark.parametrize(
    "case_id, expected",
    [
        ("s01", []),
        ("s02", []),
        ("s03", ["WRONG_ANSWER"]),
        ("s04", ["RETRIEVAL_MISS", "FALSE_ABSTAIN_MODEL"]),
        ("s05", ["FALSE_ABSTAIN_GATE"]),
        ("s06", ["FORMAT_ERROR"]),
        ("s07", []),
        ("s08", ["MISSED_ABSTAIN"]),
        ("s09", []),
        ("s10", ["MISSED_ABSTAIN", "BAD_CITATION"]),
    ],
)
def test_sample_failure_tags(sample_data, case_id, expected):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == case_id)
    answer = system.answer(case.question)
    scores = score_case(case, answer)

    assert tag_failures(case, answer, scores) == expected


def test_truncated_output_is_not_missed_abstention(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s08")
    answer = system.answer(case.question).model_copy(
        update={
            "status": "truncated",
            "answer": None,
            "cited_doc_ids": [],
        }
    )
    scores = score_case(case, answer)

    assert tag_failures(case, answer, scores) == ["FORMAT_ERROR"]


def test_retrieval_miss_and_format_error_can_coexist(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s06")
    answer = system.answer(case.question).model_copy(update={"retrieved": []})
    scores = score_case(case, answer)

    assert tag_failures(case, answer, scores) == [
        "RETRIEVAL_MISS",
        "FORMAT_ERROR",
    ]


def test_partial_gold_retrieval_is_not_a_complete_miss(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s03")
    answer = system.answer(case.question)
    answer = answer.model_copy(update={"retrieved": answer.retrieved[:1]})
    scores = score_case(case, answer)

    assert tag_failures(case, answer, scores) == ["WRONG_ANSWER"]


def test_wrong_answer_and_bad_citation_can_coexist(sample_data):
    cases, system = sample_data
    case = next(case for case in cases if case.case_id == "s03")
    answer = system.answer(case.question).model_copy(
        update={"cited_doc_ids": ["unretrieved-document"]}
    )
    scores = score_case(case, answer)

    assert tag_failures(case, answer, scores) == [
        "WRONG_ANSWER",
        "BAD_CITATION",
    ]
