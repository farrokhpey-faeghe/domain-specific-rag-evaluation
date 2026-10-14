"""Check the reference systems through the existing system builder."""

from pathlib import Path

import pytest

from deskrag.core.config import SystemConfig
from deskrag.core.dataset import load_cases
from deskrag.core.interfaces import RAGSystem
from deskrag.evaluation.metrics.answer import score_case
from deskrag.systems import build_system

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture
def cases():
    return load_cases(SAMPLES / "cases.jsonl")


def make_system(kind, cases):
    cfg = SystemConfig(name=f"test-{kind}", system=kind)
    return build_system(cfg, cases=cases)


def test_oracle_passes_every_sample(cases):
    system = make_system("oracle", cases)

    for case in cases:
        answer = system.answer(case.question)
        scores = score_case(case, answer)

        assert scores["e2e_pass"] == 1.0, case.case_id


def test_oracle_supplies_gold_documents_and_citations(cases):
    system = make_system("oracle", cases)
    case = next(case for case in cases if case.case_id == "s03")

    answer = system.answer(case.question)

    assert [item.chunk.doc_id for item in answer.retrieved] == (case.gold_doc_ids)
    assert answer.cited_doc_ids == case.gold_doc_ids
    assert [item.rank for item in answer.retrieved] == [1, 2]


def test_always_abstain_declines_every_sample(cases):
    system = make_system("always_abstain", cases)

    for case in cases:
        answer = system.answer(case.question)

        assert answer.abstained
        assert answer.answer is None
        assert answer.status == "ok"
        assert answer.usage.llm_calls == 0


def test_never_abstain_answers_every_sample(cases):
    system = make_system("never_abstain", cases)

    for case in cases:
        answer = system.answer(case.question)

        assert not answer.abstained
        assert answer.answer
        assert answer.status == "ok"
        assert answer.usage.llm_calls == 0


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("oracle", 1.0),
        ("always_abstain", 0.4),
        ("never_abstain", 0.6),
    ],
)
def test_expected_overall_pass_rates(cases, kind, expected):
    system = make_system(kind, cases)
    passes = []

    for case in cases:
        answer = system.answer(case.question)
        passes.append(score_case(case, answer)["e2e_pass"])

    assert sum(passes) / len(passes) == pytest.approx(expected)


def test_sanity_system_matches_chatbot_interface(cases):
    system = make_system("oracle", cases)

    assert isinstance(system, RAGSystem)


def test_unknown_question_is_rejected(cases):
    system = make_system("oracle", cases)

    with pytest.raises(ValueError, match="Question not found"):
        system.answer("A question absent from this test set")


def test_duplicate_questions_are_rejected(cases):
    with pytest.raises(ValueError, match="Duplicate question"):
        make_system("oracle", [cases[0], cases[0]])
