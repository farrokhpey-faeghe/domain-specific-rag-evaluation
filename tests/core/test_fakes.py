import json
from pathlib import Path

import pytest

from deskrag.core.corpus import load_corpus
from deskrag.core.dataset import load_cases
from deskrag.core.fakes import SAMPLE_OUTCOMES, FakeLLM, FakeRAGSystem, FakeRetriever
from deskrag.core.interfaces import InfraError, LLMClient, RAGSystem, Retriever

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(SAMPLES / "corpus")


@pytest.fixture(scope="module")
def cases():
    return load_cases(SAMPLES / "cases.jsonl")


def test_fakes_satisfy_the_protocols(docs):
    assert isinstance(FakeRetriever(docs), Retriever)
    assert isinstance(FakeLLM(), LLMClient)
    assert isinstance(FakeRAGSystem(docs), RAGSystem)


def test_infra_error_is_an_exception():
    assert issubclass(InfraError, Exception)


def test_fake_retriever_ranks_the_matching_document_first(docs):
    results = FakeRetriever(docs).retrieve("How much does an electronic transcript cost?", k=3)
    assert results[0].chunk.doc_id == "registrar-transcripts"
    assert [r.rank for r in results] == [1, 2, 3]


def test_fake_retriever_scores_never_increase_down_the_list(docs):
    scores = [r.score for r in FakeRetriever(docs).retrieve("visitors guest network", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_fake_llm_returns_the_reply_whose_key_is_in_the_prompt():
    reply_json = '{"abstain": false, "answer": "$10", "citations": ["registrar-transcripts"]}'
    llm = FakeLLM({"transcript": reply_json})
    reply = llm.chat([{"role": "user", "content": "Question: How much is a transcript?"}])
    assert json.loads(reply.text)["answer"] == "$10"
    assert reply.model == "fake-llm"
    assert len(llm.calls) == 1


def test_fake_llm_abstains_by_default():
    reply = FakeLLM().chat([{"role": "user", "content": "anything"}])
    assert json.loads(reply.text)["abstain"] is True


def test_every_sample_question_has_a_canned_outcome(cases):
    assert {case.question for case in cases} == set(SAMPLE_OUTCOMES)


def test_fake_rag_system_covers_the_designed_outcomes(docs, cases):
    system = FakeRAGSystem(docs)
    answers = {case.case_id: system.answer(case.question) for case in cases}
    assert "$10" in answers["s01"].answer
    assert answers["s02"].retrieved[1].chunk.doc_id == "it-wifi"  # gold doc at rank 2
    assert answers["s05"].abstain_reason == "low_retrieval_score"
    assert answers["s05"].usage.llm_calls == 0  # the gate abstains before calling the LLM
    assert answers["s06"].status == "parse_error"
    assert answers["s09"].abstain_reason == "model_declined"
    assert "student-records" in answers["s10"].cited_doc_ids  # never retrieved


def test_fake_rag_system_confidence_is_the_top_retrieval_score(docs):
    answer = FakeRAGSystem(docs).answer("How much does an electronic transcript cost?")
    assert answer.confidence == answer.retrieved[0].score


def test_fake_rag_system_declines_unknown_questions(docs):
    answer = FakeRAGSystem(docs).answer("Who won the 1998 World Cup?")
    assert answer.abstained
    assert answer.abstain_reason == "model_declined"


def test_fake_rag_system_can_load_its_own_corpus():
    system = FakeRAGSystem.from_corpus_dir(SAMPLES / "corpus")
    assert system.answer("What is my current GPA?").answer
