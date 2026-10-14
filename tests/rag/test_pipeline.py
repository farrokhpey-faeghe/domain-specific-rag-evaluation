import json
from pathlib import Path

import pytest

from deskrag.core.config import load_config
from deskrag.core.corpus import load_corpus
from deskrag.core.fakes import FakeLLM, FakeRetriever
from deskrag.core.schemas import LLMResponse
from deskrag.rag.pipeline import (
    ANSWER_SCHEMA,
    ModelReply,
    RAGPipeline,
    build_rag_system,
    load_prompt,
)

REPO = Path(__file__).resolve().parents[2]
SAMPLE_CORPUS = REPO / "data" / "samples" / "corpus"
QUESTION = "How much does an electronic transcript cost?"


def reply(abstain=False, answer="", citations=()):
    return json.dumps({"abstain": abstain, "answer": answer, "citations": list(citations)})


class ScriptedLLM:
    """Gives the same reply to every call and records what it was asked."""

    def __init__(self, text, done_reason="stop"):
        self.text = text
        self.done_reason = done_reason
        self.calls = []

    def chat(self, messages, *, json_schema=None, options=None):
        self.calls.append({"messages": messages, "json_schema": json_schema, "options": options})
        return LLMResponse(
            text=self.text,
            model="fake-llm",
            done_reason=self.done_reason,
            prompt_tokens=100,
            completion_tokens=20,
            latency_s=0.5,
        )


def retriever():
    return FakeRetriever(load_corpus(SAMPLE_CORPUS))


def pipeline(llm, **kwargs):
    return RAGPipeline(llm=llm, system_prompt="SYSTEM", retriever=retriever(), k=3, **kwargs)


def test_answers_with_the_models_citations():
    text = reply(
        answer=" Electronic transcripts cost $10 each. ", citations=["registrar-transcripts"]
    )
    answer = pipeline(ScriptedLLM(text)).answer(QUESTION)
    assert answer.status == "ok"
    assert not answer.abstained
    assert answer.answer == "Electronic transcripts cost $10 each."
    assert answer.cited_doc_ids == ["registrar-transcripts"]
    assert answer.retrieved[0].chunk.doc_id == "registrar-transcripts"
    assert answer.confidence == answer.retrieved[0].score
    assert answer.raw_output == text
    assert answer.model == "fake-llm"
    assert (answer.usage.prompt_tokens, answer.usage.completion_tokens) == (100, 20)
    assert answer.usage.llm_calls == 1


def test_the_model_declining_is_an_abstention():
    text = reply(abstain=True, answer="maybe $10", citations=["registrar-transcripts"])
    answer = pipeline(ScriptedLLM(text)).answer(QUESTION)
    assert answer.abstained
    assert answer.abstain_reason == "model_declined"
    assert answer.answer is None
    assert answer.cited_doc_ids == []
    assert answer.status == "ok"
    assert answer.raw_output == text


def test_the_gate_abstains_without_calling_the_llm():
    llm = ScriptedLLM(reply(answer="$10"))
    answer = pipeline(llm, min_score=100.0).answer(QUESTION)
    assert answer.abstained
    assert answer.abstain_reason == "low_retrieval_score"
    assert llm.calls == []
    assert answer.usage.llm_calls == 0
    assert answer.model is None
    assert answer.raw_output is None
    assert answer.retrieved
    assert answer.confidence == answer.retrieved[0].score


def test_a_top_score_equal_to_the_threshold_passes_the_gate():
    top = retriever().retrieve(QUESTION, 1)[0].score
    llm = ScriptedLLM(reply(answer="$10"))
    assert not pipeline(llm, min_score=top).answer(QUESTION).abstained
    assert len(llm.calls) == 1


def test_truncated_output_is_reported_not_abstained():
    llm = ScriptedLLM(reply(abstain=True), done_reason="length")
    answer = pipeline(llm).answer(QUESTION)
    assert answer.status == "truncated"
    assert not answer.abstained
    assert answer.answer is None


@pytest.mark.parametrize(
    "text",
    [
        "Sure! It costs $10.",
        '{"abstain": false, "answer": "$10"}',
        '{"abstain": "no", "answer": "$10", "citations": []}',
        '{"abstain": false, "answer": "   ", "citations": []}',
    ],
    ids=["not-json", "missing-key", "wrong-type", "empty-answer"],
)
def test_unusable_output_is_a_parse_error_not_an_abstention(text):
    answer = pipeline(ScriptedLLM(text)).answer(QUESTION)
    assert answer.status == "parse_error"
    assert not answer.abstained
    assert answer.raw_output == text


def test_citations_are_cleaned_but_never_filtered():
    citations = ["[it-wifi]", "it-wifi#2", " it-wifi ", "student-records"]
    answer = pipeline(ScriptedLLM(reply(answer="x", citations=citations))).answer(QUESTION)
    assert answer.cited_doc_ids == ["it-wifi", "student-records"]


def test_the_prompt_holds_the_context_blocks_then_the_question():
    llm = ScriptedLLM(reply(answer="$10"))
    pipeline(llm, llm_options={"seed": 42}).answer(QUESTION)
    call = llm.calls[0]
    system, user = call["messages"]
    assert system == {"role": "system", "content": "SYSTEM"}
    assert user["role"] == "user"
    assert user["content"].startswith("Context documents:\n\n[registrar-transcripts] ")
    assert user["content"].endswith(f"\n\nQuestion: {QUESTION}")
    assert call["json_schema"] == ANSWER_SCHEMA
    assert call["options"] == {"seed": 42}


def test_without_a_retriever_only_the_question_is_sent_and_there_is_no_gate():
    llm = ScriptedLLM(reply(answer="$10"))
    answer = RAGPipeline(llm=llm, system_prompt="SYSTEM", min_score=5.0).answer(QUESTION)
    assert llm.calls[0]["messages"][1]["content"] == f"Question: {QUESTION}"
    assert answer.answer == "$10"
    assert answer.retrieved == []
    assert answer.confidence is None


def test_works_with_the_shared_fakes():
    llm = FakeLLM({QUESTION: reply(answer="$10 each", citations=["registrar-transcripts"])})
    system = pipeline(llm)
    assert system.answer(QUESTION).answer == "$10 each"
    assert system.answer("Which Wi-Fi network should campus visitors use?").abstained


def test_the_schema_and_the_reply_model_agree():
    assert set(ANSWER_SCHEMA["required"]) == set(ModelReply.model_fields)


def test_load_prompt_reads_the_versioned_file():
    assert load_prompt("v1").startswith("You are the help desk assistant")
    with pytest.raises(ValueError, match="v1"):
        load_prompt("no-such-prompt")


def test_build_rag_system_wires_the_config(monkeypatch):
    llm = ScriptedLLM(reply(answer="$10"))
    monkeypatch.setattr("deskrag.rag.llm.build_llm", lambda cfg: llm)
    monkeypatch.setattr("deskrag.rag.retrieval.build_retriever", lambda cfg: retriever())
    cfg = load_config(REPO / "configs" / "baseline.yaml")
    system = build_rag_system(cfg)
    assert system.llm is llm
    assert system.k == cfg.retrieval.k
    assert system.min_score == cfg.abstention.min_score
    assert system.system_prompt == load_prompt(cfg.prompt_version)
    assert system.llm_options["num_predict"] == cfg.llm.max_tokens
