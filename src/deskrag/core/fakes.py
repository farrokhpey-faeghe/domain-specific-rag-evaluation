"""Deterministic stand-ins, so every workstream can build and test without Ollama or real data.

`FakeRAGSystem` answers the 10 questions in data/samples/cases.jsonl with canned outcomes chosen
to exercise every metric and failure path. Gate abstentions behave as if the score gate were 0.5.

| case | split | designed outcome                                                              |
|------|-------|-------------------------------------------------------------------------------|
| s01  | dev   | correct answer, gold doc at rank 1                                    -> pass |
| s02  | dev   | correct answer, gold doc at rank 2 (reciprocal rank 0.5)              -> pass |
| s03  | dev   | multi-doc, answer has 1 of 2 required facts                   -> WRONG_ANSWER |
| s04  | test  | gold doc not retrieved, model declines -> RETRIEVAL_MISS, FALSE_ABSTAIN_MODEL |
| s05  | dev   | gold doc at rank 1 but top score 0.4 trips the gate     -> FALSE_ABSTAIN_GATE |
| s06  | test  | model output is not JSON (status parse_error)                 -> FORMAT_ERROR |
| s07  | dev   | out of domain, the gate abstains                                      -> pass |
| s08  | dev   | missing info, the bot invents a printing price              -> MISSED_ABSTAIN |
| s09  | dev   | false premise, the model declines                                     -> pass |
| s10  | test  | personal info, invents a GPA, cites an unretrieved doc -> MISSED_ABSTAIN, BAD_CITATION |

Overall 4/10 cases pass (dev 4/7, test 0/3).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deskrag.core.corpus import load_corpus
from deskrag.core.schemas import (
    AbstainReason,
    AnswerStatus,
    Chunk,
    Document,
    LLMResponse,
    RAGAnswer,
    RetrievedChunk,
    Usage,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are at can do does for how i in is it my of on or the to what when where which who "
    "why with you your".split()
)
ABSTAIN_REPLY = json.dumps({"abstain": True, "answer": "", "citations": []})


def whole_doc_chunk(doc: Document) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#0", doc_id=doc.doc_id, title=doc.title, text=doc.text, position=0
    )


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS}


class FakeRetriever:
    """Ranks whole documents by how many distinct query words they contain. For tests only."""

    def __init__(self, docs: list[Document]):
        self._chunks = [whole_doc_chunk(doc) for doc in docs]
        self._tokens = {
            chunk.chunk_id: _tokens(f"{chunk.title} {chunk.text}") for chunk in self._chunks
        }

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        query_tokens = _tokens(query)
        scored = sorted(
            ((len(query_tokens & self._tokens[chunk.chunk_id]), chunk) for chunk in self._chunks),
            key=lambda pair: (-pair[0], pair[1].doc_id),
        )
        return [
            RetrievedChunk(chunk=chunk, score=float(score), rank=rank)
            for rank, (score, chunk) in enumerate(scored[:k], start=1)
        ]


class FakeLLM:
    """Returns the canned reply whose key appears in the last user message; otherwise abstains."""

    def __init__(self, replies: dict[str, str] | None = None, model: str = "fake-llm"):
        self.replies = dict(replies or {})
        self.model = model
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = next((reply for key, reply in self.replies.items() if key in prompt), ABSTAIN_REPLY)
        return LLMResponse(
            text=text,
            model=self.model,
            done_reason="stop",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_s=0.0,
        )


@dataclass(frozen=True)
class CannedOutcome:
    retrieved: tuple[tuple[str, float], ...]  # (doc_id, score), best first
    answer: str | None = None
    cited: tuple[str, ...] = ()
    abstain_reason: AbstainReason | None = None
    status: AnswerStatus = "ok"
    raw_output: str | None = None


SAMPLE_OUTCOMES: dict[str, CannedOutcome] = {
    "How much does an electronic transcript cost?": CannedOutcome(
        retrieved=(("registrar-transcripts", 3.2), ("parking-permits", 1.1), ("it-wifi", 0.3)),
        answer="Electronic transcripts cost $10 each.",
        cited=("registrar-transcripts",),
    ),
    "Which Wi-Fi network should campus visitors use?": CannedOutcome(
        retrieved=(("it-password-reset", 2.1), ("it-wifi", 1.9), ("library-hours", 0.2)),
        answer="Visitors should join NU-Guest.",
        cited=("it-wifi",),
    ),
    "Do I buy a parking permit in the same portal where I order transcripts?": CannedOutcome(
        retrieved=(("registrar-transcripts", 2.6), ("parking-permits", 2.4), ("it-wifi", 0.1)),
        answer="Transcripts are ordered in the student portal.",
        cited=("registrar-transcripts",),
    ),
    "is the library open late on fridays": CannedOutcome(
        retrieved=(("parking-permits", 0.9), ("it-wifi", 0.7), ("it-password-reset", 0.5)),
        abstain_reason="model_declined",
    ),
    "How long is a password reset link valid?": CannedOutcome(
        retrieved=(("it-password-reset", 0.4), ("it-wifi", 0.1), ("library-hours", 0.0)),
        abstain_reason="low_retrieval_score",
    ),
    "How much is a semester parking permit?": CannedOutcome(
        retrieved=(
            ("parking-permits", 2.8),
            ("registrar-transcripts", 0.6),
            ("library-hours", 0.1),
        ),
        status="parse_error",
        raw_output="Sure! A semester permit is $180.",
    ),
    "What's a good recipe for banana bread?": CannedOutcome(
        retrieved=(("library-hours", 0.1), ("it-wifi", 0.0), ("parking-permits", 0.0)),
        abstain_reason="low_retrieval_score",
    ),
    "How much does it cost to print a page at the library?": CannedOutcome(
        retrieved=(
            ("library-hours", 1.2),
            ("registrar-transcripts", 0.8),
            ("parking-permits", 0.7),
        ),
        answer="Printing at the library costs $0.10 per page.",
        cited=("library-hours",),
    ),
    "When does the library on the Mars campus open on Sundays?": CannedOutcome(
        retrieved=(("library-hours", 1.5), ("it-wifi", 0.2), ("parking-permits", 0.1)),
        abstain_reason="model_declined",
    ),
    "What is my current GPA?": CannedOutcome(
        retrieved=(
            ("registrar-transcripts", 0.9),
            ("it-password-reset", 0.2),
            ("library-hours", 0.1),
        ),
        answer="Your current GPA is 3.4.",
        cited=("student-records",),
    ),
}


class FakeRAGSystem:
    """Canned answers for the sample questions (see the module docstring); declines anything else."""

    def __init__(self, docs: list[Document], outcomes: dict[str, CannedOutcome] = SAMPLE_OUTCOMES):
        self._chunks = {doc.doc_id: whole_doc_chunk(doc) for doc in docs}
        self._outcomes = outcomes

    @classmethod
    def from_corpus_dir(cls, corpus_dir: str | Path) -> "FakeRAGSystem":
        return cls(load_corpus(corpus_dir))

    def answer(self, question: str) -> RAGAnswer:
        outcome = self._outcomes.get(question) or CannedOutcome(
            retrieved=(), abstain_reason="model_declined"
        )
        retrieved = [
            RetrievedChunk(chunk=self._chunks[doc_id], score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(outcome.retrieved, start=1)
        ]
        called_llm = outcome.abstain_reason != "low_retrieval_score"
        return RAGAnswer(
            question=question,
            abstained=outcome.abstain_reason is not None,
            abstain_reason=outcome.abstain_reason,
            answer=outcome.answer,
            cited_doc_ids=list(outcome.cited),
            retrieved=retrieved,
            confidence=retrieved[0].score if retrieved else None,
            status=outcome.status,
            raw_output=outcome.raw_output or (_model_reply(outcome) if called_llm else None),
            usage=(
                Usage(prompt_tokens=420, completion_tokens=24, latency_s=0.6, llm_calls=1)
                if called_llm
                else Usage(latency_s=0.02)
            ),
            model="fake-llm" if called_llm else None,
        )


def _model_reply(outcome: CannedOutcome) -> str:
    return json.dumps(
        {
            "abstain": outcome.abstain_reason is not None,
            "answer": outcome.answer or "",
            "citations": list(outcome.cited),
        }
    )
