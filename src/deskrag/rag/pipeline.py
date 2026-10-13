"""WS3 (feat/generation): the real chatbot.

retrieve (WS2's build_retriever) -> score gate (abstention.min_score) -> prompt (prompts/<version>.md)
-> Ollama chat with a JSON schema {abstain, answer, citations} -> parse -> RAGAnswer.

Parse rules, in order (docs/ws3-generation.md has them as a table):
- gate on and top score below min_score: abstain with "low_retrieval_score", no LLM call;
- done_reason == "length": status="truncated";
- not JSON, or a key missing or of the wrong type: status="parse_error";
- "abstain": true: abstain with "model_declined";
- "abstain": false with an empty answer: status="parse_error" (no answer is not an abstention);
- otherwise status="ok". Citations are not checked against the retrieved docs, so BAD_CITATION
  stays measurable.
Ollama being unreachable raises InfraError from the LLM client.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from deskrag.core.config import PROMPTS_DIR, SystemConfig
from deskrag.core.interfaces import LLMClient, Retriever
from deskrag.core.schemas import LLMResponse, RAGAnswer, RetrievedChunk, Usage

# Property order matters: the model decides whether to abstain before it writes an answer.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["abstain", "answer", "citations"],
}


class ModelReply(BaseModel):
    """The reply ANSWER_SCHEMA asks for. Strict, so "abstain": "no" is a parse error."""

    model_config = ConfigDict(strict=True)

    abstain: bool
    answer: str
    citations: list[str]


def load_prompt(version: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    path = prompts_dir / f"{version}.md"
    if not path.exists():
        available = sorted(p.stem for p in prompts_dir.glob("*.md"))
        raise ValueError(f"no prompt {version!r} in {prompts_dir}; available: {available}")
    return path.read_text(encoding="utf-8").strip()


def build_messages(
    system_prompt: str, question: str, retrieved: list[RetrievedChunk]
) -> list[dict[str, str]]:
    """The system prompt, then the context blocks and the question. No chunks: just the question."""
    user = f"Question: {question}"
    if retrieved:
        blocks = "\n\n".join(
            f"[{hit.chunk.doc_id}] {hit.chunk.title}\n{hit.chunk.text}" for hit in retrieved
        )
        user = f"Context documents:\n\n{blocks}\n\n{user}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def clean_citations(citations: list[str]) -> list[str]:
    """'[it002]' and 'it002#3' both mean doc it002, and duplicates go. The rest is kept as written."""
    cleaned: list[str] = []
    for citation in citations:
        doc_id = citation.strip().strip("[]").strip().split("#")[0]
        if doc_id and doc_id not in cleaned:
            cleaned.append(doc_id)
    return cleaned


def parse_reply(response: LLMResponse) -> dict[str, Any]:
    """The RAGAnswer fields that come from the model's reply (rules in the module docstring)."""
    if response.done_reason == "length":
        return {"abstained": False, "status": "truncated"}
    try:
        reply = ModelReply.model_validate_json(response.text)
    except ValidationError:
        return {"abstained": False, "status": "parse_error"}
    if reply.abstain:
        return {"abstained": True, "abstain_reason": "model_declined"}
    if not reply.answer.strip():
        return {"abstained": False, "status": "parse_error"}
    return {
        "abstained": False,
        "answer": reply.answer.strip(),
        "cited_doc_ids": clean_citations(reply.citations),
    }


@dataclass
class RAGPipeline:
    """Answers or abstains. Without a retriever it is closed-book: no context and no gate."""

    llm: LLMClient
    system_prompt: str
    retriever: Retriever | None = None
    k: int = 5
    min_score: float | None = None  # None turns the gate off
    llm_options: dict[str, Any] = field(default_factory=dict)

    def answer(self, question: str) -> RAGAnswer:
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(question, self.k) if self.retriever is not None else []
        confidence = retrieved[0].score if retrieved else None
        gated = self.retriever is not None and self.min_score is not None
        if gated and (confidence is None or confidence < self.min_score):
            return RAGAnswer(
                question=question,
                abstained=True,
                abstain_reason="low_retrieval_score",
                retrieved=retrieved,
                confidence=confidence,
                usage=Usage(latency_s=time.perf_counter() - start),
            )
        response = self.llm.chat(
            build_messages(self.system_prompt, question, retrieved),
            json_schema=ANSWER_SCHEMA,
            options=self.llm_options,
        )
        return RAGAnswer(
            question=question,
            **parse_reply(response),
            retrieved=retrieved,
            confidence=confidence,
            raw_output=response.text,
            usage=Usage(
                prompt_tokens=response.prompt_tokens or 0,
                completion_tokens=response.completion_tokens or 0,
                latency_s=time.perf_counter() - start,
                llm_calls=1,
            ),
            model=response.model,
        )


def build_rag_system(cfg: SystemConfig) -> RAGPipeline:
    # Imports are inside the function so loading this module stays cheap.
    from deskrag.rag.llm import build_llm, llm_options
    from deskrag.rag.retrieval import build_retriever

    llm = build_llm(cfg)  # first, so a missing model fails before the index is built
    return RAGPipeline(
        llm=llm,
        system_prompt=load_prompt(cfg.prompt_version),
        retriever=build_retriever(cfg),
        k=cfg.retrieval.k,
        min_score=cfg.abstention.min_score,
        llm_options=llm_options(cfg.llm),
    )
