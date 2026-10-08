"""WS3 (feat/generation): the real chatbot.

retrieve (WS2's build_retriever) -> score gate (abstention.min_score) -> prompt (prompts/<version>.md)
-> Ollama chat with a JSON schema {abstain, answer, citations} -> parse -> RAGAnswer.

Rules from the contract (core/schemas.py): gate abstentions make no LLM call and use
abstain_reason="low_retrieval_score"; the model declining uses "model_declined"; output that is not
valid JSON is status="parse_error"; done_reason == "length" is status="truncated"; Ollama being
unreachable raises InfraError.
"""

from deskrag.core.config import SystemConfig
from deskrag.core.interfaces import RAGSystem


def build_rag_system(cfg: SystemConfig) -> RAGSystem:
    raise NotImplementedError(
        "WS3 (feat/generation): build the retrieve -> gate -> generate pipeline"
    )
