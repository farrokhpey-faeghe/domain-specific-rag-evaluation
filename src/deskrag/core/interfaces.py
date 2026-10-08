"""The seams between workstreams. Any class with these methods fits; no inheritance needed."""

from typing import Any, Protocol, runtime_checkable

from deskrag.core.schemas import LLMResponse, RAGAnswer, RetrievedChunk


class InfraError(Exception):
    """Plumbing failed (Ollama unreachable, timeout after retries).

    The eval runner logs it to errors.jsonl and never scores it as a wrong answer.
    """


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """The top-k chunks for `query`, best first, with 1-based ranks."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """One chat completion. Raise InfraError when the server can't be reached."""
        ...


@runtime_checkable
class RAGSystem(Protocol):
    def answer(self, question: str) -> RAGAnswer:
        """Answer or abstain. Raise InfraError for plumbing failures; never return them."""
        ...
