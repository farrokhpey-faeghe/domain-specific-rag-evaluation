"""The Ollama `LLMClient`, plus the server facts a run records (version, model digests).

Calls ask for JSON-schema output (`format=`) with thinking off. Sampling options (temperature,
seed, context size, max tokens) arrive with each call, so the response cache can key on them.
"""

import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx
import ollama

from deskrag.core.interfaces import InfraError
from deskrag.core.schemas import LLMResponse

# What the ollama library raises when the server misbehaves: ResponseError for HTTP errors,
# ConnectionError when nothing is listening, and httpx transport errors (timeouts) as they are.
_RETRYABLE = (ollama.ResponseError, ConnectionError, httpx.TransportError)


def normalize_model_name(name: str) -> str:
    """Ollama treats 'llama3.2' and 'llama3.2:latest' as the same model."""
    return name if ":" in name else f"{name}:latest"


class OllamaClient:
    """`LLMClient` for one Ollama model. Retries failed requests, then raises InfraError."""

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        think: bool = False,
        timeout_s: float = 120.0,
        retries: int = 2,
        backoff_s: float = 1.0,
        client: Any = None,  # an ollama.Client; tests pass a stub
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.think = think
        self._client = client or ollama.Client(host=host, timeout=timeout_s)
        self._retries = retries
        self._backoff_s = backoff_s
        self._sleep = sleep

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        response = self._request(
            "chat",
            lambda: self._client.chat(
                model=self.model,
                messages=messages,
                format=json_schema,
                options=options,
                think=self.think,
                stream=False,
            ),
        )
        served = response.model or ""
        if normalize_model_name(served) != normalize_model_name(self.model):
            raise InfraError(f"asked Ollama for {self.model!r}, but {served!r} answered")
        return LLMResponse(
            text=response.message.content or "",
            model=served,
            done_reason=response.done_reason,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
            latency_s=time.perf_counter() - start,
        )

    def server_version(self) -> str:
        # The ollama library has no version call, so reuse its HTTP client for GET /api/version.
        response = self._request(
            "version", lambda: self._client._request_raw("GET", "/api/version")
        )
        return response.json()["version"]

    def model_digests(self, models: Iterable[str]) -> dict[str, str]:
        """Digest of each model, keyed by the name given. InfraError if one isn't pulled."""
        pulled = {
            normalize_model_name(entry.model): entry.digest
            for entry in self._request("list models", self._client.list).models
            if entry.model and entry.digest
        }
        digests = {}
        for model in models:
            if normalize_model_name(model) not in pulled:
                raise InfraError(f"model {model!r} is not pulled; run `ollama pull {model}`")
            digests[model] = pulled[normalize_model_name(model)]
        return digests

    def _request(self, what: str, send: Callable[[], Any]) -> Any:
        attempt = 1
        while True:
            try:
                return send()
            except _RETRYABLE as exc:
                if attempt > self._retries:
                    raise InfraError(
                        f"Ollama {what} failed after {attempt} attempts: {exc}"
                    ) from exc
                self._sleep(self._backoff_s * 2 ** (attempt - 1))
                attempt += 1
