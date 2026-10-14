"""Opt-in on-disk cache of LLM responses: set DESKRAG_LLM_CACHE=1.

A hit replays the stored response, including its original latency. The cache is off by default
because `eval run --reps N` measures run-to-run variation, which replayed answers would hide, and
because cached latencies are copies, not measurements.
"""

import os
from pathlib import Path
from typing import Any

from deskrag.core.hashing import sha256_json
from deskrag.core.interfaces import LLMClient
from deskrag.core.schemas import LLMResponse

CACHE_DIR = Path(".cache/llm")


def cache_enabled() -> bool:
    return os.environ.get("DESKRAG_LLM_CACHE") == "1"


class CachedLLM:
    """Wraps any LLMClient. `identity` covers what the request alone doesn't, e.g. the model digest."""

    def __init__(self, inner: LLMClient, identity: dict[str, Any], cache_dir: Path = CACHE_DIR):
        self.inner = inner
        self.identity = identity
        self.cache_dir = cache_dir

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        key = sha256_json(
            {
                "identity": self.identity,
                "messages": messages,
                "json_schema": json_schema,
                "options": options,
            }
        )
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        response = self.inner.chat(messages, json_schema=json_schema, options=options)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".tmp")
        partial.write_text(response.model_dump_json(), encoding="utf-8")
        partial.replace(path)  # an interrupted write never leaves a half-written entry
        return response
