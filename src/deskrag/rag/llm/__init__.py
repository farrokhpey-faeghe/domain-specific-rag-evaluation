"""WS3 (feat/generation): talking to the LLM.

- ollama_client.py: `OllamaClient`, the LLMClient for Ollama (JSON-schema output, thinking off,
  retries, then InfraError), plus the server version and model digests a run records.
- cache.py: `CachedLLM`, an opt-in on-disk response cache (DESKRAG_LLM_CACHE=1).
- env.py: `env_fingerprint(cfg)`, what the eval runner writes to meta.json as `env`.
"""

from typing import Any

from deskrag.core.config import LLMConfig, SystemConfig
from deskrag.core.interfaces import LLMClient


def llm_options(llm: LLMConfig) -> dict[str, Any]:
    """Ollama options for every call. They travel with each call, so the cache key covers them."""
    return {
        "temperature": llm.temperature,
        "seed": llm.seed,
        "num_ctx": llm.num_ctx,
        "num_predict": llm.max_tokens,
    }


def build_llm(cfg: SystemConfig) -> LLMClient:
    """The config's Ollama model, wrapped in the response cache when DESKRAG_LLM_CACHE=1.

    Raises InfraError right away if Ollama is unreachable or the model isn't pulled, so a run fails
    at the start instead of on every case.
    """
    # Imports are inside the function so loading this package stays cheap.
    from deskrag.rag.llm.cache import CachedLLM, cache_enabled
    from deskrag.rag.llm.ollama_client import OllamaClient

    if cfg.llm is None:
        raise ValueError(f"config {cfg.name!r} has no llm section")
    client = OllamaClient(cfg.llm.model, think=cfg.llm.think)
    digest = client.model_digests([cfg.llm.model])[cfg.llm.model]
    if not cache_enabled():
        return client
    identity = {
        "model": cfg.llm.model,
        "digest": digest,
        "ollama_version": client.server_version(),
        "think": cfg.llm.think,
    }
    return CachedLLM(client, identity)
