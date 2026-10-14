"""What a run records about Ollama. The eval runner (WS4) writes env_fingerprint(cfg) to meta.json."""

from deskrag.core.config import SystemConfig
from deskrag.core.env import basic_fingerprint
from deskrag.core.schemas import EnvFingerprint
from deskrag.rag.llm.ollama_client import OllamaClient


def models_used(cfg: SystemConfig) -> list[str]:
    """The Ollama models a config calls: the LLM, plus the embedding model for dense and hybrid."""
    models = []
    if cfg.system in ("rag", "closed_book") and cfg.llm is not None:
        models.append(cfg.llm.model)
    if cfg.system == "rag" and cfg.retrieval.method in ("dense", "hybrid"):
        if cfg.retrieval.embed_model:
            models.append(cfg.retrieval.embed_model)
    return models


def env_fingerprint(cfg: SystemConfig, client: OllamaClient | None = None) -> EnvFingerprint:
    """Machine facts plus the Ollama version and model digests. No Ollama call if none is used."""
    env = basic_fingerprint()
    models = models_used(cfg)
    if not models:
        return env
    client = client or OllamaClient(models[0])
    return env.model_copy(
        update={
            "ollama_version": client.server_version(),
            "model_digests": client.model_digests(models),
        }
    )
