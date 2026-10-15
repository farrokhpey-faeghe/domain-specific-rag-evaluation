"""What a run records about Ollama. The eval runner (WS4) writes env_fingerprint(cfg) to meta.json."""

from deskrag.core.config import SystemConfig
from deskrag.core.env import basic_fingerprint
from deskrag.core.schemas import EnvFingerprint
from deskrag.rag.llm.ollama_client import OllamaClient


def models_used(cfg: SystemConfig, *, retrieval_only: bool = False) -> list[str]:
    """The Ollama models a run calls: the LLM, plus the embedding model for dense and hybrid.

    A retrieval-only run never calls the LLM, so a BM25 one needs no Ollama at all.
    """
    models = []
    if not retrieval_only and cfg.system in ("rag", "closed_book") and cfg.llm is not None:
        models.append(cfg.llm.model)
    if cfg.system == "rag" and cfg.retrieval.method in ("dense", "hybrid"):
        if cfg.retrieval.embed_model:
            models.append(cfg.retrieval.embed_model)
    return models


def env_fingerprint(
    cfg: SystemConfig, client: OllamaClient | None = None, *, retrieval_only: bool = False
) -> EnvFingerprint:
    """Machine facts plus the Ollama version and model digests. No Ollama call if none is used.

    Pass `retrieval_only=True` for `eval run --retrieval-only`, so the CI retrieval gate can run
    BM25 on a machine without Ollama.
    """
    env = basic_fingerprint()
    models = models_used(cfg, retrieval_only=retrieval_only)
    if not models:
        return env
    client = client or OllamaClient(models[0])
    return env.model_copy(
        update={
            "ollama_version": client.server_version(),
            "model_digests": client.model_digests(models),
        }
    )
