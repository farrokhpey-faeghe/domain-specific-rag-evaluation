"""System configuration: which chatbot variant to build and how.

Configs live in `configs/*.yaml`. A config can start with `extends: <other>.yaml` (resolved
relative to its own file) and override only what differs; nested sections are deep-merged.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from deskrag.core.hashing import sha256_json
from deskrag.core.schemas import Contract

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "rag" / "prompts"

SystemKind = Literal["rag", "fake", "oracle", "always_abstain", "never_abstain", "closed_book"]


class ChunkingConfig(Contract):
    size_words: int = Field(default=300, gt=0)
    overlap_words: int = Field(default=50, ge=0)


class RetrievalConfig(Contract):
    method: Literal["bm25", "dense", "hybrid", "fake"] = "bm25"
    k: int = Field(default=5, gt=0)
    embed_model: str | None = None  # Ollama embedding model for dense/hybrid


class AbstentionConfig(Contract):
    # Gate: abstain without calling the LLM when the top retrieval score is below this.
    # The scale depends on the retrieval method. None turns the gate off.
    min_score: float | None = None


class LLMConfig(Contract):
    model: str  # exact Ollama tag; each run also records the model digest
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    max_tokens: int = 512
    think: bool = False


class SystemConfig(Contract):
    name: str
    system: SystemKind = "rag"
    corpus_dir: str = "data/corpus/docs"  # POSIX-style, relative to the repo root
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    llm: LLMConfig | None = None
    prompt_version: str = "v1"  # file name in src/deskrag/rag/prompts/

    @model_validator(mode="after")
    def _llm_backed_systems_need_llm(self) -> "SystemConfig":
        if self.system in {"rag", "closed_book"} and self.llm is None:
            raise ValueError(f"system {self.system!r} needs an llm section")
        return self


def load_config(path: str | Path) -> SystemConfig:
    return SystemConfig.model_validate(_load_with_extends(Path(path)))


def config_hash(cfg: SystemConfig, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Fingerprint of what the system does: every setting except `name`, plus the prompt text."""
    prompt_file = prompts_dir / f"{cfg.prompt_version}.md"
    prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else None
    return sha256_json({"config": cfg.model_dump(mode="json", exclude={"name"}), "prompt": prompt})


def _load_with_extends(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _deep_merge(_load_with_extends(path.parent / parent), data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
