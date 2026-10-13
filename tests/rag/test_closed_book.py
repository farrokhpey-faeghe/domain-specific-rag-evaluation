from pathlib import Path

from deskrag.core.config import load_config
from deskrag.core.fakes import FakeLLM
from deskrag.rag.pipeline import RAGPipeline, load_prompt
from deskrag.systems import build_system

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "closed-book.yaml"


def test_the_closed_book_config_builds_a_pipeline_with_no_retrieval_or_gate(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("deskrag.rag.llm.build_llm", lambda cfg: llm)
    cfg = load_config(CONFIG)
    system = build_system(cfg)
    assert isinstance(system, RAGPipeline)
    assert system.retriever is None
    assert system.min_score is None
    assert system.llm is llm
    assert system.system_prompt == load_prompt("closed-book-v1")
    assert system.llm_options["seed"] == cfg.llm.seed


def test_the_closed_book_prompt_lets_the_model_answer_from_memory():
    prompt = load_prompt("closed-book-v1")
    assert "University of Toledo" in prompt
    assert "No documents are provided" in prompt
