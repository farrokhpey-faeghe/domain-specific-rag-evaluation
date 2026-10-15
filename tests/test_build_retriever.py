from pathlib import Path

from deskrag.core.config import RetrievalConfig, SystemConfig
from deskrag.core.interfaces import Retriever
from deskrag.rag.retrieval import build_retriever

REPO = Path(__file__).resolve().parents[1]
CORPUS = str(REPO / "data" / "samples" / "corpus")


def make_config(method: str) -> SystemConfig:
    return SystemConfig(
        name="t",
        system="fake",
        corpus_dir=CORPUS,
        retrieval=RetrievalConfig(method=method),
    )


def test_bm25_config_gives_a_working_retriever():
    retriever = build_retriever(make_config("bm25"))
    assert isinstance(retriever, Retriever)
    top = retriever.retrieve("Which Wi-Fi network should campus visitors use?", k=3)
    assert top[0].chunk.doc_id == "it-wifi"


def test_fake_config_gives_a_working_retriever():
    retriever = build_retriever(make_config("fake"))
    assert isinstance(retriever, Retriever)
    assert len(retriever.retrieve("parking permit", k=2)) == 2
