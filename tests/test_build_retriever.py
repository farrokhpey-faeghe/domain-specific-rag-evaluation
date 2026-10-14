from deskrag.core.config import RetrievalConfig, SystemConfig
from deskrag.core.interfaces import Retriever
from deskrag.rag.retrieval import build_retriever


def make_config(method: str) -> SystemConfig:
    return SystemConfig(
        name="t",
        system="fake",
        corpus_dir="data/samples/corpus",
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
