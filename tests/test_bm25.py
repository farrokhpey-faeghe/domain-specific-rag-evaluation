from deskrag.core.corpus import load_corpus
from deskrag.core.interfaces import Retriever
from deskrag.rag.chunking import chunk_document
from deskrag.rag.retrieval.bm25 import BM25Retriever, tokenize


def make_retriever() -> BM25Retriever:
    docs = load_corpus("data/samples/corpus")
    return BM25Retriever([c for d in docs for c in chunk_document(d)])


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Reset your Wi-Fi Password!") == ["reset", "your", "wi", "fi", "password"]


def test_it_is_a_retriever():
    assert isinstance(make_retriever(), Retriever)


def test_wifi_question_finds_the_wifi_page():
    top = make_retriever().retrieve("Which Wi-Fi network should campus visitors use?", k=3)
    assert top[0].chunk.doc_id == "it-wifi"


def test_password_question_finds_the_password_page():
    top = make_retriever().retrieve("How long is a password reset link valid?", k=3)
    assert top[0].chunk.doc_id == "it-password-reset"


def test_ranks_start_at_one_and_scores_never_increase():
    results = make_retriever().retrieve("parking permit cost", k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_the_number_of_results():
    assert len(make_retriever().retrieve("library", k=2)) == 2
