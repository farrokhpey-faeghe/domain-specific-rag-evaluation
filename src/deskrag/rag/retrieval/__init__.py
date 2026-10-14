"""WS2 (feat/retrieval): retrievers over the chunked corpus.

Planned modules: bm25.py (rank-bm25), dense.py (Ollama /api/embed + numpy cosine, index cached in
.cache/index/), hybrid.py (reciprocal-rank fusion). Chunking lives in rag/chunking.py; chunk ids are
always f"{doc_id}#{position}".
"""

from deskrag.core.config import SystemConfig
from deskrag.core.interfaces import Retriever
from deskrag.core.schemas import Chunk, Document


def chunk_corpus(docs: list[Document], cfg: SystemConfig) -> list[Chunk]:
    """Split every document into chunks using the config's chunking settings.

    The one place that does this, so `index build` and `build_retriever` always agree.
    """
    from deskrag.rag.chunking import chunk_document

    return [
        chunk
        for doc in docs
        for chunk in chunk_document(doc, cfg.chunking.size_words, cfg.chunking.overlap_words)
    ]


def build_retriever(cfg: SystemConfig) -> Retriever:
    # Imports are inside the function so loading this package stays cheap.
    from deskrag.core.corpus import load_corpus
    from deskrag.core.fakes import FakeRetriever

    docs = load_corpus(cfg.corpus_dir)
    method = cfg.retrieval.method

    if method == "fake":
        return FakeRetriever(docs)
    if method == "bm25":
        from deskrag.rag.retrieval.bm25 import BM25Retriever

        return BM25Retriever(chunk_corpus(docs, cfg))
    raise NotImplementedError(f"retrieval method {method!r} is not built yet (M2)")
