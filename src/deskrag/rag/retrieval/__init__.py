"""WS2 (feat/retrieval): retrievers over the chunked corpus.

Planned modules: bm25.py (rank-bm25), dense.py (Ollama /api/embed + numpy cosine, index cached in
.cache/index/), hybrid.py (reciprocal-rank fusion). Chunking lives in rag/chunking.py; chunk ids are
always f"{doc_id}#{position}".
"""

from deskrag.core.config import SystemConfig
from deskrag.core.interfaces import Retriever


def build_retriever(cfg: SystemConfig) -> Retriever:
    raise NotImplementedError(
        "WS2 (feat/retrieval): return the retriever named by cfg.retrieval.method"
    )
