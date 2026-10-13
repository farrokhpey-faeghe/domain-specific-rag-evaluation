"""BM25 keyword retriever: scores chunks by how well they match the words in the question."""

import re

from rank_bm25 import BM25Okapi

from deskrag.core.schemas import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep only letters and digits. 'Wi-Fi Password' -> ['wi', 'fi', 'password']."""
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        # The title is searchable too: a question about "transcripts" should find that page.
        self._index = BM25Okapi([tokenize(f"{c.title} {c.text}") for c in chunks])

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self._index.get_scores(tokenize(query))
        # Best score first; ties broken by chunk_id so results are always the same order.
        ranked = sorted(
            zip(scores, self._chunks), key=lambda pair: (-pair[0], pair[1].chunk_id)
        )
        return [
            RetrievedChunk(chunk=chunk, score=float(score), rank=rank)
            for rank, (score, chunk) in enumerate(ranked[:k], start=1)
        ]