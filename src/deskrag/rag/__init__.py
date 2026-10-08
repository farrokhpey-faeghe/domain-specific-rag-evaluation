"""The chatbot under test.

- `chunking.py`, `retrieval/` (WS2, feat/retrieval): split docs into chunks; BM25, dense, hybrid retrievers.
- `llm/`, `prompts/`, `pipeline.py`, `abstention.py`, `closed_book.py` (WS3, feat/generation):
  Ollama client + cache, versioned prompts, the retrieve -> gate -> generate -> parse pipeline.
"""
