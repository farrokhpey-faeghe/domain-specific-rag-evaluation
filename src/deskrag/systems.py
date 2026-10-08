"""Build the system a config asks for.

Imports are lazy so each workstream's builder lives in its own package and nobody has to edit
this file: `rag` and `closed_book` belong to WS3, the sanity systems to WS4, `fake` to core.
"""

from deskrag.core.config import SystemConfig
from deskrag.core.interfaces import RAGSystem
from deskrag.core.schemas import TestCase


def build_system(cfg: SystemConfig, cases: list[TestCase] | None = None) -> RAGSystem:
    """`cases` is required for the sanity systems (oracle needs the gold answers)."""
    if cfg.system == "fake":
        from deskrag.core.fakes import FakeRAGSystem

        return FakeRAGSystem.from_corpus_dir(cfg.corpus_dir)
    if cfg.system == "rag":
        from deskrag.rag.pipeline import build_rag_system

        return build_rag_system(cfg)
    if cfg.system == "closed_book":
        from deskrag.rag.closed_book import build_closed_book_system

        return build_closed_book_system(cfg)

    # oracle, always_abstain, never_abstain
    if cases is None:
        raise ValueError(f"system {cfg.system!r} needs the test cases (pass cases=...)")
    from deskrag.evaluation.sanity import build_sanity_system

    return build_sanity_system(cfg, cases=cases)
