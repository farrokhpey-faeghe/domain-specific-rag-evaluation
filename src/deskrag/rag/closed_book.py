"""WS3 (feat/generation): the same LLM and prompt rules, but with no retrieved context.

Used in the "does retrieval matter?" experiment: if closed-book scores close to the real system,
the questions can be answered from the model's memory and the test set needs harder cases.
"""

from deskrag.core.config import SystemConfig
from deskrag.core.interfaces import RAGSystem


def build_closed_book_system(cfg: SystemConfig) -> RAGSystem:
    raise NotImplementedError("WS3 (feat/generation): answer with the LLM alone, no retrieval")
