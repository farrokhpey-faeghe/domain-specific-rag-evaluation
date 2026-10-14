"""WS3 (feat/generation): the same LLM and reply rules, but with no retrieved context.

Used in the "does retrieval matter?" experiment: if closed-book scores close to the real system,
the questions can be answered from the model's memory and the test set needs harder cases. Give it
a prompt written for it (configs/experiments/closed-book.yaml uses closed-book-v1): prompt v1 says
to use only the context, so without context the model would abstain on everything.
"""

from deskrag.core.config import SystemConfig
from deskrag.rag.pipeline import RAGPipeline, load_prompt


def build_closed_book_system(cfg: SystemConfig) -> RAGPipeline:
    from deskrag.rag.llm import build_llm, llm_options

    return RAGPipeline(
        llm=build_llm(cfg),
        system_prompt=load_prompt(cfg.prompt_version),
        llm_options=llm_options(cfg.llm),
    )
