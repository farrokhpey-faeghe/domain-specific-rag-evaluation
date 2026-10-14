"""Simple reference systems that test the evaluation harness."""

from deskrag.core.config import SystemConfig
from deskrag.core.interfaces import RAGSystem
from deskrag.core.schemas import (
    Chunk,
    RAGAnswer,
    RetrievedChunk,
    TestCase,
)

SANITY_KINDS = {"oracle", "always_abstain", "never_abstain"}


class SanitySystem:
    """Return deterministic responses based on a chosen policy."""

    def __init__(self, kind: str, cases: list[TestCase]):
        if kind not in SANITY_KINDS:
            raise ValueError(f"Unknown sanity system: {kind}")

        self.kind = kind
        self.cases_by_question: dict[str, TestCase] = {}

        for case in cases:
            if case.question in self.cases_by_question:
                raise ValueError(f"Duplicate question in sanity cases: {case.question}")

            self.cases_by_question[case.question] = case

    def answer(self, question: str) -> RAGAnswer:
        """Answer a labeled question according to the selected policy."""

        case = self.cases_by_question.get(question)

        if case is None:
            raise ValueError(f"Question not found in sanity cases: {question}")

        should_decline = self.kind == "always_abstain" or (
            self.kind == "oracle" and not case.answerable
        )

        if should_decline:
            return RAGAnswer(
                question=question,
                abstained=True,
                abstain_reason="model_declined",
            )

        if not case.answerable:
            # Deliberately answer even though no reference answer exists.
            return RAGAnswer(
                question=question,
                abstained=False,
                answer="This policy answers without supporting information.",
            )

        # Make synthetic evidence from the reference labels.
        # This is evaluator test data, not real retrieval.
        retrieved = []

        for rank, doc_id in enumerate(case.gold_doc_ids, start=1):
            chunk = Chunk(
                chunk_id=f"{doc_id}#0",
                doc_id=doc_id,
                title=doc_id,
                text=case.reference_answer or "",
                position=0,
            )

            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=1.0,
                    rank=rank,
                )
            )

        return RAGAnswer(
            question=question,
            abstained=False,
            answer=case.reference_answer,
            cited_doc_ids=list(case.gold_doc_ids),
            retrieved=retrieved,
            confidence=1.0,
        )


def build_sanity_system(
    cfg: SystemConfig,
    cases: list[TestCase],
) -> RAGSystem:
    """Build the sanity system selected in the configuration."""
    return SanitySystem(cfg.system, cases)
