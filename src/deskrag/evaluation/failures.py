"""Explain evaluation failures using the project's shared failure tags."""

from deskrag.core.schemas import FailureTag, RAGAnswer, TestCase


def tag_failures(
    case: TestCase,
    answer: RAGAnswer,
    scores: dict[str, float | None],
) -> list[FailureTag]:
    """Return all applicable failure tags for one response."""

    tags: list[FailureTag] = []

    # Retrieval can fail independently of the model's output status.
    if case.answerable:
        retrieved_docs = {item.chunk.doc_id for item in answer.retrieved}
        gold_docs = set(case.gold_doc_ids)

        if not (retrieved_docs & gold_docs):
            tags.append("RETRIEVAL_MISS")

    # Broken output is neither a valid answer nor an abstention.
    if answer.status != "ok":
        tags.append("FORMAT_ERROR")
        return tags

    if answer.abstained:
        if case.answerable:
            if answer.abstain_reason == "low_retrieval_score":
                tags.append("FALSE_ABSTAIN_GATE")
            else:
                tags.append("FALSE_ABSTAIN_MODEL")

        return tags

    # The remaining responses are valid, non-abstained answers.
    if not case.answerable:
        tags.append("MISSED_ABSTAIN")
    elif scores.get("answer_correct") == 0.0:
        tags.append("WRONG_ANSWER")

    retrieved_docs = {item.chunk.doc_id for item in answer.retrieved}
    cited_docs = set(answer.cited_doc_ids)

    if not cited_docs.issubset(retrieved_docs):
        tags.append("BAD_CITATION")

    return tags
