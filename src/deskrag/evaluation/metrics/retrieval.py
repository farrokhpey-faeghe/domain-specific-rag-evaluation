"""Document-level retrieval metrics for WS4."""

from deskrag.core.schemas import CaseResult, RAGAnswer, TestCase

METRIC_NAMES = ("hit@1", "hit@3", "hit@5", "recall@5", "mrr")


def score_case(
    case: TestCase,
    answer: RAGAnswer,
) -> dict[str, float | None]:
    """Calculate retrieval scores for one question."""

    # Unanswerable questions have no gold documents.
    if not case.answerable:
        return {name: None for name in METRIC_NAMES}

    # Keep the document ranking, removing repeated documents.
    ranked_docs: list[str] = []
    seen: set[str] = set()

    for item in answer.retrieved:
        doc_id = item.chunk.doc_id

        if doc_id not in seen:
            ranked_docs.append(doc_id)
            seen.add(doc_id)

    gold_docs = set(case.gold_doc_ids)

    # A hit means at least one gold document appears in the top k.
    scores: dict[str, float | None] = {}

    for k in (1, 3, 5):
        top_docs = set(ranked_docs[:k])
        found_gold = top_docs & gold_docs
        scores[f"hit@{k}"] = float(bool(found_gold))

    # Recall measures the fraction of gold documents found.
    top_five = set(ranked_docs[:5])
    scores["recall@5"] = len(top_five & gold_docs) / len(gold_docs)

    # Reciprocal rank uses the first relevant document's position.
    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in gold_docs:
            reciprocal_rank = 1.0 / rank
            break

    scores["mrr"] = reciprocal_rank

    return scores


def aggregate(
    rows: list[CaseResult],
) -> dict[str, float | None]:
    """Average each retrieval metric, excluding unavailable scores."""

    averages: dict[str, float | None] = {}

    for name in METRIC_NAMES:
        values: list[float] = []

        for row in rows:
            value = row.scores.get(name)

            if value is not None:
                values.append(value)

        averages[name] = sum(values) / len(values) if values else None

    return averages
