"""Required-fact, citation, and overall response metrics for WS4."""

from deskrag.core.schemas import CaseResult, RAGAnswer, TestCase

METRIC_NAMES = (
    "fact_recall",
    "answer_correct",
    "citation_valid",
    "citation_hit",
    "e2e_pass",
)


def _normalize(text: str) -> str:
    """Lowercase text and collapse repeated whitespace."""
    return " ".join(text.lower().split())


def _fact_present(fact: str, answer_text: str) -> bool:
    """Check whether any pipe-separated alternative appears."""
    alternatives = fact.split("|")

    for alternative in alternatives:
        normalized = _normalize(alternative)

        if normalized and normalized in answer_text:
            return True

    return False


def score_case(
    case: TestCase,
    answer: RAGAnswer,
) -> dict[str, float | None]:
    """Score answer quality and overall success for one question."""

    scores: dict[str, float | None] = {name: None for name in METRIC_NAMES}

    # Broken output fails overall and has no valid answer to score.
    if answer.status != "ok":
        scores["e2e_pass"] = 0.0
        return scores

    # Declining is correct overall only for unanswerable questions.
    if answer.abstained:
        scores["e2e_pass"] = float(not case.answerable)
        return scores

    retrieved_docs = {item.chunk.doc_id for item in answer.retrieved}
    cited_docs = set(answer.cited_doc_ids)
    gold_docs = set(case.gold_doc_ids)

    # Every citation must refer to a retrieved document.
    scores["citation_valid"] = float(cited_docs.issubset(retrieved_docs))

    # At least one citation must match a gold document.
    scores["citation_hit"] = float(bool(cited_docs & gold_docs))

    # Answering an unanswerable question fails overall.
    if not case.answerable:
        scores["e2e_pass"] = 0.0
        return scores

    answer_text = _normalize(answer.answer or "")

    facts_found = sum(_fact_present(fact, answer_text) for fact in case.must_include)

    scores["fact_recall"] = facts_found / len(case.must_include)
    all_facts_present = facts_found == len(case.must_include)

    scores["answer_correct"] = float(all_facts_present)
    scores["e2e_pass"] = float(all_facts_present)

    return scores


def aggregate(
    rows: list[CaseResult],
) -> dict[str, float | None]:
    """Average applicable answer metrics across result records."""

    metrics: dict[str, float | None] = {}

    for name in METRIC_NAMES:
        values: list[float] = []

        for row in rows:
            value = row.scores.get(name)

            if value is not None:
                values.append(value)

        metrics[name] = sum(values) / len(values) if values else None

    return metrics
