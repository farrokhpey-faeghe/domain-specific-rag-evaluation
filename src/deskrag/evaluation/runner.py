"""Connect metric scoring and failure tracking for evaluation runs."""

from collections import Counter

from deskrag.core.schemas import (
    CaseResult,
    ErrorRow,
    RAGAnswer,
    RunCounts,
    RunSummary,
    TestCase,
)
from deskrag.evaluation.failures import tag_failures
from deskrag.evaluation.metrics import abstention, answer, ops, retrieval

METRIC_MODULES = (retrieval, abstention, answer, ops)


def score_response(
    run_id: str,
    case: TestCase,
    response: RAGAnswer,
    rep: int = 0,
) -> CaseResult:
    """Combine all metrics and failure tags for one response."""

    scores: dict[str, float | None] = {}

    for module in METRIC_MODULES:
        scores.update(module.score_case(case, response))

    return CaseResult(
        run_id=run_id,
        case_id=case.case_id,
        rep=rep,
        answer=response,
        scores=scores,
        passed=scores["e2e_pass"] == 1.0,
        failure_tags=tag_failures(case, response, scores),
    )


def summarize(
    run_id: str,
    rows: list[CaseResult],
    errors: list[ErrorRow],
) -> RunSummary:
    """Combine scored responses into a run summary."""

    metrics: dict[str, float | None] = {}

    for module in METRIC_MODULES:
        metrics.update(module.aggregate(rows))

    failures = Counter(tag for row in rows for tag in row.failure_tags)

    counts = RunCounts(
        cases=len({row.case_id for row in rows + errors}),
        ok=sum(row.answer.status == "ok" for row in rows),
        errors=len(errors),
        truncated=sum(row.answer.status == "truncated" for row in rows),
        parse_errors=sum(row.answer.status == "parse_error" for row in rows),
    )

    return RunSummary(
        run_id=run_id,
        metrics=metrics,
        counts=counts,
        failure_counts=dict(failures),
    )
