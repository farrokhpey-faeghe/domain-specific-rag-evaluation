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


def run_cases(
    run_id: str,
    cases: list[TestCase],
    system,
    run_dir,
    reps: int = 1,
) -> RunSummary:
    """Evaluate cases and save results to a new run directory."""

    from pathlib import Path

    from deskrag.core.interfaces import InfraError
    from deskrag.core.runio import RunPaths, append_jsonl, write_json

    if reps < 1:
        raise ValueError("reps must be at least 1")

    if not cases:
        raise ValueError("No test cases were selected")

    case_ids = [case.case_id for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Test cases must have unique case IDs")

    paths = RunPaths(Path(run_dir))

    # Prevent accidentally overwriting an existing evaluation.
    paths.root.mkdir(parents=True, exist_ok=False)

    # Create both files even when one ends up empty.
    paths.results.touch()
    paths.errors.touch()

    rows: list[CaseResult] = []
    errors: list[ErrorRow] = []

    for case in cases:
        for rep in range(reps):
            try:
                response = system.answer(case.question)
                row = score_response(
                    run_id,
                    case,
                    response,
                    rep=rep,
                )
            except InfraError as exc:
                error = ErrorRow(
                    run_id=run_id,
                    case_id=case.case_id,
                    rep=rep,
                    error_class="infra",
                    message=str(exc),
                )
                append_jsonl(paths.errors, error)
                errors.append(error)
            except Exception as exc:
                error = ErrorRow(
                    run_id=run_id,
                    case_id=case.case_id,
                    rep=rep,
                    error_class="exception",
                    message=f"{type(exc).__name__}: {exc}",
                )
                append_jsonl(paths.errors, error)
                errors.append(error)
            else:
                append_jsonl(paths.results, row)
                rows.append(row)

            # Keep a current summary after each completed attempt.
            summary = summarize(run_id, rows, errors)
            write_json(paths.summary, summary)

    return summary
