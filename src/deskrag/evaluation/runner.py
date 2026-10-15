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
    cases: list[TestCase] | None = None,
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

    by_case_type = {}
    if cases is not None:
        case_types = {case.case_id: case.case_type for case in cases}
        for case_type in sorted(set(case_types.values())):
            group = [row for row in rows if case_types[row.case_id] == case_type]
            grouped_metrics = {}
            for module in METRIC_MODULES:
                grouped_metrics.update(module.aggregate(group))
            by_case_type[case_type] = grouped_metrics

    return RunSummary(
        run_id=run_id,
        metrics=metrics,
        counts=counts,
        failure_counts=dict(failures),
        by_case_type=by_case_type,
    )


def _replace_errors(path, errors: list[ErrorRow]) -> None:
    """Atomically replace outstanding errors after a retry."""
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(error.model_dump_json() + "\n" for error in errors),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_cases(
    run_id: str,
    cases: list[TestCase],
    system,
    run_dir,
    reps: int = 1,
    *,
    config=None,
    dataset_path: str = "in-memory",
    resume: bool = False,
) -> RunSummary:
    """Save responses incrementally; resume only with matching metadata.

    Pass the actual SystemConfig as config to write meta.json and enable
    safe resume. Legacy callers without config can create new runs only.
    Resume retries failed and unfinished pairs, never completed results.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    from deskrag.core.config import config_hash
    from deskrag.core.corpus import corpus_sha256, load_corpus
    from deskrag.core.dataset import dataset_sha256
    from deskrag.core.env import basic_fingerprint, git_info
    from deskrag.core.interfaces import InfraError
    from deskrag.core.runio import (
        RunPaths,
        append_jsonl,
        read_json,
        read_jsonl,
        write_json,
    )
    from deskrag.core.schemas import CorpusRef, DatasetRef, RunMeta

    if reps < 1:
        raise ValueError("reps must be at least 1")
    if not cases:
        raise ValueError("No test cases were selected")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Test cases must have unique case IDs")
    if resume and config is None:
        raise ValueError("Resume requires config and matching meta.json")

    paths = RunPaths(Path(run_dir))
    fingerprint = dataset_sha256(cases)
    splits = {case.split for case in cases}
    split = next(iter(splits)) if len(splits) == 1 else "all"
    meta = None
    if config is not None:
        docs = load_corpus(config.corpus_dir)
        sha, dirty = git_info()
        environment = basic_fingerprint()
        if config.llm is not None or config.retrieval.embed_model is not None:
            # WS3 supplies Ollama version/model digests; avoid any model calls
            # for fake and sanity configs that use no model.
            from deskrag.rag.llm.env import env_fingerprint

            environment = env_fingerprint(config)
        meta = RunMeta(
            run_id=run_id,
            created_at=datetime.now(UTC),
            config=config.model_dump(mode="json"),
            config_hash=config_hash(config),
            git_sha=sha,
            git_dirty=dirty,
            dataset=DatasetRef(
                path=str(dataset_path),
                split=split,
                sha256=fingerprint,
                n_cases=len(cases),
            ),
            corpus=CorpusRef(
                path=config.corpus_dir,
                sha256=corpus_sha256(config.corpus_dir),
                n_docs=len(docs),
            ),
            env=environment,
            reps=reps,
        )

    if resume:
        if not paths.meta.is_file():
            raise ValueError("Cannot resume without meta.json")
        stored = read_json(paths.meta, RunMeta)
        for field in ("run_id", "config_hash", "reps"):
            if getattr(stored, field) != getattr(meta, field):
                raise ValueError(f"Cannot resume: {field} differs from meta.json")
        if (stored.dataset.sha256, stored.dataset.split, stored.dataset.n_cases) != (
            fingerprint,
            split,
            len(cases),
        ):
            raise ValueError("Cannot resume: dataset differs from meta.json")
        if stored.corpus != meta.corpus:
            raise ValueError("Cannot resume: corpus differs from meta.json")
        rows = read_jsonl(paths.results, CaseResult)
        saved_errors = read_jsonl(paths.errors, ErrorRow)
    else:
        paths.root.mkdir(parents=True, exist_ok=False)
        if meta is not None:
            write_json(paths.meta, meta)
        paths.results.touch()
        paths.errors.touch()
        rows = []
        saved_errors = []

    allowed = {(case.case_id, rep) for case in cases for rep in range(reps)}
    completed = set()
    for row in rows:
        key = (row.case_id, row.rep)
        if row.run_id != run_id or key not in allowed or key in completed:
            raise ValueError("Invalid or duplicate saved result")
        completed.add(key)
    outstanding = {}
    for error in saved_errors:
        key = (error.case_id, error.rep)
        if error.run_id != run_id or key not in allowed:
            raise ValueError("Invalid saved error")
        # A crash after appending a success may leave its old error behind.
        if key not in completed:
            outstanding[key] = error

    _replace_errors(paths.errors, list(outstanding.values()))
    summary = summarize(run_id, rows, list(outstanding.values()), cases=cases)
    write_json(paths.summary, summary)

    for case in cases:
        for rep in range(reps):
            key = (case.case_id, rep)
            if key in completed:
                continue
            previous = outstanding.get(key)
            attempts = previous.attempts + 1 if previous else 1
            try:
                response = system.answer(case.question)
            except Exception as exc:
                error_class = "infra" if isinstance(exc, InfraError) else "exception"
                outstanding[key] = ErrorRow(
                    run_id=run_id,
                    case_id=case.case_id,
                    rep=rep,
                    error_class=error_class,
                    message=f"{type(exc).__name__}: {exc}",
                    attempts=attempts,
                )
            else:
                # Scorer bugs must propagate, not become system error rows.
                row = score_response(run_id, case, response, rep=rep)
                # Persist success before removing its previous error.
                append_jsonl(paths.results, row)
                rows.append(row)
                completed.add(key)
                outstanding.pop(key, None)
            _replace_errors(paths.errors, list(outstanding.values()))
            summary = summarize(run_id, rows, list(outstanding.values()), cases=cases)
            write_json(paths.summary, summary)

    return summary
