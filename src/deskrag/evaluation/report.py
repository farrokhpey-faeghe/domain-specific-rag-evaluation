"""Markdown reports from frozen run artifacts."""

from pathlib import Path

from deskrag.evaluation.compare import cell, load_run


def report_run(path: Path) -> str:
    run = load_run(path)
    lines = [
        "# Evaluation report",
        "",
        f"Run: `{cell(run.meta.run_id)}`",
        f"Split: {cell(run.meta.dataset.split)}; cases: {run.meta.dataset.n_cases}; reps: {run.meta.reps}",
        f"Dataset SHA256: `{run.meta.dataset.sha256}`",
        f"Config SHA256: `{run.meta.config_hash}`",
        f"Machine: {cell(run.meta.env.machine)}",
        "",
        "## Counts",
        "",
        "| Count | Value |",
        "|---|---:|",
    ]
    if run.meta.config.get("fixture_only"):
        lines.insert(
            2, "**Offline fixture demonstration. These scores are not real model performance.**"
        )
    lines += [f"| {k} | {v} |" for k, v in run.summary.counts.model_dump().items()]
    lines += ["", "## Metrics", "", "| Metric | Value |", "|---|---:|"]
    lines += [
        f"| {cell(k)} | {v:.6g} |" if v is not None else f"| {cell(k)} | n/a |"
        for k, v in sorted(run.summary.metrics.items())
    ]
    for kind, metrics in sorted(run.summary.by_case_type.items()):
        lines += ["", f"## {cell(kind)}", "", "| Metric | Value |", "|---|---:|"]
        lines += [
            f"| {cell(k)} | {v:.6g} |" if v is not None else f"| {cell(k)} | n/a |"
            for k, v in sorted(metrics.items())
        ]
    lines += ["", "## Failure counts", "", "| Tag | Count |", "|---|---:|"]
    lines += [f"| {cell(k)} | {v} |" for k, v in sorted(run.summary.failure_counts.items())]
    lines += [
        "",
        "## Failed cases",
        "",
        "| Case | Rep | Status | Failure tags |",
        "|---|---:|---|---|",
    ]
    lines += [
        f"| {cell(r.case_id)} | {r.rep} | {r.answer.status} | {cell(', '.join(r.failure_tags))} |"
        for r in run.rows.values()
        if not r.passed
    ]
    lines += [
        "",
        "Infrastructure errors are unscored and prevent comparison or baseline promotion.",
        "A report is descriptive. A separate comparison determines regression verdicts.",
    ]
    text = "\n".join(lines) + "\n"
    (path / "report.md").write_text(text, encoding="utf-8")
    return text


def runs_table(paths: list[Path]) -> str:
    runs = [load_run(path) for path in paths]
    metrics = sorted({m for run in runs for m in run.summary.metrics})
    lines = [
        "| Metric | " + " | ".join(cell(r.meta.run_id) for r in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for metric in metrics:
        values = [r.summary.metrics.get(metric) for r in runs]
        lines.append(
            "| "
            + cell(metric)
            + " | "
            + " | ".join("n/a" if v is None else f"{v:.6g}" for v in values)
            + " |"
        )
    return "\n".join(lines) + "\n"
