"""WS5: regression comparison, reports, baselines, run tables and integration."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.evaluation.baseline import evaluate_candidate, promote_baseline
from deskrag.evaluation.compare import compare_runs, load_run
from deskrag.evaluation.report import report_run, runs_table

GATES = Path("configs/regression.yaml")
RunDir = Annotated[Path, typer.Argument(help="Run directory.")]
baseline_app = typer.Typer(help="Manage committed baselines.", no_args_is_help=True)
runs_app = typer.Typer(help="List and tabulate runs.", no_args_is_help=True)


def fail(exc: Exception) -> None:
    typer.echo(f"Invalid comparison/run: {exc}", err=True)
    raise typer.Exit(2) from exc


def compare(
    baseline: RunDir,
    candidate: RunDir,
    gates: Annotated[Path, typer.Option(help="Metric tolerances.")] = GATES,
) -> None:
    """Write candidate/compare.md; exit 1 on regression, 2 on invalid inputs."""
    try:
        text, blocked = compare_runs(baseline, candidate, gates)
    except (ValueError, OSError) as exc:
        fail(exc)
    typer.echo(text)
    if blocked:
        raise typer.Exit(1)


def report(run_dir: RunDir) -> None:
    """Write report.md with metrics, breakdowns and failures."""
    try:
        typer.echo(report_run(run_dir))
    except (ValueError, OSError) as exc:
        fail(exc)


@baseline_app.command("promote")
def promote(
    run_dir: RunDir, name: Annotated[str, typer.Option(help="Name under baselines/.")] = "dev"
) -> None:
    """Stage a complete dev run. Commit and review it through a PR."""
    try:
        typer.echo(str(promote_baseline(run_dir, name)))
    except (ValueError, OSError) as exc:
        fail(exc)


@runs_app.command("list")
def list_runs(runs_dir: Annotated[Path, typer.Option(help="Run root.")] = Path("runs")) -> None:
    """List runs newest first; flag malformed runs rather than hiding them."""
    for path in sorted(runs_dir.glob("*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            run = load_run(path)
            typer.echo(
                f"{run.meta.run_id}\t{run.meta.dataset.split}\te2e_pass={run.summary.metrics.get('e2e_pass')}\terrors={run.summary.counts.errors}"
            )
        except (ValueError, OSError) as exc:
            typer.echo(f"INVALID {path}: {exc}", err=True)


@runs_app.command("table")
def table(run_dirs: Annotated[list[Path], typer.Argument(help="Runs to compare.")]) -> None:
    """Print Markdown results table for the report."""
    try:
        typer.echo(runs_table(run_dirs))
    except (ValueError, OSError) as exc:
        fail(exc)


def regress(
    config: Annotated[Path, typer.Option()] = Path("configs/baseline.yaml"),
    baseline: Annotated[Path, typer.Option()] = Path("baselines/dev"),
    candidate: Annotated[
        Path | None, typer.Option(help="Use a recorded run instead of evaluating.")
    ] = None,
    gates: Annotated[Path, typer.Option()] = GATES,
    retrieval_only: Annotated[bool, typer.Option()] = False,
) -> None:
    """Evaluate on dev, report and compare. Requires WS2/WS4 for a live evaluation."""
    try:
        path = candidate or evaluate_candidate(config, retrieval_only)
        report_run(path)
    except (ValueError, OSError) as exc:
        fail(exc)
    compare(baseline, path, gates)


def register(app: typer.Typer) -> None:
    app.command("compare")(compare)
    app.command("report")(report)
    app.command("regress")(regress)
    app.add_typer(baseline_app, name="baseline")
    app.add_typer(runs_app, name="runs")
