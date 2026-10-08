"""WS5 (feat/regression-ci): compare, report, baseline promote, runs list / runs table."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.commands import todo

OWNER = "WS5 (feat/regression-ci)"
GATES = Path("configs/regression.yaml")
RUNS_DIR = Path("runs")

RunDir = Annotated[Path, typer.Argument(help="Run directory, e.g. runs/<run_id>.")]

baseline_app = typer.Typer(help="Manage committed baselines (WS5).", no_args_is_help=True)
runs_app = typer.Typer(help="List and tabulate runs (WS5).", no_args_is_help=True)


def compare(
    baseline: RunDir,
    candidate: RunDir,
    gates: Annotated[Path, typer.Option(help="Per-metric tolerances.")] = GATES,
) -> None:
    """Metric deltas, McNemar p-values, and flipped cases. Exits 1 on a gated regression."""
    todo(OWNER, "compare")


def report(run_dir: RunDir) -> None:
    """Write report.md for a run: headline metrics, breakdowns, worst failures."""
    todo(OWNER, "report")


@baseline_app.command("promote")
def promote(
    run_dir: RunDir,
    name: Annotated[str, typer.Option(help="Baseline name under baselines/.")] = "dev",
) -> None:
    """Copy a run into baselines/<name>/. Commit it in a PR together with its compare.md."""
    todo(OWNER, "baseline promote")


@runs_app.command("list")
def list_runs(
    runs_dir: Annotated[Path, typer.Option(help="Where runs are stored.")] = RUNS_DIR,
) -> None:
    """List runs with their headline metrics, newest first."""
    todo(OWNER, "runs list")


@runs_app.command("table")
def table(
    run_dirs: Annotated[list[Path], typer.Argument(help="Runs to put side by side.")],
) -> None:
    """Markdown table of headline metrics, for the class report."""
    todo(OWNER, "runs table")


def register(app: typer.Typer) -> None:
    app.command("compare")(compare)
    app.command("report")(report)
    app.add_typer(baseline_app, name="baseline")
    app.add_typer(runs_app, name="runs")
