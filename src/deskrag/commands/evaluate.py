"""WS4 (feat/eval-metrics): `deskrag eval run` and `deskrag eval sweep`."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.commands import todo

OWNER = "WS4 (feat/eval-metrics)"
BASELINE_CONFIG = Path("configs/baseline.yaml")
TESTSET_DIR = Path("data/testset")

eval_app = typer.Typer(
    help="Run the test set through a system and score it (WS4).", no_args_is_help=True
)


@eval_app.command("run")
def run(
    config: Annotated[Path, typer.Option(help="System config.")] = BASELINE_CONFIG,
    split: Annotated[str, typer.Option(help="dev or test. Run test only once, at M3.")] = "dev",
    cases: Annotated[Path, typer.Option(help="Test-set file or directory.")] = TESTSET_DIR,
    limit: Annotated[int | None, typer.Option(help="Only the first N cases.")] = None,
    retrieval_only: Annotated[
        bool, typer.Option("--retrieval-only", help="Score retrieval only; no LLM calls.")
    ] = False,
    reps: Annotated[int, typer.Option(help="Repetitions per case.")] = 1,
) -> None:
    """Run every case and write runs/<run_id>/ (meta, results, errors, summary)."""
    todo(OWNER, "eval run")


@eval_app.command("sweep")
def sweep(
    run_dir: Annotated[Path, typer.Argument(help="A run made with the score gate turned off.")],
) -> None:
    """Abstention precision/recall at each gate threshold, from recorded scores (no LLM calls)."""
    todo(OWNER, "eval sweep")


def register(app: typer.Typer) -> None:
    app.add_typer(eval_app, name="eval")
