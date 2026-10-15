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
    reps: Annotated[int, typer.Option(min=1, help="Repetitions per case.")] = 1,
    run_dir: Annotated[Path | None, typer.Option(help="Output directory for this run.")] = None,
    resume: Annotated[
        bool, typer.Option(help="Resume --run-dir; skip completed responses.")
    ] = False,
) -> None:
    """Run every case and write runs/<run_id>/ (meta, results, errors, summary)."""
    if retrieval_only:
        todo(OWNER, "eval run --retrieval-only (M2)")
    if split not in {"dev", "test"}:
        raise typer.BadParameter("Choose dev or test", param_hint="--split")
    if limit is not None and limit < 1:
        raise typer.BadParameter("Must be at least 1", param_hint="--limit")
    if resume and run_dir is None:
        raise typer.BadParameter("Resume requires --run-dir", param_hint="--resume")

    from deskrag.core.config import config_hash, load_config
    from deskrag.core.dataset import load_cases
    from deskrag.core.runio import make_run_id
    from deskrag.evaluation.runner import run_cases
    from deskrag.systems import build_system

    cfg = load_config(config)
    selected = load_cases(cases, split=split)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise typer.BadParameter("No cases selected", param_hint="--cases / --split")
    if resume:
        from deskrag.core.runio import read_json
        from deskrag.core.schemas import RunMeta

        run_id = read_json(run_dir / "meta.json", RunMeta).run_id
    else:
        run_id = make_run_id(cfg.name, config_hash(cfg))
    destination = run_dir if run_dir is not None else Path("runs") / run_id
    system = build_system(cfg, cases=selected)
    summary = run_cases(
        run_id,
        selected,
        system,
        destination,
        reps=reps,
        config=cfg,
        dataset_path=str(cases),
        resume=resume,
    )
    typer.echo(f"Run saved: {destination}")
    typer.echo(f"e2e_pass: {summary.metrics['e2e_pass']}")
    if summary.counts.errors:
        typer.echo(f"{summary.counts.errors} unresolved error(s); resume to retry.", err=True)
        raise typer.Exit(1)


@eval_app.command("sweep")
def sweep(
    run_dir: Annotated[Path, typer.Argument(help="A run made with the score gate turned off.")],
) -> None:
    """Abstention precision/recall at each gate threshold, from recorded scores (no LLM calls)."""
    todo(OWNER, "eval sweep")


def register(app: typer.Typer) -> None:
    app.add_typer(eval_app, name="eval")
