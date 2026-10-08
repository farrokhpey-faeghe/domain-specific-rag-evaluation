"""WS3 (feat/generation): `deskrag ask`."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.commands import todo

OWNER = "WS3 (feat/generation)"
BASELINE_CONFIG = Path("configs/baseline.yaml")


def ask(
    question: Annotated[str, typer.Argument(help="A question for the help desk.")],
    config: Annotated[Path, typer.Option(help="System config.")] = BASELINE_CONFIG,
) -> None:
    """Ask the chatbot one question. Prints the answer or abstention, citations, and top docs."""
    todo(OWNER, "ask")


def register(app: typer.Typer) -> None:
    app.command("ask")(ask)
