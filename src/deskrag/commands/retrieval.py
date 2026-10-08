"""WS2 (feat/retrieval): `deskrag index build` and `deskrag retrieve`."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.commands import todo

OWNER = "WS2 (feat/retrieval)"
BASELINE_CONFIG = Path("configs/baseline.yaml")

ConfigOption = Annotated[Path, typer.Option(help="System config.")]

index_app = typer.Typer(help="Build retrieval indexes (WS2).", no_args_is_help=True)


@index_app.command("build")
def build(config: ConfigOption = BASELINE_CONFIG) -> None:
    """Chunk the corpus and build (or refresh) the index the config's retriever needs."""
    todo(OWNER, "index build")


def retrieve(
    query: Annotated[str, typer.Argument(help="Question to search for.")],
    config: ConfigOption = BASELINE_CONFIG,
    k: Annotated[int, typer.Option("-k", help="How many chunks to show.")] = 5,
) -> None:
    """Print the top-k chunks for QUERY: rank, score, doc_id, and a snippet."""
    todo(OWNER, "retrieve")


def register(app: typer.Typer) -> None:
    app.add_typer(index_app, name="index")
    app.command("retrieve")(retrieve)
