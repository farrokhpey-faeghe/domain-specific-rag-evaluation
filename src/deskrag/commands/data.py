"""WS1 (feat/data): `deskrag data snapshot` and `deskrag data validate`."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.commands import todo

OWNER = "WS1 (feat/data)"
SOURCES_CSV = Path("data/corpus/sources.csv")
CORPUS_DIR = Path("data/corpus/docs")
TESTSET_DIR = Path("data/testset")

data_app = typer.Typer(
    help="Snapshot the help pages and validate the test set (WS1).", no_args_is_help=True
)


@data_app.command()
def snapshot(
    sources: Annotated[Path, typer.Option(help="CSV with doc_id,url,section.")] = SOURCES_CSV,
    out: Annotated[Path, typer.Option(help="One <doc_id>.md per page goes here.")] = CORPUS_DIR,
) -> None:
    """Fetch every page in SOURCES into markdown with front-matter (1 request/s, respects robots.txt)."""
    todo(OWNER, "data snapshot")


@data_app.command()
def validate(
    cases: Annotated[
        Path, typer.Option(help="Test-set file or directory of *.jsonl.")
    ] = TESTSET_DIR,
    corpus: Annotated[Path, typer.Option(help="Corpus directory.")] = CORPUS_DIR,
) -> None:
    """Check the test set against the corpus: gold docs exist, facts appear in gold docs, balance."""
    todo(OWNER, "data validate")


def register(app: typer.Typer) -> None:
    app.add_typer(data_app, name="data")
