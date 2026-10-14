"""WS2 (feat/retrieval): `deskrag index build` and `deskrag retrieve`."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.core.config import load_config
from deskrag.core.corpus import load_corpus
from deskrag.rag.retrieval import build_retriever, chunk_corpus

BASELINE_CONFIG = Path("configs/baseline.yaml")

ConfigOption = Annotated[Path, typer.Option(help="System config.")]

index_app = typer.Typer(help="Build retrieval indexes (WS2).", no_args_is_help=True)


@index_app.command("build")
def build(config: ConfigOption = BASELINE_CONFIG) -> None:
    """Chunk the corpus and build (or refresh) the index the config's retriever needs."""
    cfg = load_config(config)
    docs = load_corpus(cfg.corpus_dir)
    chunks = chunk_corpus(docs, cfg)
    build_retriever(cfg)  # builds the index once, to check it works
    typer.echo(f"{len(docs)} docs -> {len(chunks)} chunks (method: {cfg.retrieval.method})")


def retrieve(
    query: Annotated[str, typer.Argument(help="Question to search for.")],
    config: ConfigOption = BASELINE_CONFIG,
    k: Annotated[int, typer.Option("-k", help="How many chunks to show.")] = 5,
) -> None:
    """Print the top-k chunks for QUERY: rank, score, doc_id, and a snippet."""
    retriever = build_retriever(load_config(config))
    for hit in retriever.retrieve(query, k):
        snippet = " ".join(hit.chunk.text.split())[:120]
        typer.echo(f"{hit.rank}. {hit.score:.2f}  {hit.chunk.chunk_id}  {snippet}")


def register(app: typer.Typer) -> None:
    app.add_typer(index_app, name="index")
    app.command("retrieve")(retrieve)
