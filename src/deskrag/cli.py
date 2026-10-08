"""The `deskrag` command. Each workstream registers its commands from deskrag/commands/<module>.py."""

import typer

from deskrag.commands import ask, data, evaluate, regress, retrieval

app = typer.Typer(
    help="University help-desk RAG chatbot and the harness that evaluates it.",
    no_args_is_help=True,
)

for module in (data, retrieval, ask, evaluate, regress):
    module.register(app)
