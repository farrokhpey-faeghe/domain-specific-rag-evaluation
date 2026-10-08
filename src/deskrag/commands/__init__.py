"""CLI commands, one module per workstream, each exposing `register(app)`:

    data.py       WS1  data snapshot | data validate
    retrieval.py  WS2  index build, retrieve
    ask.py        WS3  ask
    evaluate.py   WS4  eval run | eval sweep
    regress.py    WS5  compare, report, baseline promote, runs list | runs table

Owners replace the `todo(...)` call in each command with the real implementation.
"""

from typing import NoReturn

import typer

NOT_IMPLEMENTED_EXIT_CODE = 3


def todo(owner: str, command: str) -> NoReturn:
    typer.echo(
        f"`deskrag {command}` is not implemented yet. Owner: {owner}. See docs/workplan.md.",
        err=True,
    )
    raise typer.Exit(code=NOT_IMPLEMENTED_EXIT_CODE)
