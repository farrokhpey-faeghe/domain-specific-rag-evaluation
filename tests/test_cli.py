import pytest
import typer
from typer.testing import CliRunner

from deskrag.cli import app
from deskrag.commands import NOT_IMPLEMENTED_EXIT_CODE, todo

runner = CliRunner()

# The command map from docs/design.md. Renaming a command means updating this list and the docs.
COMMANDS = [
    ["data", "snapshot"],
    ["data", "validate"],
    ["index", "build"],
    ["retrieve"],
    ["ask"],
    ["eval", "run"],
    ["eval", "sweep"],
    ["compare"],
    ["report"],
    ["baseline", "promote"],
    ["runs", "list"],
    ["runs", "table"],
]


def test_top_level_help_lists_every_workstreams_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "data",
        "index",
        "retrieve",
        "ask",
        "eval",
        "compare",
        "report",
        "baseline",
        "runs",
    ):
        assert name in result.output


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_every_command_has_working_help(command):
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output


def test_todo_says_who_owns_an_unfinished_command():
    demo = typer.Typer()

    @demo.command()
    def unfinished() -> None:
        todo("WS5 (feat/regression-ci)", "compare")

    result = runner.invoke(demo, [])
    assert result.exit_code == NOT_IMPLEMENTED_EXIT_CODE
    assert "WS5 (feat/regression-ci)" in result.output
