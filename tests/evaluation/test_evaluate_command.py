"""Exercise evaluation through the public CLI without Ollama."""

from pathlib import Path

from typer.testing import CliRunner

from deskrag.cli import app
from deskrag.core.runio import read_json, read_jsonl
from deskrag.core.schemas import CaseResult, RunMeta, RunSummary

REPO = Path(__file__).resolve().parents[2]


def test_cli_writes_complete_run_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    target = tmp_path / "cli-run"
    args = [
        "eval",
        "run",
        "--config",
        "configs/fake.yaml",
        "--cases",
        "data/samples/cases.jsonl",
        "--run-dir",
        str(target),
    ]
    cli = CliRunner()
    result = cli.invoke(app, args)
    assert result.exit_code == 0, result.output
    meta = read_json(target / "meta.json", RunMeta)
    summary = read_json(target / "summary.json", RunSummary)
    assert meta.dataset.split == "dev"
    assert meta.dataset.n_cases == 7
    assert summary.metrics["e2e_pass"] == 4 / 7
    assert summary.by_case_type
    assert len(read_jsonl(target / "results.jsonl", CaseResult)) == 7
    before = (target / "results.jsonl").read_bytes()
    resumed = cli.invoke(app, args + ["--resume"])
    assert resumed.exit_code == 0, resumed.output
    assert (target / "results.jsonl").read_bytes() == before


def test_cli_rejects_resume_without_directory():
    result = CliRunner().invoke(app, ["eval", "run", "--resume"])
    assert result.exit_code == 2
    assert "requires --run-dir" in result.output
