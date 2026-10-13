from pathlib import Path

from typer.testing import CliRunner

from deskrag.cli import app
from deskrag.core.interfaces import InfraError
from deskrag.core.schemas import RAGAnswer

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()


def ask(monkeypatch, *args):
    monkeypatch.chdir(REPO)  # corpus_dir in configs is relative to the repo root
    return runner.invoke(app, ["ask", *args, "--config", "configs/fake.yaml"])


def test_prints_the_answer_citations_and_retrieved_chunks(monkeypatch):
    result = ask(monkeypatch, "How much does an electronic transcript cost?")
    assert result.exit_code == 0, result.output
    assert "Answer: Electronic transcripts cost $10 each." in result.output
    assert "Cited: registrar-transcripts" in result.output
    assert "registrar-transcripts#0" in result.output


def test_explains_an_abstention(monkeypatch):
    result = ask(monkeypatch, "What's a good recipe for banana bread?")
    assert result.exit_code == 0, result.output
    assert "Abstained (low_retrieval_score)" in result.output


def test_shows_the_raw_output_when_the_reply_is_unusable(monkeypatch):
    result = ask(monkeypatch, "How much is a semester parking permit?")
    assert "No usable answer (status parse_error)" in result.output
    assert "Sure! A semester permit is $180." in result.output


def test_json_prints_the_whole_answer(monkeypatch):
    result = ask(monkeypatch, "How much does an electronic transcript cost?", "--json")
    assert RAGAnswer.model_validate_json(result.output).cited_doc_ids == ["registrar-transcripts"]


def test_an_ollama_problem_exits_1_with_the_reason(monkeypatch):
    class OllamaDown:
        def answer(self, question):
            raise InfraError("cannot reach Ollama")

    monkeypatch.setattr("deskrag.commands.ask.build_system", lambda cfg: OllamaDown())
    result = ask(monkeypatch, "anything")
    assert result.exit_code == 1
    assert "cannot reach Ollama" in result.output
