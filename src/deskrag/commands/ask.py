"""WS3 (feat/generation): `deskrag ask`."""

from pathlib import Path
from typing import Annotated

import typer

from deskrag.core.config import load_config
from deskrag.core.interfaces import InfraError
from deskrag.core.schemas import RAGAnswer
from deskrag.systems import build_system

BASELINE_CONFIG = Path("configs/baseline.yaml")

ABSTAIN_MEANING = {
    "low_retrieval_score": "no help page matched well enough to ask the model",
    "model_declined": "the model said it can't answer this from what it was given",
}


def ask(
    question: Annotated[str, typer.Argument(help="A question for the help desk.")],
    config: Annotated[Path, typer.Option(help="System config.")] = BASELINE_CONFIG,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the whole RAGAnswer as JSON.")
    ] = False,
) -> None:
    """Ask the chatbot one question. Prints the answer or abstention, citations, and top docs."""
    try:
        answer = build_system(load_config(config)).answer(question)
    except InfraError as exc:
        typer.echo(f"Ollama problem: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(answer.model_dump_json(indent=2) if json_output else format_answer(answer))


def format_answer(answer: RAGAnswer) -> str:
    if answer.abstained:
        lines = [f"Abstained ({answer.abstain_reason}): {ABSTAIN_MEANING[answer.abstain_reason]}."]
    elif answer.status == "ok":
        lines = [f"Answer: {answer.answer}", f"Cited: {', '.join(answer.cited_doc_ids) or '-'}"]
    else:
        lines = [
            f"No usable answer (status {answer.status}). The model said:",
            answer.raw_output or "",
        ]
    if answer.retrieved:
        lines += ["", "Retrieved:"]
        lines += [
            f"  {hit.rank:>2}  {hit.score:7.2f}  {hit.chunk.chunk_id}  {hit.chunk.title}"
            for hit in answer.retrieved
        ]
    usage = answer.usage
    lines += [
        "",
        f"{answer.status} | {answer.model or 'no LLM call'} | {usage.latency_s:.1f} s"
        f" | {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens",
    ]
    return "\n".join(lines)


def register(app: typer.Typer) -> None:
    app.command("ask")(ask)
