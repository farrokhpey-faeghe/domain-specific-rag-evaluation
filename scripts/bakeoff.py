"""Model bake-off (WS3, Day 1): run candidate models through the real pipeline and compare them.

    # the 10 sample questions, retrieved by FakeRetriever over the sample corpus
    uv run python scripts/bakeoff.py --cases data/samples/cases.jsonl --split all \
        --corpus data/samples/corpus --models llama3.2:3b gemma3:4b
    # the real dev cases, retrieved by the config's retriever (needs WS2's build_retriever)
    uv run python scripts/bakeoff.py --cases data/testset --models llama3.2:3b gemma3:4b

The score gate is off, so each model's own abstention is measured. The scoring is a rough
stand-in for WS4's metrics, good enough to pick a model. It prints a markdown table.
"""

import argparse
import json
import statistics
from pathlib import Path

from deskrag.core.config import SystemConfig, load_config
from deskrag.core.corpus import load_corpus
from deskrag.core.dataset import load_cases
from deskrag.core.fakes import FakeRetriever
from deskrag.core.interfaces import InfraError, Retriever
from deskrag.core.schemas import RAGAnswer, TestCase
from deskrag.rag.llm import llm_options
from deskrag.rag.llm.ollama_client import OllamaClient
from deskrag.rag.pipeline import RAGPipeline, load_prompt

COLUMNS = [
    "model",
    "e2e pass",
    "answerable: correct",
    "unanswerable: abstained",
    "false abstain",
    "cites retrieved docs",
    "format errors",
    "infra errors",
    "s/q p50",
    "s/q max",
    "prompt tokens",
]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def passed(case: TestCase, answer: RAGAnswer) -> bool:
    """Unanswerable: abstained. Answerable: answered (status ok) with every must_include fact."""
    if not case.answerable:
        return answer.abstained
    if answer.abstained or answer.status != "ok":
        return False
    text = normalize(answer.answer or "")
    return all(any(normalize(alt) in text for alt in fact.split("|")) for fact in case.must_include)


def run_model(
    model: str, cases: list[TestCase], retriever: Retriever, cfg: SystemConfig
) -> list[dict]:
    llm = OllamaClient(model, think=cfg.llm.think)
    llm.model_digests([model])  # fail fast if it isn't pulled
    llm.chat([{"role": "user", "content": "Reply OK."}], options={"num_predict": 1})  # load it
    pipeline = RAGPipeline(
        llm=llm,
        system_prompt=load_prompt(cfg.prompt_version),
        retriever=retriever,
        k=cfg.retrieval.k,
        llm_options=llm_options(cfg.llm),
    )
    rows = []
    for case in cases:
        try:
            answer = pipeline.answer(case.question)
        except InfraError as exc:
            rows.append({"model": model, "case_id": case.case_id, "error": str(exc)})
            print(f"{model:32} {case.case_id:10} ERROR {exc}", flush=True)
            continue
        rows.append(
            {
                "model": model,
                "case_id": case.case_id,
                "case_type": case.case_type,
                "answerable": case.answerable,
                "passed": passed(case, answer),
                "status": answer.status,
                "abstained": answer.abstained,
                "answer": answer.answer,
                "cited": answer.cited_doc_ids,
                "retrieved": sorted({hit.chunk.doc_id for hit in answer.retrieved}),
                "gold": case.gold_doc_ids,
                "latency_s": round(answer.usage.latency_s, 2),
                "prompt_tokens": answer.usage.prompt_tokens,
                "raw_output": answer.raw_output,
            }
        )
        verdict = "pass" if rows[-1]["passed"] else "FAIL"
        print(f"{model:32} {case.case_id:10} {verdict} {answer.usage.latency_s:5.1f}s", flush=True)
    return rows


def summarize(model: str, rows: list[dict]) -> str:
    done = [row for row in rows if "error" not in row]
    answerable = [row for row in done if row["answerable"]]
    unanswerable = [row for row in done if not row["answerable"]]
    answered = [row for row in done if row["status"] == "ok" and not row["abstained"]]
    latencies = [row["latency_s"] for row in done]

    def share(part: list, whole: list) -> str:
        return f"{len(part)}/{len(whole)}"

    cells = [
        model,
        share([row for row in done if row["passed"]], done),
        share([row for row in answerable if row["passed"]], answerable),
        share([row for row in unanswerable if row["passed"]], unanswerable),
        share([row for row in answerable if row["abstained"]], answerable),
        share(
            [
                row
                for row in answered
                if row["cited"] and set(row["cited"]) <= set(row["retrieved"])
            ],
            answered,
        ),
        share([row for row in done if row["status"] != "ok"], done),
        str(len(rows) - len(done)),
        f"{statistics.median(latencies):.1f}" if latencies else "-",
        f"{max(latencies):.1f}" if latencies else "-",
        f"{statistics.mean(row['prompt_tokens'] for row in done):.0f}" if done else "-",
    ]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate Ollama models on the pipeline.")
    parser.add_argument("--models", nargs="+", required=True, help="Ollama tags to compare.")
    parser.add_argument("--cases", type=Path, default=Path("data/testset"))
    parser.add_argument("--split", default="dev", help="dev, test, or all. Never tune on test.")
    parser.add_argument("--corpus", type=Path, help="Use FakeRetriever over this corpus.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--out", type=Path, help="Also write every answer to this JSONL file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cases = load_cases(args.cases, split=None if args.split == "all" else args.split)
    if args.corpus:
        retriever = FakeRetriever(load_corpus(args.corpus))
    else:
        from deskrag.rag.retrieval import build_retriever  # WS2

        retriever = build_retriever(cfg)
    results = {model: run_model(model, cases, retriever, cfg) for model in args.models}

    if args.out:
        with args.out.open("w", encoding="utf-8") as file:
            for rows in results.values():
                file.writelines(json.dumps(row) + "\n" for row in rows)
    print()
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for model, rows in results.items():
        print(summarize(model, rows))


if __name__ == "__main__":
    main()
