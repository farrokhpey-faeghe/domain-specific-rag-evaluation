"""Generate offline demo run records, not official model evaluation results.

Run from repo root: uv run python tests/regression/generate_fixtures.py
"""

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from deskrag.core.config import config_hash, load_config
from deskrag.core.corpus import corpus_sha256, load_corpus
from deskrag.core.dataset import dataset_sha256, load_cases
from deskrag.core.fakes import FakeRAGSystem
from deskrag.core.runio import append_jsonl, write_json
from deskrag.core.schemas import (
    CaseResult,
    CorpusRef,
    DatasetRef,
    EnvFingerprint,
    RAGAnswer,
    RunCounts,
    RunMeta,
    RunSummary,
)

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent / "fixtures"


def write_fixture(path, run_id, cases, answers, passes, hits, *, synthetic=False):
    path.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs/fake.yaml")
    docs = load_corpus(REPO / "data/samples/corpus")
    meta = RunMeta(
        run_id=run_id,
        created_at=datetime(2026, 10, 12, tzinfo=UTC),
        config={**cfg.model_dump(mode="json"), "fixture_only": True, "synthetic": synthetic},
        config_hash=config_hash(cfg),
        git_sha=None,
        git_dirty=False,
        dataset=DatasetRef(
            path="fixture://synthetic" if synthetic else "data/samples/cases.jsonl",
            split="dev",
            sha256="synthetic-fixture-v1" if synthetic else dataset_sha256(cases),
            n_cases=len(cases),
        ),
        corpus=CorpusRef(
            path="data/samples/corpus",
            sha256=corpus_sha256(REPO / "data/samples/corpus"),
            n_docs=len(docs),
        ),
        env=EnvFingerprint(machine="fixture", platform="offline", python="3.12"),
        reps=1,
    )
    write_json(path / "meta.json", meta)
    (path / "results.jsonl").write_text("", encoding="utf-8")
    (path / "errors.jsonl").write_text("", encoding="utf-8")
    failures = Counter()
    for case, answer, passed, hit in zip(cases, answers, passes, hits, strict=True):
        scores = {"e2e_pass": float(passed), "hit@5": hit, "abstain_correct": float(passed)}
        tags = [] if passed else ["WRONG_ANSWER"]
        failures.update(tags)
        append_jsonl(
            path / "results.jsonl",
            CaseResult(
                run_id=run_id,
                case_id=case.case_id,
                answer=answer,
                scores=scores,
                passed=passed,
                failure_tags=tags,
            ),
        )
    metrics = {
        "e2e_pass": sum(passes) / len(cases),
        "abstain_correct": sum(passes) / len(cases),
        "hit@5": sum(h for h in hits if h is not None) / sum(h is not None for h in hits),
    }
    write_json(
        path / "summary.json",
        RunSummary(
            run_id=run_id,
            metrics=metrics,
            counts=RunCounts(
                cases=len(cases), ok=len(cases), errors=0, truncated=0, parse_errors=0
            ),
            failure_counts=dict(failures),
        ),
    )


def main():
    cases = load_cases(REPO / "data/samples/cases.jsonl", split="dev")
    system = FakeRAGSystem.from_corpus_dir(REPO / "data/samples/corpus")
    answers = [system.answer(c.question) for c in cases]
    passes = [c.case_id in {"s01", "s02", "s07", "s09"} for c in cases]
    hits = [
        float(any(r.chunk.doc_id in c.gold_doc_ids for r in a.retrieved)) if c.answerable else None
        for c, a in zip(cases, answers, strict=True)
    ]
    write_fixture(ROOT / "fake-baseline", "fake-baseline", cases, answers, passes, hits)
    flipped = [
        False if c.case_id in {"s01", "s02", "s07"} else p
        for c, p in zip(cases, passes, strict=True)
    ]
    # Scores deliberately flipped to test WS5 reading already-scored WS4 artifacts.
    write_fixture(ROOT / "fake-candidate", "fake-candidate", cases, answers, flipped, hits)
    synthetic = [cases[0].model_copy(update={"case_id": f"demo-{i:02}"}) for i in range(20)]
    answer = RAGAnswer(question="Synthetic gate fixture", abstained=False, answer="Fixture only")
    write_fixture(
        ROOT / "gate-baseline",
        "gate-baseline",
        synthetic,
        [answer] * 20,
        [True] * 12 + [False] * 8,
        [1.0] * 20,
        synthetic=True,
    )
    write_fixture(
        ROOT / "gate-candidate",
        "gate-candidate",
        synthetic,
        [answer] * 20,
        [True] * 4 + [False] * 16,
        [1.0] * 20,
        synthetic=True,
    )


if __name__ == "__main__":
    main()
