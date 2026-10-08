from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from deskrag.core.schemas import (
    ANSWERABLE_TYPES,
    METRIC_DIRECTION,
    CaseResult,
    Chunk,
    CorpusRef,
    DatasetRef,
    Document,
    EnvFingerprint,
    ErrorRow,
    RAGAnswer,
    RetrievedChunk,
    RunCounts,
    RunMeta,
    RunSummary,
    TestCase,
    Usage,
)


def make_chunk(doc_id: str = "it-wifi", position: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{position}",
        doc_id=doc_id,
        title="Wi-Fi",
        text="Guests use NU-Guest.",
        position=position,
    )


def make_case(**overrides) -> TestCase:
    fields = dict(
        case_id="s01",
        question="How much is an electronic transcript?",
        answerable=True,
        case_type="single_doc",
        gold_doc_ids=["registrar-transcripts"],
        reference_answer="Each electronic transcript costs $10.",
        must_include=["$10|10 dollars"],
        split="dev",
        author="omar",
    )
    fields.update(overrides)
    return TestCase(**fields)


# --- RAGAnswer: an abstention is a deliberate outcome, never a broken output ---


def test_answered_rag_answer_is_valid():
    answer = RAGAnswer(
        question="q", abstained=False, answer="Use NU-Guest.", cited_doc_ids=["it-wifi"]
    )
    assert answer.status == "ok"


def test_abstention_requires_a_reason():
    with pytest.raises(ValidationError, match="abstain_reason"):
        RAGAnswer(question="q", abstained=True)


def test_reason_without_abstention_is_rejected():
    with pytest.raises(ValidationError, match="abstain_reason"):
        RAGAnswer(question="q", abstained=False, answer="x", abstain_reason="model_declined")


def test_abstention_cannot_carry_an_answer():
    with pytest.raises(ValidationError, match="answer"):
        RAGAnswer(question="q", abstained=True, abstain_reason="model_declined", answer="maybe")


def test_parse_error_is_never_an_abstention():
    with pytest.raises(ValidationError, match="status"):
        RAGAnswer(
            question="q", abstained=True, abstain_reason="model_declined", status="parse_error"
        )


def test_parse_error_may_have_no_answer():
    answer = RAGAnswer(question="q", abstained=False, status="parse_error", raw_output="{not json")
    assert answer.answer is None


def test_ok_non_abstained_answer_must_have_text():
    with pytest.raises(ValidationError, match="answer"):
        RAGAnswer(question="q", abstained=False, answer="   ")


# --- Chunks ---


def test_chunk_id_must_be_doc_id_hash_position():
    with pytest.raises(ValidationError, match="chunk_id"):
        Chunk(chunk_id="it-wifi-0", doc_id="it-wifi", title="t", text="x", position=0)


def test_retrieved_rank_is_one_based():
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk=make_chunk(), score=1.0, rank=0)


# --- TestCase labeling rules ---


def test_answerable_case_is_valid():
    assert make_case().answerable


def test_case_type_must_agree_with_answerable():
    with pytest.raises(ValidationError, match="case_type"):
        make_case(case_type="out_of_domain")


@pytest.mark.parametrize(
    "missing", [{"gold_doc_ids": []}, {"must_include": []}, {"reference_answer": None}]
)
def test_answerable_case_needs_gold_docs_facts_and_reference(missing):
    with pytest.raises(ValidationError, match="answerable"):
        make_case(**missing)


def test_unanswerable_case_cannot_have_gold_docs_or_facts():
    with pytest.raises(ValidationError, match="unanswerable"):
        make_case(answerable=False, case_type="missing_info", reference_answer=None)


def test_valid_unanswerable_case():
    case = make_case(
        answerable=False,
        case_type="personal",
        gold_doc_ids=[],
        must_include=[],
        reference_answer=None,
    )
    assert not case.answerable


def test_unknown_fields_are_rejected_to_catch_typos():
    with pytest.raises(ValidationError):
        make_case(gold_docs=["registrar-transcripts"])


def test_pytest_does_not_collect_testcase_as_a_test_class():
    assert TestCase.__test__ is False


def test_answerable_types():
    assert ANSWERABLE_TYPES == {"single_doc", "multi_doc", "paraphrase"}


# --- Run records ---


def test_case_result_round_trips_through_json():
    answer = RAGAnswer(
        question="What network should guests use?",
        abstained=False,
        answer="Guests use NU-Guest.",
        cited_doc_ids=["it-wifi"],
        retrieved=[RetrievedChunk(chunk=make_chunk(), score=1.5, rank=1)],
        usage=Usage(prompt_tokens=120, completion_tokens=12, latency_s=0.8, llm_calls=1),
    )
    row = CaseResult(
        run_id="r1",
        case_id="s02",
        answer=answer,
        scores={"hit@1": 1.0, "fact_recall": None},
        passed=True,
    )
    assert CaseResult.model_validate_json(row.model_dump_json()) == row


def test_failure_tags_are_a_closed_set():
    answer = RAGAnswer(question="q", abstained=True, abstain_reason="model_declined")
    with pytest.raises(ValidationError):
        CaseResult(run_id="r", case_id="c", answer=answer, passed=False, failure_tags=["OOPS"])


def test_error_row_requires_a_known_error_class():
    with pytest.raises(ValidationError):
        ErrorRow(run_id="r", case_id="c", error_class="oops", message="boom")


def test_run_meta_and_summary_round_trip_through_json():
    meta = RunMeta(
        run_id="20261008-120000-baseline-1a2b3c4d",
        created_at=datetime(2026, 10, 8, 12, 0, tzinfo=UTC),
        config={"name": "baseline", "retrieval": {"k": 5}},
        config_hash="1a2b3c4d",
        git_sha="9f48587",
        git_dirty=False,
        dataset=DatasetRef(path="data/testset", split="dev", sha256="ab", n_cases=100),
        corpus=CorpusRef(path="data/corpus/docs", sha256="cd", n_docs=60),
        env=EnvFingerprint(machine="omar-mbp", platform="macOS-15-arm64", python="3.12.13"),
    )
    summary = RunSummary(
        run_id=meta.run_id,
        metrics={"e2e_pass": 0.62, "latency_p95_s": None},
        counts=RunCounts(cases=100, ok=97, errors=1, truncated=1, parse_errors=1),
        failure_counts={"RETRIEVAL_MISS": 9},
    )
    assert RunMeta.model_validate_json(meta.model_dump_json()) == meta
    assert RunSummary.model_validate_json(summary.model_dump_json()) == summary


def test_document_round_trips_through_json():
    doc = Document(
        doc_id="it-wifi",
        title="Connecting to campus Wi-Fi",
        url="https://example.edu/wifi",
        section="IT",
        snapshot_date=date(2026, 10, 8),
        text="Guests use NU-Guest.",
    )
    assert Document.model_validate_json(doc.model_dump_json()) == doc


def test_metric_directions_are_higher_or_lower():
    assert METRIC_DIRECTION["e2e_pass"] == "higher"
    assert METRIC_DIRECTION["missed_abstain_rate"] == "lower"
    assert set(METRIC_DIRECTION.values()) == {"higher", "lower"}
