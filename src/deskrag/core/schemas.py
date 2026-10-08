"""Data contracts shared by every workstream.

These models are the agreement that lets five people build in parallel: the retriever produces
`RetrievedChunk`s, the chatbot returns a `RAGAnswer`, the test set is a list of `TestCase`s, and the
eval harness writes `CaseResult`/`ErrorRow`/`RunMeta`/`RunSummary` files. Change them only through a
`contract/*` PR (see CONTRIBUTING.md).
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    """Base for all contracts. Unknown fields are rejected so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


# --- Corpus -----------------------------------------------------------------------------------


class Document(Contract):
    doc_id: str  # stable slug, also the file name: data/corpus/docs/<doc_id>.md
    title: str
    url: str  # page the snapshot was taken from
    section: str  # e.g. "IT", "Registrar"
    snapshot_date: date
    text: str  # cleaned markdown body, without front-matter


class Chunk(Contract):
    chunk_id: str  # always f"{doc_id}#{position}"
    doc_id: str
    title: str  # parent document title, shown to the LLM
    text: str
    position: int = Field(ge=0)  # 0-based index of the chunk within its document

    @model_validator(mode="after")
    def _chunk_id_matches_doc_and_position(self) -> "Chunk":
        expected = f"{self.doc_id}#{self.position}"
        if self.chunk_id != expected:
            raise ValueError(f"chunk_id must be {expected!r}, got {self.chunk_id!r}")
        return self


class RetrievedChunk(Contract):
    chunk: Chunk
    score: float  # higher = more relevant; the scale depends on the retriever
    rank: int = Field(ge=1)  # 1-based position in the ranked list


# --- LLM --------------------------------------------------------------------------------------


class LLMResponse(Contract):
    text: str
    model: str  # model name as reported by the server, not the one we asked for
    done_reason: str | None = None  # Ollama: "stop", "length", ...
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None


# --- Chatbot output ---------------------------------------------------------------------------

AbstainReason = Literal["low_retrieval_score", "model_declined"]
AnswerStatus = Literal["ok", "truncated", "parse_error"]


class Usage(Contract):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    llm_calls: int = 0


class RAGAnswer(Contract):
    """What the chatbot returns for one question.

    "No answer" is not "abstained": an abstention is a deliberate, well-formed outcome
    (`status == "ok"`), while broken model output is reported through `status`. Plumbing failures
    (Ollama down, timeouts) are raised as `InfraError`, never returned.
    """

    question: str
    abstained: bool
    abstain_reason: AbstainReason | None = None
    answer: str | None = None
    cited_doc_ids: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    confidence: float | None = None  # e.g. the top retrieval score
    status: AnswerStatus = "ok"
    raw_output: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None

    @model_validator(mode="after")
    def _abstention_is_consistent(self) -> "RAGAnswer":
        if self.abstained != (self.abstain_reason is not None):
            raise ValueError("abstain_reason must be set exactly when abstained is true")
        if self.abstained and self.answer is not None:
            raise ValueError("an abstention cannot carry an answer")
        if self.abstained and self.status != "ok":
            raise ValueError("an abstention must have status 'ok'; broken output is not abstaining")
        if self.status == "ok" and not self.abstained and not (self.answer and self.answer.strip()):
            raise ValueError("an ok, non-abstained answer must have answer text")
        return self


# --- Test set ---------------------------------------------------------------------------------

CaseType = Literal[
    # answerable
    "single_doc",
    "multi_doc",
    "paraphrase",
    # unanswerable
    "out_of_domain",
    "missing_info",
    "false_premise",
    "personal",
]
ANSWERABLE_TYPES = frozenset({"single_doc", "multi_doc", "paraphrase"})


class TestCase(Contract):
    """One labeled question. Every `must_include` entry may list alternatives separated by `|`."""

    __test__ = False  # stop pytest from trying to collect this class

    case_id: str
    question: str
    answerable: bool
    case_type: CaseType
    gold_doc_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    must_include: list[str] = Field(default_factory=list)
    split: Literal["dev", "test"]
    author: str
    notes: str | None = None

    @model_validator(mode="after")
    def _labels_are_consistent(self) -> "TestCase":
        if self.answerable != (self.case_type in ANSWERABLE_TYPES):
            raise ValueError(
                f"case_type {self.case_type!r} does not match answerable={self.answerable}"
            )
        if self.answerable and not (
            self.gold_doc_ids and self.must_include and self.reference_answer
        ):
            raise ValueError(
                "answerable cases need gold_doc_ids, must_include, and reference_answer"
            )
        if not self.answerable and (self.gold_doc_ids or self.must_include):
            raise ValueError("unanswerable cases must not have gold_doc_ids or must_include")
        return self


# --- Run records (written by the eval runner, read by compare/report) -------------------------

FailureTag = Literal[
    "RETRIEVAL_MISS",  # answerable, no gold doc in the top-k
    "FALSE_ABSTAIN_GATE",  # answerable, the retrieval-score gate abstained
    "FALSE_ABSTAIN_MODEL",  # answerable, the LLM declined
    "MISSED_ABSTAIN",  # unanswerable, but the bot answered (hallucination)
    "WRONG_ANSWER",  # answerable and answered, but required facts are missing
    "BAD_CITATION",  # cited a doc that was not retrieved
    "FORMAT_ERROR",  # truncated or unparseable model output
]
ErrorClass = Literal["timeout", "infra", "exception"]


class CaseResult(Contract):
    """One row of results.jsonl."""

    run_id: str
    case_id: str
    rep: int = 0
    answer: RAGAnswer
    scores: dict[str, float | None] = Field(default_factory=dict)  # None = not applicable
    passed: bool
    failure_tags: list[FailureTag] = Field(default_factory=list)


class ErrorRow(Contract):
    """One row of errors.jsonl: an attempt that never produced a scorable answer."""

    run_id: str
    case_id: str
    rep: int = 0
    error_class: ErrorClass
    message: str
    attempts: int = 1


class EnvFingerprint(Contract):
    machine: str
    platform: str
    python: str
    ollama_version: str | None = None
    model_digests: dict[str, str] = Field(default_factory=dict)  # model tag -> digest


class DatasetRef(Contract):
    path: str
    split: str
    sha256: str
    n_cases: int


class CorpusRef(Contract):
    path: str
    sha256: str
    n_docs: int


class RunMeta(Contract):
    """meta.json. `config` is stored as plain data so old runs stay readable as configs evolve."""

    run_id: str
    created_at: datetime
    config: dict[str, Any]
    config_hash: str
    git_sha: str | None
    git_dirty: bool
    dataset: DatasetRef
    corpus: CorpusRef | None = None
    env: EnvFingerprint
    reps: int = 1


class RunCounts(Contract):
    cases: int
    ok: int
    errors: int
    truncated: int
    parse_errors: int


class RunSummary(Contract):
    """summary.json: aggregate metrics for one run."""

    run_id: str
    metrics: dict[str, float | None]
    by_case_type: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    counts: RunCounts
    failure_counts: dict[str, int] = Field(default_factory=dict)


# Which direction is "better" for each metric name. The eval runner (WS4) writes these names and
# compare (WS5) uses the direction to decide what a regression is. Add new metrics here.
METRIC_DIRECTION: dict[str, Literal["higher", "lower"]] = {
    "e2e_pass": "higher",
    "hit@1": "higher",
    "hit@3": "higher",
    "hit@5": "higher",
    "recall@5": "higher",
    "mrr": "higher",
    "abstain_correct": "higher",
    "abstain_precision": "higher",
    "abstain_recall": "higher",
    "abstain_f1": "higher",
    "fact_recall": "higher",
    "answer_correct": "higher",
    "citation_valid": "higher",
    "citation_hit": "higher",
    "false_abstain_rate": "lower",
    "missed_abstain_rate": "lower",
    "format_error_rate": "lower",
    "latency_p50_s": "lower",
    "latency_p95_s": "lower",
    "prompt_tokens_mean": "lower",
    "completion_tokens_mean": "lower",
}
