# Workplan

Due ≈ **Thu Oct 29** (shift the dates if that changes). Design: [design.md](design.md). Git rules: [CONTRIBUTING.md](../CONTRIBUTING.md).

## Owners

| WS | Branch | Owner | Notes |
|---|---|---|---|
| WS1 Data | `feat/data` | | no GPU needed; leads the case-writing drive |
| WS2 Retrieval | `feat/retrieval` | | light hardware |
| WS3 Generation | `feat/generation` | | **strongest laptop**: becomes the reference machine for official runs |
| WS4 Eval metrics | `feat/eval-metrics` | | no GPU needed; integration owner for M1 |
| WS5 Regression + CI | `feat/regression-ci` | | no GPU needed; integration owner for M2 |

## Milestones

| Date | Milestone | Done when |
|---|---|---|
| Thu Oct 8 – Fri Oct 9 | **M0** Foundation + kickoff | foundation merged; branches pushed; owners filled in above; everyone finished the kickoff checklist |
| Wed Oct 14 | **M1** First end-to-end run | ≥50 docs; 40 dev cases; BM25; prompt v1; runner + core metrics + report; first `baselines/dev` committed |
| Wed Oct 21 | **M2** Regression detection live | ~150 cases with the split frozen; dense + hybrid retrieval; gate tuned; failure tags; compare + McNemar + flips; CI retrieval gate blocking; `make regress` works |
| Mon Oct 26 | **M3** Code freeze + results | experiments done; 3 planted regressions caught; one held-out test-split run |
| Oct 27–29 | Write-up | report, slides, demo rehearsal |

## Kickoff checklist (everyone)

- [ ] Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama](https://ollama.com/download). Then run `uv sync` and `uv run pytest`; both should pass.
- [ ] `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`
- [ ] Read [design.md](design.md), especially the contracts, metrics, and the "no answer ≠ abstained" rule (~20 min).
- [ ] Claim a workstream in the Owners table.
- [ ] `git switch feat/<yours>`, make a first small commit, and open a **draft PR** so CI runs on every push.
- [ ] Write your first 5 test cases in `data/testset/<your-name>.jsonl` (see [data/testset/README.md](../data/testset/README.md)).
- [ ] Repo owner only: protect `main` (see CONTRIBUTING.md).

## Case-writing drive (everyone, weeks 1–2)

Everyone writes ~20 cases in their own file, covering every `case_type`; WS1 reviews them.

| Target | When |
|---|---|
| 40 dev cases | M1 |
| ~150 cases | M2 |

At M2, WS1 has 20% of the cases re-labeled blind by a second person, then runs the stratified split.

---

## WS1 `feat/data`: corpus and test set

**Day 1**
- [ ] List 60–100 public help-page URLs in `data/corpus/sources.csv` (`doc_id,url,section`). Cover IT, registrar, financial aid, housing, library, parking, health, dining, ….
- [ ] Write `docs/authoring-guide.md`: two examples per `case_type`, how to pick `must_include` facts, and how to write good unanswerable cases (especially `missing_info` near-misses).

**M1 (Oct 14)**
- [ ] `deskrag data snapshot` (`src/deskrag/data/snapshot.py`, run `uv sync --group scrape` first). It uses httpx + trafilatura to write `data/corpus/docs/<doc_id>.md` with front-matter, makes 1 request per second, and skips URLs that robots.txt disallows.
- [ ] Clean up by hand (navigation and footer leftovers); reach ≥50 docs. Write `corpus_sha256` to `data/corpus/MANIFEST`.
- [ ] `deskrag data validate` (`src/deskrag/data/validate.py`). It checks that gold docs exist, that each `must_include` fact appears in a gold doc, that there are no near-duplicate questions, and prints counts by split × case_type. It exits 1 on any error.
- [ ] 40 validated dev cases (yours plus teammates').

**M2 (Oct 21)**
- [ ] ~150 cases, ≥10 per `case_type`, ~60/40 answerable/unanswerable.
- [ ] Double-label 20% blind and report agreement (percent and Cohen's κ).
- [ ] Split script: stratified by `case_type`, seed 42, ~100 dev / ~50 test. Writes `split` into the files, then freeze.

**M3**: data section of the report (corpus stats, case-type table, agreement).

## WS2 `feat/retrieval`: chunking and retrievers

**Day 1**
- [ ] `src/deskrag/rag/chunking.py`: `chunk_document(doc, size_words, overlap_words) -> list[Chunk]`. It respects headings and paragraphs and produces ids `f"{doc_id}#{n}"`. Test on `data/samples/corpus`.
- [ ] `src/deskrag/rag/retrieval/bm25.py`: BM25 (rank-bm25, lowercase alphanumeric tokens). Implement `build_retriever` for `method: bm25`.

**M1**
- [ ] `deskrag index build` and `deskrag retrieve "…" -k 5` (prints rank, score, doc_id, and a snippet).
- [ ] BM25 on the real corpus; spot-check 10 questions with WS1.

**M2**
- [ ] `dense.py`: batched Ollama `/api/embed` with nomic's `search_query:` / `search_document:` prefixes, numpy cosine similarity, and a cache at `.cache/index/<corpus_sha>-<model>.npz`.
- [ ] `hybrid.py`: reciprocal-rank fusion of BM25 and dense.
- [ ] Using WS4's `eval run --retrieval-only`, compare bm25 / dense / hybrid, chunk sizes {200, 300, 500}, and k ∈ {3, 5, 8} on dev. Pick the baseline retriever.
- [ ] Give WS3 a sensible gate-score range for each method.

**M3**: retrieval experiment table.

## WS3 `feat/generation`: LLM client, prompts, abstention, `ask`

**Day 1**
- [ ] Model bake-off on the **slowest** teammate's laptop. Try 2–3 small instruct models (e.g. `llama3.2:3b`, `qwen3:4b`, `gemma3:4b`, or newer equivalents) on the 10 sample questions. Compare JSON validity, answer quality, and seconds per question.
- [ ] Pin the winner in `configs/baseline.yaml` and one alternative in `configs/experiments/model-alt.yaml`.
- [ ] `src/deskrag/rag/llm/ollama_client.py` implementing `LLMClient`:
  - Sends `format=<schema>`, `temperature`, `seed`, `num_ctx`, `num_predict`, and `think=False`.
  - Maps `done_reason == "length"` to `truncated`.
  - Asserts that the served model matches the requested one.
  - Retries twice with backoff, then raises `InfraError`.

**M1**
- [ ] `llm/cache.py`: on-disk response cache keyed by sha256(model digest, messages, schema, options).
- [ ] `pipeline.py` `build_rag_system`: retrieve → gate → prompt (`prompts/v1.md`) → chat → parse → `RAGAnswer`. Statuses and reasons follow the contract. Test with `FakeLLM` + `FakeRetriever`.
- [ ] `deskrag ask`.
- [ ] A helper that returns the Ollama version and model digests, which WS4 records in `meta.json`.

**M2**
- [ ] Tune the gate threshold with `deskrag eval sweep` on dev. Write prompt v2 if it helps, with `compare.md` in the PR.
- [ ] `closed_book.py`.
- [ ] Run the official baseline on the reference machine (`DESKRAG_MACHINE=reference`) and open the promotion PR.

**M3**: model-swap and abstention-layer experiments; the live `ask` demo for the presentation.

## WS4 `feat/eval-metrics`: runner, metrics, failure tracking

**Day 1**
- [ ] Metric modules in `src/deskrag/evaluation/metrics/` (`retrieval.py`, `abstention.py`, `answer.py`, `ops.py`), tested on `FakeRAGSystem` + `data/samples` with exact expected values. For example, s02's reciprocal rank is 0.5, and e2e is 4/10 overall (4/7 on dev). See the table in `core/fakes.py`.

**M1**
- [ ] `runner.py` and `deskrag eval run`:
  - Appends a row per case, resumes by (case_id, rep), and enforces a 120 s ceiling per case.
  - Sends `InfraError` to `errors.jsonl`.
  - Writes `meta.json` (config hash, git info, dataset and corpus fingerprints, env fingerprint) and `summary.json`.
- [ ] `failures.py`: the failure tags.
- [ ] `sanity.py`: `oracle`, `always_abstain`, and `never_abstain`. The oracle should score ≈100%.
- [ ] `make eval` works end to end on the real system. WS4 is the integration owner for M1.

**M2**
- [ ] by-`case_type` breakdowns, `--retrieval-only`, and `deskrag eval sweep`.

**M3**: metrics section of the report.

## WS5 `feat/regression-ci`: baselines, comparison, reports, CI

**Day 1**
- [ ] Fixtures: two fake runs (the fake config, plus a copy with a few flipped rows).
- [ ] `src/deskrag/evaluation/stats.py`: exact two-sided McNemar test in pure Python. For example, b=1 and c=9 should give p ≈ 0.021.

**M1**
- [ ] `compare.py` and `deskrag compare`, with gates in `configs/regression.yaml`. It produces verdicts and flipped cases, writes `compare.md`, and exits 1 on a gated regression.
- [ ] `report.py` and `deskrag report`.
- [ ] `deskrag baseline promote`, `runs list`, and `runs table`.

**M2**
- [ ] CI `retrieval-gate` job: a BM25 retrieval-only eval (`configs/ci-retrieval.yaml`) compared against `baselines/dev-retrieval/`. Warn-only until M2, blocking after.
- [ ] `make regress`. WS5 is the integration owner for M2.

**M3**
- [ ] Planted-regression branches, never merged:
  - `demo/regress-prompt`: abstain instruction removed. Should be caught by `make regress`.
  - `demo/regress-k1`: k=1. CI should go red.
  - `demo/regress-model`: a smaller model.
- [ ] Results tables for the report via `deskrag runs table`.

---

## Experiments for the report (M3)

| Experiment | Compare | Main metrics |
|---|---|---|
| Retriever | bm25 vs dense vs hybrid | hit@k, MRR, e2e_pass |
| Abstention layers | none vs prompt-only vs gate + prompt | abstain P/R/F1, false_abstain_rate |
| Model swap | default vs alternative model | e2e_pass, fact_recall, format errors, latency |
| Does retrieval matter? | closed-book vs RAG | e2e_pass, missed_abstain_rate |
| Final | best config on the **test** split, once | everything |

**Stretch, only if ahead:** LLM-as-judge calibrated against human labels, risk-coverage curves / AURC, a full LLM eval in CI, a dashboard, post-hoc answer verification.
