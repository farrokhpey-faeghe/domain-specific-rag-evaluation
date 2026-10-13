# WS3: generation

What `feat/generation` provides, and the rules other workstreams can rely on. Shared contracts are in [design.md](design.md).

## Modules

| Module | Provides |
|---|---|
| `rag/llm/ollama_client.py` | `OllamaClient`, the `LLMClient` for Ollama. It also reports the server version and model digests. |
| `rag/llm/cache.py` | `CachedLLM`, an opt-in on-disk response cache that wraps any `LLMClient` |
| `rag/llm/env.py` | `env_fingerprint(cfg)`: what WS4 writes to `meta.json` as `env` |
| `rag/llm/__init__.py` | `build_llm(cfg)` and `llm_options(cfg.llm)`, shared by the RAG and closed-book builders |
| `rag/pipeline.py` | `RAGPipeline` (retrieve → gate → prompt → chat → parse) and `build_rag_system(cfg)` |
| `rag/closed_book.py` | `build_closed_book_system(cfg)`: the same pipeline with no retrieval and no gate |
| `rag/prompts/*.md` | Versioned system prompts. Editing one changes `config_hash`. |
| `commands/ask.py` | `deskrag ask` |
| `scripts/bakeoff.py` | The Day-1 model bake-off, which anyone can re-run on their own laptop |

`RAGPipeline` takes any `Retriever` and any `LLMClient`, so tests run it with `FakeRetriever` and `FakeLLM`, and nothing in WS3 waits on WS2's retrievers.

## The Ollama client

- Every call sends `format=<JSON schema>`, `think=False`, `stream=False`, and the options from the config: `temperature`, `seed`, `num_ctx`, and `num_predict` (`llm.max_tokens`). Options travel with each call, so the cache key covers them.
- Connection errors, timeouts, and server errors are retried twice (after 1 s, then 2 s), then raised as `InfraError`.
- If the reply comes from a different model than the one requested, the client raises `InfraError`.
- `build_llm` checks up front that Ollama is reachable and the model is pulled. If not, it raises `InfraError` with the `ollama pull <model>` command to run, so a run fails at the start instead of on every case.
- `latency_s` is wall-clock time for the whole call.

## From model reply to `RAGAnswer`

The model must return `{"abstain": bool, "answer": str, "citations": [str]}`. The rules apply in order:

| Situation | `RAGAnswer` |
|---|---|
| Gate on and top retrieval score < `abstention.min_score` (or nothing retrieved) | `abstained`, `low_retrieval_score`, **no LLM call**, `llm_calls == 0` |
| `done_reason == "length"` | `status="truncated"` |
| Not JSON, or a key missing or of the wrong type | `status="parse_error"` |
| `abstain: true` | `abstained`, `model_declined`; the answer and citations are dropped (`raw_output` keeps them) |
| `abstain: false` with an empty answer | `status="parse_error"`. "No answer" is never counted as an abstention. |
| `abstain: false` with an answer | `status="ok"`, `answer`, and `cited_doc_ids` |

- **Citations are not filtered against the retrieved docs**, so `BAD_CITATION` stays measurable. The only cleanup: `[it002]` and `it002#3` both become `it002`, and duplicates are removed.
- `confidence` is the top retrieval score. `usage` holds the token counts and the wall-clock time of the whole `answer()` call. `model` is the model name the server reported, and `raw_output` is the reply text.
- **Prompt:** the system message is `prompts/<prompt_version>.md`. The user message lists the retrieved chunks as `[doc_id] title\ntext` blocks, then `Question: ...`. Closed-book sends only the question.

## Response cache (off by default)

Turn it on with `DESKRAG_LLM_CACHE=1`. Responses are stored in `.cache/llm/` (gitignored). The key is the sha256 of the model digest, the Ollama version, the think flag, the messages, the schema, and the options. A hit replays the stored response, including its original latency.

It is off by default for two reasons. `deskrag eval run --reps 3` measures run-to-run variation, and with the cache on every rep would replay the same answer. Latencies in a cached run are also copies, not measurements. Use the cache to re-run an unchanged system cheaply; while developing metrics or for the demo.

## Environment fingerprint (for WS4)

`env_fingerprint(cfg) -> EnvFingerprint` is `core.env.basic_fingerprint()` plus `ollama_version` and `model_digests` for the models the config uses: `llm.model`, plus `retrieval.embed_model` for `dense` and `hybrid`. For `fake` and the sanity systems it doesn't contact Ollama at all.

Official runs happen on one reference machine (`DESKRAG_MACHINE=reference`) with one Ollama server setup. Server settings such as flash attention or KV-cache quantization can change outputs, so keep them the same between a baseline and the runs compared against it.

## `deskrag ask`

```bash
uv run deskrag ask "How do I reset my password?"
uv run deskrag ask "How much does an electronic transcript cost?" --config configs/fake.yaml
uv run deskrag ask "..." --json
```

It prints either the answer and its citations, or the abstention and its reason. It then lists the retrieved chunks (rank, score, chunk id, title) and a status line with the model, latency, and tokens. `--json` prints the whole `RAGAnswer`. When Ollama can't be reached it prints the problem and exits 1. The baseline config needs WS2's `build_retriever`.

## Closed-book

`configs/experiments/closed-book.yaml` sets `system: closed_book` and `prompt_version: closed-book-v1`. That prompt names the university and lets the model answer from memory, abstaining when unsure. Prompt v1 says "use only the context", so without context it would abstain on everything and the experiment would show nothing. No retrieval runs and the gate is off. If closed-book scores close to RAG, the questions are answerable from memory and the test set needs harder cases.

## Model bake-off

`scripts/bakeoff.py` runs each candidate model on:
1. the 10 sample cases (`FakeRetriever` over `data/samples/corpus`), which check format and abstention behavior;
2. the real dev cases (the config's retriever over the real corpus), which give realistic quality and speed.

The gate is off, so the model's own abstention is measured. For each model it reports format errors, correct abstentions, answer correctness (`must_include`), a rough e2e pass rate, seconds per question (p50 and max), and prompt tokens. This scoring is a quick approximation of WS4's metrics, used only to pick a model.

Results are in [model-bakeoff.md](model-bakeoff.md). The winner is pinned by exact tag in `configs/baseline.yaml`, and the runner-up in `configs/experiments/model-alt.yaml`. The first bake-off ran on the reference machine (M4 MacBook Air, 16 GB), so its speeds are best-case. The slowest laptop on the team should re-run the script before M2.

## Not yet (needs WS4's runner or the frozen test set)

- Tune `abstention.min_score` with `deskrag eval sweep` on dev.
- Prompt v2 if it helps, with `compare.md` in the PR.
- Official baseline on the reference machine and the promotion PR.
