# Model bake-off (WS3, Day 1)

The bake-off chose the baseline model and the alternative for the model-swap experiment. To re-run it, use `scripts/bakeoff.py` with the commands at the end of this page.

**Decision:**
- **Baseline:** `qwen3:4b-instruct-2507-q4_K_M`.
- **Alternative** (`configs/experiments/model-alt.yaml`): `llama3.2:3b`.

## Setup

- **When and where:** 2026-10-13, on a MacBook Air (M4, 16 GB). Ollama 0.40.3 with default server settings.
- **Pipeline:** `configs/baseline.yaml` with the score gate off, so each model's own abstention is measured.
  - Chunks of 300 words with 50 overlapping; k = 5; prompt v1.
  - Temperature 0, seed 42, `num_ctx` 4096, at most 512 output tokens.
- **Candidates:**

| Tag | Digest | Size |
|---|---|---|
| `llama3.2:3b` | `a80c4f17acd5` | 2.0 GB |
| `qwen3:4b-instruct-2507-q4_K_M` | `0edcdef34593` | 2.5 GB |
| `gemma3:4b` | `68ceab7e468a` | 3.4 GB |

  `qwen3:4b` now points to Qwen's thinking-only variant, so we used the instruct tag.
- **Cases:**
  1. The 10 sample cases, retrieved by `FakeRetriever` over `data/samples/corpus`.
  2. The 21 real dev cases in `data/testset/faeghe.jsonl` (12 answerable, 9 unanswerable), retrieved by WS2's BM25 over the real corpus. BM25 was still a draft PR, so we merged it into a throwaway checkout for this run.

## Results

Column meanings:
- **e2e pass:** answerable cases answered with every `must_include` fact, plus unanswerable cases abstained.
- **cites retrieved docs:** answered cases that cite at least one doc, and only retrieved ones.
- **format errors:** truncated or unparseable replies.
- **s/q:** seconds per question.

This scoring is a rough stand-in for WS4's metrics.

### Sample questions

| model | e2e pass | answerable: correct | unanswerable: abstained | false abstain | cites retrieved docs | format errors | infra errors | s/q p50 | s/q max | prompt tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 9/10 | 6/6 | 3/4 | 0/6 | 5/7 | 0/10 | 0 | 1.5 | 2.6 | 787 |
| qwen3:4b-instruct-2507-q4_K_M | 10/10 | 6/6 | 4/4 | 0/6 | 6/6 | 0/10 | 0 | 2.5 | 4.0 | 795 |
| gemma3:4b | 9/10 | 5/6 | 4/4 | 0/6 | 6/6 | 0/10 | 0 | 3.2 | 3.9 | 837 |

### Real dev cases (as scored by the script)

| model | e2e pass | answerable: correct | unanswerable: abstained | false abstain | cites retrieved docs | format errors | infra errors | s/q p50 | s/q max | prompt tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 14/21 | 11/12 | 3/9 | 0/12 | 17/18 | 0/21 | 0 | 4.5 | 6.0 | 1612 |
| qwen3:4b-instruct-2507-q4_K_M | 16/21 | 8/12 | 8/9 | 0/12 | 12/13 | 0/21 | 0 | 6.7 | 9.7 | 1649 |
| gemma3:4b | 13/20 | 7/11 | 6/9 | 1/11 | 11/13 | 0/20 | 1 | 7.7 | 10.1 | 1684 |

### After reading every failed answer

The script matches `must_include` facts word for word. Reading the answers changes the picture:

- **One case is mislabeled.** ff-015 ("How much does a semester student parking permit cost?") is labeled `missing_info`, but `acct007` lists "Parking Permit (Per Semester) | $152.00". All three models answered "$152.00" and cited `acct007`.
- **Four of qwen3's "wrong" answers are right**, just worded differently from the required phrase (details under "For WS1" below).

| model | e2e pass, judged by meaning | truly unanswerable cases abstained (8) | real misses |
|---|---|---|---|
| qwen3:4b-instruct-2507-q4_K_M | **20/21** | **8/8** | ff-009 is borderline: it adds "student organizations" to who can reserve rooms |
| gemma3:4b | 17/20 (1 error) | 6/8 | false abstain on a multi-doc case; invented a drop date; one refusal written as an answer |
| llama3.2:3b | 15/21 | 3/8 | answered both false-premise questions; invented a drop date; two refusals written as answers; half-answered a multi-doc case |

## Observations

- **Abstention is where the models differ most.** qwen3 abstained on every truly unanswerable question. llama3.2 gave library hours for the nonexistent "Mars campus" and said where to file a fourth-year housing waiver that doesn't exist.
- **Refusals written as answers.** llama3.2 (ff-017, ff-021, sample s08) and gemma3 (ff-017) sometimes reply `abstain: false` with an answer like "the documents don't specify…". By the "no answer ≠ abstained" rule this counts as an answer, so it scores as `MISSED_ABSTAIN`. Prompt v2 could say this explicitly.
- **Citations.**
  - llama3.2 made up `doc-1` (s06).
  - qwen3 wrote `doc005` for `reg005` (ff-012).
  - gemma3 prefixed ids with `doc-id:`, copying the prompt's example `["doc-id"]`, and once wrote its citations into the answer text.
  - Prompt v2 could use a real-looking id in the example, such as `["it002"]`.
- **gemma3** hit Ollama's "token repeat limit" three times on ff-005, so that case became an `InfraError`. It was also the slowest.
- **No format errors** from any model: JSON-schema output works on all three.
- **Speed and context:** real prompts average about 1.6k tokens and peak at 2.2k, so `num_ctx` 4096 leaves room for the 512 output tokens. On this laptop, a 100-case dev run with qwen3 takes about 11 minutes.

## Why this choice

- **qwen3-4B-instruct** has by far the best abstention, which is what this project measures. Its answers are correct, and it produced no format errors. The cost is about 1.5× llama3.2's time per question.
- **llama3.2:3b** is the fastest and answers well, but it rarely abstains. That makes it a good contrast for the model-swap experiment.
- **gemma3:4b** is dropped: slowest, one aborted generation, and a false abstain.

## For WS1: test-set issues found

These are suggestions; the case authors decide.

- **ff-015 is answerable.** Relabel it `single_doc` with gold `acct007` and `must_include` `$152`.
- **Some `must_include` facts are full phrases that correct answers word differently.** Shorter facts, with `|` alternatives, would score fairly:

| Case | Current fact | What correct answers said | Suggested fact |
|---|---|---|---|
| ff-004 | `noon-4 p.m.` | "noon to 4 p.m." | `noon` |
| ff-008 | `$40 housing application fee` | "$40 non-refundable housing application fee" | `$40` |
| ff-009 | `Faculty & Staff` | "Faculty, staff, …" | `faculty` |
| ff-009 | `Academic Transcripts` | "Academic Transcript" | `Academic Transcript` |
| ff-010 | `not required to have a meal plan` | "do not have to get a meal plan" | `not required\|do not have to\|don't have to` |

## Caveats

- 31 questions is a small sample: one case moves a rate by 5–10 points.
- Speeds come from the team's fastest laptop. The slowest laptop should re-run the bake-off before M2.
- If WS2 changes the baseline retriever or chunk size, prompt sizes change, so recheck `num_ctx`.

## Re-run

```bash
# Sample questions (no WS2 code needed)
uv run python scripts/bakeoff.py --cases data/samples/cases.jsonl --split all --corpus data/samples/corpus --models qwen3:4b-instruct-2507-q4_K_M llama3.2:3b gemma3:4b
# Real dev cases (needs WS2's BM25 on your branch); --out keeps every answer for review
uv run python scripts/bakeoff.py --cases data/testset --models qwen3:4b-instruct-2507-q4_K_M llama3.2:3b gemma3:4b --out bakeoff.jsonl
```
