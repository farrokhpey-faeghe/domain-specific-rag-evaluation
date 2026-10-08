# domain-specific-rag-evaluation

A domain-specific RAG system evaluated using retrieval quality and abstention metrics.

**The system:** a university help-desk chatbot (`deskrag`). It answers questions from a frozen snapshot of the university's public help pages, and abstains when the pages don't contain the answer.

**The main project:** our own evaluation harness. It runs a labeled test set through the chatbot, scores retrieval, abstention, and answer quality, tags every failure, and flags regressions when a change to the prompt, code, or model makes things worse.

Everything runs locally and free. The LLM and the embeddings come from [Ollama](https://ollama.com).

| Read this | For |
|---|---|
| [docs/design.md](docs/design.md) | architecture, shared contracts, metrics, regression detection |
| [docs/workplan.md](docs/workplan.md) | owners, branches, milestones, per-workstream checklists |
| [CONTRIBUTING.md](CONTRIBUTING.md) | git rules and file ownership |

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama](https://ollama.com/download).
2. Install the project. uv downloads Python 3.12 if you don't have it.

   ```bash
   uv sync
   ```

3. Pull the models named in `configs/baseline.yaml`. The chat model is a placeholder until the Day-1 bake-off.

   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

4. Check that everything works:

   ```bash
   uv run pytest
   uv run deskrag --help
   ```

`make test`, `make lint`, and `make fmt` wrap the common commands. On Windows without make, run the `uv run ...` commands directly.

## Commands

| Command | What it does | Owner |
|---|---|---|
| `deskrag data snapshot`, `deskrag data validate` | fetch the help pages; check the test set against the corpus | WS1 |
| `deskrag index build`, `deskrag retrieve "…"` | build retrieval indexes; show the top chunks for a query | WS2 |
| `deskrag ask "…"` | ask the chatbot one question | WS3 |
| `deskrag eval run`, `deskrag eval sweep` | run and score the test set; tune the abstention gate | WS4 |
| `deskrag compare A B`, `deskrag report`, `deskrag baseline promote`, `deskrag runs list` / `runs table` | regression check, reports, baselines | WS5 |

A command that isn't built yet exits with code 3 and names its owner.

## Repo map

```
configs/                 system configs: baseline.yaml, fake.yaml, experiments/*.yaml
data/samples/            5 fictional docs + 10 labeled questions for tests and Day-1 work
data/corpus/             the real help-page snapshot (WS1)
data/testset/            the real test set, one <author>.jsonl per person
src/deskrag/core/        shared contracts: schemas, config, loaders, run files, fakes (frozen)
src/deskrag/rag/         the chatbot: chunking + retrieval (WS2), LLM + prompts + pipeline (WS3)
src/deskrag/evaluation/  runner + metrics (WS4), compare + reports (WS5)
src/deskrag/commands/    one CLI module per workstream
baselines/               committed reference runs (WS5)
runs/                    your local eval runs (git-ignored)
```
