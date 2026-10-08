# Contributing

Five people, five branches, three weeks. These rules keep `main` green and merges painless.

## Branches

| Branch | Workstream |
|---|---|
| `feat/data` | WS1: corpus snapshot and test set |
| `feat/retrieval` | WS2: chunking and retrievers |
| `feat/generation` | WS3: LLM client, prompts, abstention, `deskrag ask` |
| `feat/eval-metrics` | WS4: runner, metrics, failure tags, sanity systems |
| `feat/regression-ci` | WS5: compare, reports, baselines, CI |

Owners are listed in [docs/workplan.md](docs/workplan.md).

## Daily loop

```bash
git switch feat/<yours>
git pull
git merge origin/main       # pick up teammates' merged work; do this every day
# ...work, commit in small steps...
uv run ruff format . && uv run ruff check . && uv run pytest
git push
```

## Rules

1. **Stay in your lane.** Only edit the paths your workstream owns (see the table below). If you need a change somewhere else, ask its owner or open an issue.
2. **Merge early and often.** Open your first PR within 2–3 days, then merge at least every 2–3 days. PRs are squash-merged. Keep working on the same branch after a merge.
3. **Contracts are frozen.** `src/deskrag/core/`, `systems.py`, `cli.py`, `pyproject.toml`, and `uv.lock` change only through a small `contract/<what>` PR. It needs 2 reviewers, including the owners of the affected workstreams. Merge it the same day and announce it in the group chat.
4. **A new dependency gets its own tiny PR** (`uv add <package>`), so `uv.lock` never conflicts.
5. **Show the eval impact.** A PR that touches `src/deskrag/rag/`, `configs/`, prompts, or `data/` includes the `compare.md` from `deskrag compare` (once WS5 ships it).
6. **Test cases go in your own file**, `data/testset/<your-name>.jsonl`, with case ids prefixed by your initials. See [data/testset/README.md](data/testset/README.md).
7. **Never tune on the test split.** Prompts, thresholds, and `k` are tuned on `dev`. The `test` split is run once, at M3.

## File ownership

| Paths | Owner |
|---|---|
| `src/deskrag/core/**`, `src/deskrag/systems.py`, `src/deskrag/cli.py`, `pyproject.toml`, `uv.lock` | shared (contract PR) |
| `data/corpus/**`, `src/deskrag/data/**`, `src/deskrag/commands/data.py`, `docs/authoring-guide.md` | WS1 |
| `data/testset/<name>.jsonl` | each author (WS1 reviews) |
| `src/deskrag/rag/chunking.py`, `src/deskrag/rag/retrieval/**`, `src/deskrag/commands/retrieval.py` | WS2 |
| `src/deskrag/rag/{llm/**, prompts/**, pipeline.py, abstention.py, closed_book.py}`, `src/deskrag/commands/ask.py`, `configs/baseline.yaml` | WS3 |
| `src/deskrag/evaluation/{runner.py, metrics/**, failures.py, sanity.py}`, `src/deskrag/commands/evaluate.py` | WS4 |
| `src/deskrag/evaluation/{compare.py, stats.py, report.py, baseline.py}`, `src/deskrag/commands/regress.py`, `configs/{regression,ci-retrieval}.yaml`, `baselines/**`, `.github/**`, `Makefile` | WS5 |
| `configs/experiments/<name>.yaml` | anyone (one file per experiment) |
| `tests/<area>/` | same owner as the code it tests |

Baseline promotions: WS3 runs them on the reference machine and opens the PR; WS5 reviews it.

## For the repo owner: protect `main`

In Settings → Branches, add a rule for `main`:
- require a pull request with 1 approval
- require the `CI / test` check to pass
- allow squash merging
