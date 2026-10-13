# Offline fixtures only

These are schema-valid recorded runs for testing WS5 without WS2/WS4 or Ollama.
They are NOT real model evaluation results, independent research observations,
or official baselines. Do not use their scores in a model performance table.

- `fake-baseline`: seven DEV sample cases using FakeRAGSystem canned answers.
- `fake-candidate`: the same records with three pass scores deliberately flipped.
  Its drop is POSSIBLE because three discordant pairs give p=0.25.
- `gate-baseline` / `gate-candidate`: twenty synthetic identities built from a
  sample template, with eight deliberate pass→fail changes. They validate code
  branches: 0.60 → 0.20, b=8, c=0, p=0.0078125, REGRESSION, exit 1.
  The duplicated question makes these unsuitable for statistical research.

`abstain_correct` scores in these test records are canned to match the pass
scores, not computed abstention measurements. WS4 owns real metric calculation.

Regenerate: `uv run python tests/regression/generate_fixtures.py`.
No held-out test split is evaluated by this demo.
