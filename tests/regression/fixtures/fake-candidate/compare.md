# Regression comparison

**Offline fixture demonstration. Not measured model performance.**
Baseline: `fake-baseline`
Candidate: `fake-candidate`

Deltas are candidate minus baseline. Tolerances are absolute metric units.

| Metric | Baseline | Candidate | Delta | Tolerance | b | c | p | Verdict | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| e2e_pass | 0.571429 | 0.142857 | -0.428571 | 0.05 | 3 | 0 | 0.25 | POSSIBLE | yes |
| hit@5 | 1 | 1 | +0 | 0.05 | 0 | 0 | 1 | OK | yes |
| abstain_correct | 0.571429 | 0.142857 | -0.428571 | 0.05 | 3 | 0 | 0.25 | POSSIBLE | yes |
| fact_recall | — | — | — | — | — | — | — | NOT APPLICABLE | no |
| format_error_rate | — | — | — | — | — | — | — | NOT APPLICABLE | no |
| latency_p95_s | — | — | — | — | — | — | — | NOT APPLICABLE | no |

## Flipped cases

| Metric | Case | Change | Candidate failure tags |
|---|---|---|---|
| e2e_pass | s01 | pass→fail | WRONG_ANSWER |
| e2e_pass | s02 | pass→fail | WRONG_ANSWER |
| e2e_pass | s07 | pass→fail | WRONG_ANSWER |
| abstain_correct | s01 | pass→fail | WRONG_ANSWER |
| abstain_correct | s02 | pass→fail | WRONG_ANSWER |
| abstain_correct | s07 | pass→fail | WRONG_ANSWER |

## Interpretation

No gated statistically significant regression detected.
A POSSIBLE verdict remains a warning, not evidence of equivalence. Continuous metrics have no McNemar p-value and are warn-only.
Tests are per metric without multiplicity correction. Small datasets have limited power.
