# Regression comparison

**Offline fixture demonstration. Not measured model performance.**
Baseline: `gate-baseline`
Candidate: `gate-candidate`

Deltas are candidate minus baseline. Tolerances are absolute metric units.

| Metric | Baseline | Candidate | Delta | Tolerance | b | c | p | Verdict | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| e2e_pass | 0.6 | 0.2 | -0.4 | 0.05 | 8 | 0 | 0.0078125 | REGRESSION | yes |
| hit@5 | 1 | 1 | +0 | 0.05 | 0 | 0 | 1 | OK | yes |
| abstain_correct | 0.6 | 0.2 | -0.4 | 0.05 | 8 | 0 | 0.0078125 | REGRESSION | yes |
| fact_recall | — | — | — | — | — | — | — | NOT APPLICABLE | no |
| format_error_rate | — | — | — | — | — | — | — | NOT APPLICABLE | no |
| latency_p95_s | — | — | — | — | — | — | — | NOT APPLICABLE | no |

## Flipped cases

| Metric | Case | Change | Candidate failure tags |
|---|---|---|---|
| e2e_pass | demo-04 | pass→fail | WRONG_ANSWER |
| e2e_pass | demo-05 | pass→fail | WRONG_ANSWER |
| e2e_pass | demo-06 | pass→fail | WRONG_ANSWER |
| e2e_pass | demo-07 | pass→fail | WRONG_ANSWER |
| e2e_pass | demo-08 | pass→fail | WRONG_ANSWER |
| e2e_pass | demo-09 | pass→fail | WRONG_ANSWER |
| e2e_pass | demo-10 | pass→fail | WRONG_ANSWER |
| e2e_pass | demo-11 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-04 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-05 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-06 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-07 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-08 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-09 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-10 | pass→fail | WRONG_ANSWER |
| abstain_correct | demo-11 | pass→fail | WRONG_ANSWER |

## Interpretation

BLOCKED: gated regression detected.
A POSSIBLE verdict remains a warning, not evidence of equivalence. Continuous metrics have no McNemar p-value and are warn-only.
Tests are per metric without multiplicity correction. Small datasets have limited power.
