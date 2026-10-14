# Regression comparison

**Offline fixture demonstration. Not measured model performance.**
Baseline: `gate-baseline`
Candidate: `gate-baseline`

Deltas are candidate minus baseline. Tolerances are absolute metric units.

| Metric | Baseline | Candidate | Delta | Tolerance | b | c | p | Verdict | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| e2e_pass | 0.6 | 0.6 | +0 | 0.05 | 0 | 0 | 1 | OK | yes |
| hit@5 | 1 | 1 | +0 | 0.05 | 0 | 0 | 1 | OK | yes |
| abstain_correct | 0.6 | 0.6 | +0 | 0.05 | 0 | 0 | 1 | OK | yes |
| fact_recall | — | — | — | — | — | — | — | NOT APPLICABLE | no |
| format_error_rate | — | — | — | — | — | — | — | NOT APPLICABLE | no |
| latency_p95_s | — | — | — | — | — | — | — | NOT APPLICABLE | no |

## Flipped cases

| Metric | Case | Change | Candidate failure tags |
|---|---|---|---|
| — | — | no flips | — |

## Interpretation

No gated statistically significant regression detected.
A POSSIBLE verdict remains a warning, not evidence of equivalence. Continuous metrics have no McNemar p-value and are warn-only.
Tests are per metric without multiplicity correction. Small datasets have limited power.
