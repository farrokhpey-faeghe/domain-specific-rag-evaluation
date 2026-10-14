# Evaluation report

**Offline fixture demonstration. These scores are not real model performance.**
Run: `gate-baseline`
Split: dev; cases: 20; reps: 1
Dataset SHA256: `synthetic-fixture-v1`
Config SHA256: `fa5058c470f76eab4948595e6001468a5699d2ace66fbc51b0af4c1984f18a97`
Machine: fixture

## Counts

| Count | Value |
|---|---:|
| cases | 20 |
| ok | 20 |
| errors | 0 |
| truncated | 0 |
| parse_errors | 0 |

## Metrics

| Metric | Value |
|---|---:|
| abstain_correct | 0.6 |
| e2e_pass | 0.6 |
| hit@5 | 1 |

## Failure counts

| Tag | Count |
|---|---:|
| WRONG_ANSWER | 8 |

## Failed cases

| Case | Rep | Status | Failure tags |
|---|---:|---|---|
| demo-12 | 0 | ok | WRONG_ANSWER |
| demo-13 | 0 | ok | WRONG_ANSWER |
| demo-14 | 0 | ok | WRONG_ANSWER |
| demo-15 | 0 | ok | WRONG_ANSWER |
| demo-16 | 0 | ok | WRONG_ANSWER |
| demo-17 | 0 | ok | WRONG_ANSWER |
| demo-18 | 0 | ok | WRONG_ANSWER |
| demo-19 | 0 | ok | WRONG_ANSWER |

Infrastructure errors are unscored and prevent comparison or baseline promotion.
A report is descriptive. A separate comparison determines regression verdicts.
