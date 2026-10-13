"""Validate recorded runs and detect paired regressions. Never runs a model."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from deskrag.core.runio import RunPaths, read_json, read_jsonl
from deskrag.core.schemas import METRIC_DIRECTION, CaseResult, ErrorRow, RunMeta, RunSummary
from deskrag.evaluation.stats import mcnemar_exact

BINARY_METRICS = {
    "e2e_pass",
    "hit@1",
    "hit@3",
    "hit@5",
    "abstain_correct",
    "answer_correct",
    "citation_valid",
    "citation_hit",
    "false_abstain_rate",
    "missed_abstain_rate",
    "format_error_rate",
}


@dataclass
class RecordedRun:
    path: Path
    meta: RunMeta
    summary: RunSummary
    rows: dict[tuple[str, int], CaseResult]
    errors: list[ErrorRow]


def load_run(path: Path) -> RecordedRun:
    paths = RunPaths(path)
    for file in (paths.meta, paths.summary, paths.results):
        if not file.is_file():
            raise ValueError(f"Missing run file: {file}")
    meta = read_json(paths.meta, RunMeta)
    summary = read_json(paths.summary, RunSummary)
    results = read_jsonl(paths.results, CaseResult)
    errors = read_jsonl(paths.errors, ErrorRow)
    if summary.run_id != meta.run_id or any(r.run_id != meta.run_id for r in results + errors):
        raise ValueError("Run IDs disagree")
    rows = {(r.case_id, r.rep): r for r in results}
    if len(rows) != len(results):
        raise ValueError("Duplicate (case_id, rep) result")
    if meta.reps < 1 or meta.dataset.n_cases < 1:
        raise ValueError("Run must contain cases and positive reps")
    if any(r.rep < 0 or r.rep >= meta.reps for r in results):
        raise ValueError("Result repetition outside metadata range")
    for value in summary.metrics.values():
        if value is not None and not math.isfinite(value):
            raise ValueError("Nonfinite aggregate metric")
    for row in results:
        if any(v is not None and not math.isfinite(v) for v in row.scores.values()):
            raise ValueError("Nonfinite per-case metric")
    return RecordedRun(path, meta, summary, rows, errors)


def require_complete(run: RecordedRun) -> None:
    if run.errors or run.summary.counts.errors:
        raise ValueError("Run has infrastructure errors; retry them before comparison/promotion")
    if len(run.rows) != run.meta.dataset.n_cases * run.meta.reps:
        raise ValueError("Incomplete run: result count does not match dataset and repetitions")
    if len({key[0] for key in run.rows}) != run.meta.dataset.n_cases:
        raise ValueError("Case count does not match metadata")
    if run.summary.counts.cases != run.meta.dataset.n_cases:
        raise ValueError("Summary case count disagrees with metadata")


def load_gates(path: Path) -> dict:
    try:
        gates = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid gate YAML: {exc}") from exc
    if not isinstance(gates, dict) or set(gates) - {"alpha", "metrics"}:
        raise ValueError("Gate config must contain alpha and metrics only")
    alpha = gates.get("alpha", 0.05)
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(gates.get("metrics"), dict) or not gates["metrics"]:
        raise ValueError("At least one metric rule is required")
    for name, rule in gates["metrics"].items():
        if name not in METRIC_DIRECTION:
            raise ValueError(f"Unknown metric: {name}")
        if not isinstance(rule, dict) or set(rule) - {"tolerance", "gate"}:
            raise ValueError(f"Invalid rule for {name}")
        tolerance = rule.get("tolerance", 0)
        if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"Invalid tolerance for {name}")
        if type(rule.get("gate", False)) is not bool:
            raise ValueError(f"gate must be boolean: {name}")
        if rule.get("gate") and name not in BINARY_METRICS:
            raise ValueError(f"{name} has no paired binary test; use gate: false")
    return gates


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def compare_runs(baseline: Path, candidate: Path, gates_path: Path) -> tuple[str, bool]:
    base, cand = load_run(baseline), load_run(candidate)
    require_complete(base)
    require_complete(cand)
    if (base.meta.dataset.sha256, base.meta.dataset.split) != (
        cand.meta.dataset.sha256,
        cand.meta.dataset.split,
    ):
        raise ValueError("Dataset fingerprint or split differs")
    if base.meta.reps != 1 or cand.meta.reps != 1:
        raise ValueError("Paired McNemar comparisons require reps=1 to avoid pseudoreplication")
    if set(base.rows) != set(cand.rows):
        raise ValueError("Case identities differ; runs must be paired exactly")
    if (base.meta.corpus.sha256 if base.meta.corpus else None) != (
        cand.meta.corpus.sha256 if cand.meta.corpus else None
    ):
        raise ValueError("Corpus fingerprint differs; regenerate a comparable baseline")
    gates = load_gates(gates_path)
    warnings = []
    if base.meta.env.machine != cand.meta.env.machine:
        warnings.append("Machine differs. Official comparisons should use the reference machine.")
    if base.meta.env.model_digests != cand.meta.env.model_digests:
        warnings.append(
            "Model digests differ. This may be intentional for a model-swap experiment."
        )
    lines = [
        "# Regression comparison",
        "",
        f"Baseline: `{cell(base.meta.run_id)}`",
        f"Candidate: `{cell(cand.meta.run_id)}`",
        "",
        "Deltas are candidate minus baseline. Tolerances are absolute metric units.",
        "",
        "| Metric | Baseline | Candidate | Delta | Tolerance | b | c | p | Verdict | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    if base.meta.config.get("fixture_only") or cand.meta.config.get("fixture_only"):
        lines.insert(2, "**Offline fixture demonstration. Not measured model performance.**")
    flips = []
    blocked = False
    for metric, rule in gates["metrics"].items():
        bv, cv = base.summary.metrics.get(metric), cand.summary.metrics.get(metric)
        if bv is None or cv is None:
            if rule.get("gate"):
                raise ValueError(f"Gated metric missing/not applicable: {metric}")
            lines.append(f"| {metric} | — | — | — | — | — | — | — | NOT APPLICABLE | no |")
            continue
        direction = METRIC_DIRECTION[metric]
        tolerance = rule.get("tolerance", 0)
        delta = cv - bv
        benefit = delta if direction == "higher" else -delta
        b = c = 0
        p = None
        if metric in BINARY_METRICS:
            pairs = []
            for key in sorted(base.rows):
                old = base.rows[key].scores.get(metric)
                new = cand.rows[key].scores.get(metric)
                if (old is None) != (new is None):
                    raise ValueError(f"Applicability changed for {metric} on {key}")
                if old is None:
                    continue
                if old not in (0, 1) or new not in (0, 1):
                    raise ValueError(f"Nonbinary score for {metric} on {key}")
                pairs.append((old, new))
                old_ok = old == (1 if direction == "higher" else 0)
                new_ok = new == (1 if direction == "higher" else 0)
                if old_ok != new_ok:
                    b += int(old_ok)
                    c += int(new_ok)
                    flips.append(
                        (
                            metric,
                            key[0],
                            "pass→fail" if old_ok else "fail→pass",
                            ", ".join(cand.rows[key].failure_tags) or "none",
                        )
                    )
            if not pairs:
                raise ValueError(f"No paired scores for {metric}")
            if not math.isclose(
                sum(a for a, _ in pairs) / len(pairs), bv, abs_tol=1e-8
            ) or not math.isclose(sum(a for _, a in pairs) / len(pairs), cv, abs_tol=1e-8):
                raise ValueError(f"Aggregate {metric} disagrees with paired scores")
            p = mcnemar_exact(b, c)
        worse = benefit < -tolerance and not math.isclose(benefit, -tolerance, abs_tol=1e-12)
        verdict = (
            ("REGRESSION" if p is not None and p < gates.get("alpha", 0.05) else "POSSIBLE")
            if worse
            else ("IMPROVED" if benefit > 0 else "OK")
        )
        gated = rule.get("gate", False)
        blocked |= gated and verdict == "REGRESSION"
        ptext = f"{p:.6g}" if p is not None else "n/a"
        lines.append(
            f"| {metric} | {bv:.6g} | {cv:.6g} | {delta:+.6g} | {tolerance:g} | {b} | {c} | {ptext} | {verdict} | {'yes' if gated else 'no'} |"
        )
    lines += [
        "",
        "## Flipped cases",
        "",
        "| Metric | Case | Change | Candidate failure tags |",
        "|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(cell(x) for x in flip) + " |" for flip in flips]
    if not flips:
        lines.append("| — | — | no flips | — |")
    lines += [
        "",
        "## Interpretation",
        "",
        "BLOCKED: gated regression detected."
        if blocked
        else "No gated statistically significant regression detected.",
        "A POSSIBLE verdict remains a warning, not evidence of equivalence. Continuous metrics have no McNemar p-value and are warn-only.",
        "Tests are per metric without multiplicity correction. Small datasets have limited power.",
    ]
    lines += ["", "## Environment warnings", "", *warnings] if warnings else []
    text = "\n".join(lines) + "\n"
    (candidate / "compare.md").write_text(text, encoding="utf-8")
    return text, blocked
