"""Format-error, latency, and token-usage metrics for WS4."""

from deskrag.core.schemas import CaseResult, RAGAnswer, TestCase


def _percentile(values: list[float], fraction: float) -> float | None:
    """Calculate a percentile using linear interpolation."""

    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def score_case(
    case: TestCase,
    answer: RAGAnswer,
) -> dict[str, float | None]:
    """Record operational measurements for one response."""

    return {
        "format_error_rate": float(answer.status != "ok"),
        "latency_s": answer.usage.latency_s,
        "prompt_tokens": float(answer.usage.prompt_tokens),
        "completion_tokens": float(answer.usage.completion_tokens),
    }


def aggregate(
    rows: list[CaseResult],
) -> dict[str, float | None]:
    """Calculate overall operational metrics."""

    metrics: dict[str, float | None] = {}

    mean_metrics = {
        "format_error_rate": "format_error_rate",
        "prompt_tokens_mean": "prompt_tokens",
        "completion_tokens_mean": "completion_tokens",
    }

    for metric_name, score_name in mean_metrics.items():
        values: list[float] = []

        for row in rows:
            value = row.scores.get(score_name)

            if value is not None:
                values.append(value)

        metrics[metric_name] = sum(values) / len(values) if values else None

    latencies: list[float] = []

    for row in rows:
        value = row.scores.get("latency_s")

        if value is not None:
            latencies.append(value)

    metrics["latency_p50_s"] = _percentile(latencies, 0.50)
    metrics["latency_p95_s"] = _percentile(latencies, 0.95)

    return metrics
