"""WS4 (feat/eval-metrics): metric modules.

Each module (retrieval.py, abstention.py, answer.py, ops.py) exposes
    score_case(case: TestCase, answer: RAGAnswer) -> dict[str, float | None]   # None = not applicable
    aggregate(rows: list[CaseResult]) -> dict[str, float | None]
Metric names and their better direction are listed in core.schemas.METRIC_DIRECTION.
"""
