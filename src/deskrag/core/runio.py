"""Run directories: where an eval run's files live and how they are read and written.

runs/<run_id>/
    meta.json       RunMeta     what ran: config, git sha, dataset/corpus hashes, machine
    results.jsonl   CaseResult  one row per (case, rep), appended as each case finishes
    errors.jsonl    ErrorRow    attempts that never produced a scorable answer
    summary.json    RunSummary  aggregate metrics
    report.md                   human-readable report
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

RUNS_DIR = Path("runs")

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def results(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def errors(self) -> Path:
        return self.root / "errors.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"


def make_run_id(config_name: str, config_hash: str, now: datetime | None = None) -> str:
    """e.g. 20261008-153012-baseline-1a2b3c4d (UTC time, slugged name, short config hash)."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", config_name.lower()).strip("-")
    return f"{stamp}-{slug}-{config_hash[:8]}"


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Append one record as a line. Call it as each case finishes so a crash loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def write_json(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, model: type[M]) -> M:
    return model.model_validate_json(path.read_text(encoding="utf-8"))
