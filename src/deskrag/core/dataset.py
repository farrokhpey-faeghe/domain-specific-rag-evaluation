"""Load labeled test cases from JSONL.

`path` can be one .jsonl file or a directory of them. The test set lives in
`data/testset/<author>.jsonl` (one file per person, so parallel edits don't conflict in git);
each row is one `TestCase`.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from deskrag.core.hashing import sha256_json
from deskrag.core.schemas import TestCase


class DatasetError(ValueError):
    """A test-set file is malformed or inconsistent."""


def load_cases(path: str | Path, split: str | None = None) -> list[TestCase]:
    """Cases in file order (files sorted by name), optionally keeping one split."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise DatasetError(f"{path}: no *.jsonl files found")

    cases: list[TestCase] = []
    first_seen: dict[str, str] = {}
    for file in files:
        for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{file}:{line_no}"
            try:
                case = TestCase.model_validate_json(line)
            except ValidationError as exc:
                raise DatasetError(f"{where}: {exc}") from exc
            if case.case_id in first_seen:
                raise DatasetError(
                    f"{where}: duplicate case_id {case.case_id!r} (first seen at {first_seen[case.case_id]})"
                )
            first_seen[case.case_id] = where
            cases.append(case)
    return [case for case in cases if split is None or case.split == split]


def dataset_sha256(cases: Iterable[TestCase]) -> str:
    """Fingerprint of a set of cases, independent of file layout and order."""
    rows = sorted((case.model_dump(mode="json") for case in cases), key=lambda row: row["case_id"])
    return sha256_json(rows)
