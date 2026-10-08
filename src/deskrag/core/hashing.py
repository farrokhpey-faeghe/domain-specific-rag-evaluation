"""Stable content hashes used to fingerprint configs, corpora, datasets, and runs."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a JSON-serializable object. Dict key order does not matter."""
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return sha256_text(canonical)


def sha256_text_files(paths: Iterable[Path], root: Path) -> str:
    """Hash of text files: their paths relative to `root` plus their contents.

    Argument order does not matter, and CRLF/LF line endings hash the same, so teammates on
    Windows and macOS get identical corpus fingerprints.
    """
    entries = sorted((path.relative_to(root).as_posix(), path) for path in paths)
    digest = hashlib.sha256()
    for relative, path in entries:
        content_hash = sha256_text(path.read_text(encoding="utf-8"))  # universal newlines
        digest.update(f"{relative}\0{content_hash}\n".encode())
    return digest.hexdigest()
