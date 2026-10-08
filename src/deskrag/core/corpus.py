"""Load the document corpus: one markdown file per page, with YAML front-matter.

---
doc_id: it-wifi                  # must equal the file name (it-wifi.md)
title: Connecting to campus Wi-Fi
url: https://...
section: IT
snapshot_date: 2026-10-08
---
# Connecting to campus Wi-Fi
...
"""

import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from deskrag.core.hashing import sha256_text_files
from deskrag.core.schemas import Document

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)(.*)\Z", re.DOTALL)


class CorpusError(ValueError):
    """A corpus file is missing, malformed, or inconsistent."""


def load_document(path: Path) -> Document:
    match = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise CorpusError(f"{path}: missing YAML front-matter (the file must start with '---')")
    meta = yaml.safe_load(match.group(1)) or {}
    try:
        doc = Document(**meta, text=match.group(2).strip())
    except (TypeError, ValidationError) as exc:
        raise CorpusError(f"{path}: invalid front-matter: {exc}") from exc
    if doc.doc_id != path.stem:
        raise CorpusError(f"{path}: doc_id {doc.doc_id!r} must match the file name {path.stem!r}")
    return doc


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """All documents in `corpus_dir`, sorted by doc_id."""
    paths = sorted(Path(corpus_dir).glob("*.md"))
    if not paths:
        raise CorpusError(f"{corpus_dir}: no documents (*.md) found")
    return [load_document(path) for path in paths]


def corpus_sha256(corpus_dir: str | Path) -> str:
    corpus_dir = Path(corpus_dir)
    return sha256_text_files(corpus_dir.glob("*.md"), root=corpus_dir)
