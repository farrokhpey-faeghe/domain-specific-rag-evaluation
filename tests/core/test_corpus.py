from datetime import date
from pathlib import Path

import pytest

from deskrag.core.corpus import CorpusError, corpus_sha256, load_corpus, load_document

REPO = Path(__file__).resolve().parents[2]
SAMPLE_CORPUS = REPO / "data" / "samples" / "corpus"

DOC = """---
doc_id: it-wifi
title: Connecting to campus Wi-Fi
url: https://help.northgate.example/it/wifi
section: IT
snapshot_date: 2026-10-08
---

# Connecting to campus Wi-Fi

Visitors can join NU-Guest.
"""


def write_doc(directory: Path, name: str = "it-wifi.md", text: str = DOC) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_document_parses_front_matter_and_body(tmp_path):
    doc = load_document(write_doc(tmp_path))
    assert doc.doc_id == "it-wifi"
    assert doc.section == "IT"
    assert doc.snapshot_date == date(2026, 10, 8)
    assert doc.text == "# Connecting to campus Wi-Fi\n\nVisitors can join NU-Guest."


def test_doc_id_must_match_the_file_name(tmp_path):
    with pytest.raises(CorpusError, match="it-wifi"):
        load_document(write_doc(tmp_path, name="wifi.md"))


def test_missing_front_matter_is_an_error(tmp_path):
    with pytest.raises(CorpusError, match="front-matter"):
        load_document(write_doc(tmp_path, text="# No front-matter here\n"))


def test_invalid_front_matter_error_names_the_file(tmp_path):
    with pytest.raises(CorpusError, match="it-wifi.md"):
        load_document(
            write_doc(tmp_path, text=DOC.replace("title: Connecting to campus Wi-Fi\n", ""))
        )


def test_load_corpus_returns_documents_sorted_by_id():
    docs = load_corpus(SAMPLE_CORPUS)
    assert len(docs) == 5
    assert [d.doc_id for d in docs] == sorted(d.doc_id for d in docs)


def test_empty_corpus_dir_is_an_error(tmp_path):
    with pytest.raises(CorpusError, match="no documents"):
        load_corpus(tmp_path)


def test_corpus_sha256_changes_when_a_document_changes(tmp_path):
    write_doc(tmp_path)
    before = corpus_sha256(tmp_path)
    write_doc(tmp_path, text=DOC.replace("NU-Guest", "NU-Visitor"))
    assert corpus_sha256(tmp_path) != before
