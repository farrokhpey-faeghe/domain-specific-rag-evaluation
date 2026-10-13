from datetime import date

from deskrag.core.schemas import Document
from deskrag.rag.chunking import chunk_document


def make_doc(text: str) -> Document:
    return Document(
        doc_id="demo",
        title="Demo",
        url="https://example.edu",
        section="IT",
        snapshot_date=date(2026, 10, 8),
        text=text,
    )


def test_ids_follow_the_contract():
    doc = make_doc("# A\n\n" + "\n\n".join(["word " * 120] * 5))
    chunks = chunk_document(doc, size_words=200, overlap_words=20)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"demo#{i}" for i in range(len(chunks))]


def test_short_doc_is_one_chunk():
    assert len(chunk_document(make_doc("Short page."))) == 1


def test_empty_doc_gives_no_chunks():
    assert chunk_document(make_doc("")) == []
