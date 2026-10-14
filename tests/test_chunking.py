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


def paragraph(tag: str, n_words: int) -> str:
    """A paragraph of n distinct words, so overlap can be checked exactly."""
    return " ".join(f"{tag}w{j}" for j in range(n_words))


def test_next_chunk_starts_with_the_last_words_of_the_previous_one():
    text = "\n\n".join(paragraph(f"p{i}", 60) for i in range(4))
    chunks = chunk_document(make_doc(text), size_words=100, overlap_words=10)
    assert len(chunks) >= 3
    for before, after in zip(chunks, chunks[1:], strict=False):
        assert after.text.split()[:10] == before.text.split()[-10:]


def test_heading_starts_a_new_chunk_once_the_current_one_is_half_full():
    text = paragraph("a", 60) + "\n\n# New Section\n\n" + paragraph("b", 20)
    chunks = chunk_document(make_doc(text), size_words=100, overlap_words=10)
    assert len(chunks) == 2
    assert chunks[1].text.startswith("# New Section")


def test_heading_does_not_split_a_chunk_that_is_less_than_half_full():
    text = paragraph("a", 20) + "\n\n# New Section\n\n" + paragraph("b", 20)
    chunks = chunk_document(make_doc(text), size_words=100, overlap_words=10)
    assert len(chunks) == 1


def test_chunks_never_exceed_size_plus_overlap():
    text = "\n\n".join(paragraph(f"p{i}", 60) for i in range(8))
    chunks = chunk_document(make_doc(text), size_words=100, overlap_words=10)
    assert all(len(c.text.split()) <= 110 for c in chunks)
