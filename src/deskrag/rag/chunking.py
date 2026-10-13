"""Cut a Document into overlapping, paragraph-aware Chunks."""

import re

from deskrag.core.schemas import Chunk, Document


def _split_long(block: str, size_words: int) -> list[str]:
    """A single block bigger than one chunk gets cut by words."""
    words = block.split()
    if len(words) <= size_words:
        return [block]
    return [" ".join(words[i : i + size_words]) for i in range(0, len(words), size_words)]


def chunk_document(doc: Document, size_words: int = 300, overlap_words: int = 50) -> list[Chunk]:
    if overlap_words >= size_words:
        raise ValueError("overlap_words must be smaller than size_words")

    # Step 1: paragraphs (blocks separated by blank lines)
    units: list[str] = []
    for block in re.split(r"\n\s*\n", doc.text):
        block = block.strip()
        if block:
            units.extend(_split_long(block, size_words))

    pieces: list[str] = []  # finished chunk texts
    current: list[str] = []  # paragraphs in the chunk being built
    current_words = 0
    has_new = False  # True once the chunk holds something besides carried-over overlap

    def flush(keep_overlap: bool) -> None:
        nonlocal current, current_words, has_new
        text = "\n\n".join(current)
        pieces.append(text)
        tail = ""
        if keep_overlap and overlap_words:
            tail = " ".join(text.split()[-overlap_words:])
        current = [tail] if tail else []
        current_words = len(tail.split())
        has_new = False

    # Steps 2-4
    for unit in units:
        n = len(unit.split())
        is_heading = unit.lstrip().startswith("#")
        if has_new and current_words + n > size_words:
            flush(keep_overlap=True)
        elif has_new and is_heading and current_words >= size_words // 2:
            flush(keep_overlap=False)  # clean break at a new section
        current.append(unit)
        current_words += n
        has_new = True

    if has_new:
        flush(keep_overlap=False)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{position}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=text,
            position=position,
        )
        for position, text in enumerate(pieces)
    ]
