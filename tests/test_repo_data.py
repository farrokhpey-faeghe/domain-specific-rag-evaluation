"""Checks on data committed to the repo. CI runs these, so a typo in a test case fails the PR."""

from pathlib import Path

import pytest

from deskrag.core.corpus import load_corpus
from deskrag.core.dataset import load_cases

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "data" / "samples"
TESTSET = REPO / "data" / "testset"
CORPUS = REPO / "data" / "corpus" / "docs"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def test_sample_gold_docs_exist_and_contain_every_required_fact():
    docs = {doc.doc_id: normalize(doc.text) for doc in load_corpus(SAMPLES / "corpus")}
    for case in load_cases(SAMPLES / "cases.jsonl"):
        missing = [doc_id for doc_id in case.gold_doc_ids if doc_id not in docs]
        assert not missing, f"{case.case_id}: unknown gold docs {missing}"
        for fact in case.must_include:
            alternatives = [normalize(alt) for alt in fact.split("|")]
            found = any(alt in docs[doc_id] for doc_id in case.gold_doc_ids for alt in alternatives)
            assert found, f"{case.case_id}: {fact!r} does not appear in its gold docs"


@pytest.mark.skipif(not any(TESTSET.glob("*.jsonl")), reason="no test cases committed yet")
def test_committed_test_cases_are_valid():
    assert load_cases(TESTSET)


@pytest.mark.skipif(not any(CORPUS.glob("*.md")), reason="no corpus committed yet")
def test_committed_corpus_documents_are_valid():
    assert load_corpus(CORPUS)
