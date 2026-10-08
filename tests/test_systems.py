from pathlib import Path

import pytest

from deskrag.core.config import SystemConfig, load_config
from deskrag.core.fakes import FakeRAGSystem
from deskrag.systems import build_system

REPO = Path(__file__).resolve().parents[1]


def test_builds_the_fake_system_from_the_repo_config(monkeypatch):
    monkeypatch.chdir(REPO)  # corpus_dir in configs is relative to the repo root
    system = build_system(load_config(REPO / "configs" / "fake.yaml"))
    assert isinstance(system, FakeRAGSystem)


@pytest.mark.parametrize(
    "kind, target",
    [
        ("rag", "deskrag.rag.pipeline.build_rag_system"),
        ("closed_book", "deskrag.rag.closed_book.build_closed_book_system"),
        ("oracle", "deskrag.evaluation.sanity.build_sanity_system"),
        ("always_abstain", "deskrag.evaluation.sanity.build_sanity_system"),
        ("never_abstain", "deskrag.evaluation.sanity.build_sanity_system"),
    ],
)
def test_dispatches_to_the_owning_workstreams_builder(monkeypatch, kind, target):
    built = []
    monkeypatch.setattr(target, lambda cfg, **kwargs: built.append(cfg.system) or "built")
    llm = {"model": "tiny:1b"} if kind in {"rag", "closed_book"} else None
    cfg = SystemConfig(name="x", system=kind, llm=llm)
    assert build_system(cfg, cases=[]) == "built"
    assert built == [kind]


def test_sanity_systems_need_the_test_cases():
    with pytest.raises(ValueError, match="cases"):
        build_system(SystemConfig(name="x", system="oracle"))
