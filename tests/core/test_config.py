from pathlib import Path

import pytest
from pydantic import ValidationError

from deskrag.core.config import SystemConfig, config_hash, load_config

REPO = Path(__file__).resolve().parents[2]


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_applies_defaults(tmp_path):
    cfg = load_config(write(tmp_path / "c.yaml", "name: mini\nsystem: fake\n"))
    assert cfg.retrieval.method == "bm25"
    assert cfg.retrieval.k == 5
    assert cfg.abstention.min_score is None
    assert cfg.llm is None


def test_extends_deep_merges_parent_and_child(tmp_path):
    write(
        tmp_path / "base.yaml",
        "name: base\nretrieval:\n  method: dense\n  k: 5\nllm:\n  model: tiny:1b\n",
    )
    child = write(tmp_path / "child.yaml", "extends: base.yaml\nname: k3\nretrieval:\n  k: 3\n")
    cfg = load_config(child)
    assert cfg.name == "k3"
    assert cfg.retrieval.k == 3
    assert cfg.retrieval.method == "dense"  # sibling keys from the parent survive
    assert cfg.llm.model == "tiny:1b"


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path / "c.yaml", "name: x\nsystem: fake\nretreival:\n  k: 3\n"))


def test_llm_backed_systems_require_an_llm_section(tmp_path):
    with pytest.raises(ValidationError, match="llm"):
        load_config(write(tmp_path / "c.yaml", "name: x\nsystem: rag\n"))


def test_config_hash_is_stable(tmp_path):
    cfg = SystemConfig(name="a", system="fake")
    assert config_hash(cfg, prompts_dir=tmp_path) == config_hash(
        cfg.model_copy(), prompts_dir=tmp_path
    )


def test_config_hash_ignores_the_name(tmp_path):
    a = SystemConfig(name="a", system="fake")
    b = SystemConfig(name="b", system="fake")
    assert config_hash(a, prompts_dir=tmp_path) == config_hash(b, prompts_dir=tmp_path)


def test_config_hash_changes_when_a_setting_changes(tmp_path):
    a = SystemConfig(name="a", system="fake")
    b = SystemConfig(name="a", system="fake", retrieval={"k": 3})
    assert config_hash(a, prompts_dir=tmp_path) != config_hash(b, prompts_dir=tmp_path)


def test_config_hash_changes_when_the_prompt_text_changes(tmp_path):
    cfg = SystemConfig(name="a", system="fake", prompt_version="v1")
    (tmp_path / "v1.md").write_text("Answer only from the context.", encoding="utf-8")
    before = config_hash(cfg, prompts_dir=tmp_path)
    (tmp_path / "v1.md").write_text("Answer only from the context below.", encoding="utf-8")
    assert config_hash(cfg, prompts_dir=tmp_path) != before


@pytest.mark.parametrize("name", ["baseline", "fake"])
def test_repo_configs_load(name):
    assert load_config(REPO / "configs" / f"{name}.yaml").name == name


@pytest.mark.parametrize(
    "path", sorted((REPO / "configs" / "experiments").glob("*.yaml")), ids=lambda p: p.name
)
def test_experiment_configs_load(path):
    assert load_config(path).name
