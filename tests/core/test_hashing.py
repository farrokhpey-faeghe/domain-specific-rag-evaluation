from pathlib import Path

from deskrag.core.hashing import sha256_json, sha256_text, sha256_text_files


def test_sha256_text_is_hex_digest_of_utf8():
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_json_ignores_key_order():
    assert sha256_json({"a": 1, "b": [1, 2]}) == sha256_json({"b": [1, 2], "a": 1})


def test_sha256_json_changes_when_a_value_changes():
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_sha256_text_files_ignores_argument_order(tmp_path: Path):
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "b.md").write_text("beta")
    paths = [tmp_path / "a.md", tmp_path / "b.md"]
    assert sha256_text_files(paths, root=tmp_path) == sha256_text_files(paths[::-1], root=tmp_path)


def test_sha256_text_files_changes_when_content_changes(tmp_path: Path):
    path = tmp_path / "a.md"
    path.write_text("alpha")
    before = sha256_text_files([path], root=tmp_path)
    path.write_text("alpha!")
    assert sha256_text_files([path], root=tmp_path) != before


def test_sha256_text_files_changes_when_a_file_is_renamed(tmp_path: Path):
    (tmp_path / "a.md").write_text("same")
    before = sha256_text_files([tmp_path / "a.md"], root=tmp_path)
    (tmp_path / "a.md").rename(tmp_path / "c.md")
    assert sha256_text_files([tmp_path / "c.md"], root=tmp_path) != before


def test_sha256_text_files_treats_crlf_and_lf_the_same(tmp_path: Path):
    (tmp_path / "unix").mkdir()
    (tmp_path / "windows").mkdir()
    (tmp_path / "unix" / "a.md").write_bytes(b"line one\nline two\n")
    (tmp_path / "windows" / "a.md").write_bytes(b"line one\r\nline two\r\n")
    unix = sha256_text_files([tmp_path / "unix" / "a.md"], root=tmp_path / "unix")
    windows = sha256_text_files([tmp_path / "windows" / "a.md"], root=tmp_path / "windows")
    assert unix == windows
