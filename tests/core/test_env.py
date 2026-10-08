import platform
import subprocess

from deskrag.core.env import basic_fingerprint, git_info


def git(*args: str, cwd) -> None:
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_basic_fingerprint_uses_the_machine_override(monkeypatch):
    monkeypatch.setenv("DESKRAG_MACHINE", "reference-mbp")
    fingerprint = basic_fingerprint()
    assert fingerprint.machine == "reference-mbp"
    assert fingerprint.python == platform.python_version()


def test_git_info_reports_sha_and_dirty_state(tmp_path):
    git("init", "-q", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("x")
    git("add", ".", cwd=tmp_path)
    git(
        "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init", cwd=tmp_path
    )
    sha, dirty = git_info(tmp_path)
    assert sha is not None and len(sha) == 40
    assert dirty is False
    (tmp_path / "a.txt").write_text("y")
    assert git_info(tmp_path)[1] is True


def test_git_info_outside_a_repo_is_unknown(tmp_path):
    assert git_info(tmp_path) == (None, False)
