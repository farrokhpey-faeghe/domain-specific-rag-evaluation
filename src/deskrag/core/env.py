"""Facts about where and from what code a run was produced.

Ollama output can differ across hardware, so every run records its machine; `compare` warns when a
baseline and a candidate come from different machines. Set DESKRAG_MACHINE to give the reference
machine a stable name.
"""

import os
import platform
import subprocess
from pathlib import Path

from deskrag.core.schemas import EnvFingerprint


def basic_fingerprint() -> EnvFingerprint:
    """Machine facts. Whoever talks to Ollama fills in `ollama_version` and `model_digests`."""
    return EnvFingerprint(
        machine=os.environ.get("DESKRAG_MACHINE") or platform.node(),
        platform=platform.platform(),
        python=platform.python_version(),
    )


def git_info(repo: str | Path = ".") -> tuple[str | None, bool]:
    """(HEAD sha, has uncommitted changes), or (None, False) outside a git checkout."""
    try:
        sha = _git(repo, "rev-parse", "HEAD")
        dirty = bool(_git(repo, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return sha, dirty


def _git(repo: str | Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()
