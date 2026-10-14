"""Baseline promotion and execution of WS4's evaluation command."""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from deskrag.evaluation.compare import load_run, require_complete
from deskrag.evaluation.report import report_run


def promote_baseline(run_dir: Path, name: str, root: Path = Path("baselines")) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Baseline name must be a simple name, not a path")
    run = load_run(run_dir)
    require_complete(run)
    if run.meta.config.get("fixture_only") and not name.startswith("demo-"):
        raise ValueError("Fixture baselines must use a demo- name; they are not official results")
    if run.meta.dataset.split != "dev":
        raise ValueError("Only dev baselines can be promoted; never tune on test")
    if run.meta.reps != 1:
        raise ValueError("Regression baselines require reps=1")
    target = root / name
    if target.exists():
        raise ValueError(
            f"Baseline already exists: {target}; review/remove it explicitly before replacing"
        )
    if root.is_symlink():
        raise ValueError("Baseline root must not be a symlink")
    report_run(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".promote-", dir=root))
    try:
        for filename in (
            "meta.json",
            "summary.json",
            "results.jsonl",
            "errors.jsonl",
            "report.md",
            "compare.md",
        ):
            source = run_dir / filename
            if source.is_file():
                shutil.copy2(source, stage / filename)
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def evaluate_candidate(config: Path, retrieval_only: bool = False) -> Path:
    """Find the new run by set difference, never by a stale latest-run guess."""
    root = Path("runs")
    before = set(root.glob("*")) if root.exists() else set()
    args = [
        sys.executable,
        "-c",
        "from deskrag.cli import app; app()",
        "eval",
        "run",
        "--config",
        str(config),
        "--split",
        "dev",
    ]
    if retrieval_only:
        args.append("--retrieval-only")
    proc = subprocess.run(args, check=False)
    if proc.returncode:
        raise ValueError(
            f"Evaluation failed (exit {proc.returncode}); no baseline comparison performed"
        )
    created = [
        p for p in set(root.glob("*")) - before if p.is_dir() and (p / "summary.json").is_file()
    ]
    if len(created) != 1:
        raise ValueError(
            "Expected exactly one new completed run; run eval separately and pass --candidate"
        )
    return created[0]
