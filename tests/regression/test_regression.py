import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deskrag.cli import app
from deskrag.evaluation.baseline import promote_baseline
from deskrag.evaluation.compare import compare_runs, load_gates
from deskrag.evaluation.report import report_run
from deskrag.evaluation.stats import mcnemar_exact

ROOT = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[2]
GATES = REPO / "configs/regression.yaml"


@pytest.fixture
def pair(tmp_path):
    for name in ("gate-baseline", "gate-candidate"):
        shutil.copytree(ROOT / name, tmp_path / name)
    return tmp_path / "gate-baseline", tmp_path / "gate-candidate"


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [(1, 9, 0.021484375), (9, 1, 0.021484375), (0, 0, 1), (0, 8, 0.0078125), (2, 2, 1)],
)
def test_exact_mcnemar(b, c, p):
    assert mcnemar_exact(b, c) == p


@pytest.mark.parametrize("value", [-1, True, 1.2])
def test_bad_counts(value):
    with pytest.raises(ValueError):
        mcnemar_exact(value, 0)


def test_regression_and_flips(pair):
    text, blocked = compare_runs(*pair, GATES)
    assert blocked and "REGRESSION" in text and "0.0078125" in text
    assert "demo-04" in text and "pass→fail" in text
    assert (pair[1] / "compare.md").read_text(encoding="utf-8") == text


def test_identity_is_ok(pair):
    text, blocked = compare_runs(pair[0], pair[0], GATES)
    assert not blocked and "no flips" in text


def test_improvement(pair):
    text, blocked = compare_runs(pair[1], pair[0], GATES)
    assert not blocked and "IMPROVED" in text and "fail→pass" in text


def test_small_drop_possible(tmp_path):
    shutil.copytree(ROOT / "fake-candidate", tmp_path / "cand")
    text, blocked = compare_runs(ROOT / "fake-baseline", tmp_path / "cand", GATES)
    assert not blocked and "POSSIBLE" in text


def alter(path, filename, change):
    p = path / filename
    obj = json.loads(p.read_text())
    change(obj)
    p.write_text(json.dumps(obj))


@pytest.mark.parametrize("field", ["sha256", "split"])
def test_incompatible_data(pair, field):
    alter(pair[1], "meta.json", lambda x: x["dataset"].update({field: "other"}))
    with pytest.raises(ValueError, match="Dataset"):
        compare_runs(*pair, GATES)


def test_environment_warnings(pair):
    alter(
        pair[1], "meta.json", lambda x: x["env"].update(machine="other", model_digests={"a": "b"})
    )
    text, _ = compare_runs(*pair, GATES)
    assert "Machine differs" in text and "Model digests differ" in text


def test_partial_run_rejected(pair):
    p = pair[1] / "results.jsonl"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        compare_runs(*pair, GATES)


def test_duplicate_rejected(pair):
    p = pair[1] / "results.jsonl"
    p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        compare_runs(*pair, GATES)


def test_missing_gated_metric(pair):
    alter(pair[1], "summary.json", lambda x: x["metrics"].pop("hit@5"))
    with pytest.raises(ValueError, match="Gated metric"):
        compare_runs(*pair, GATES)


def test_infra_errors_rejected(pair):
    alter(pair[1], "summary.json", lambda x: x["counts"].update(errors=1))
    with pytest.raises(ValueError, match="infrastructure"):
        compare_runs(*pair, GATES)


def test_aggregate_tampering_rejected(pair):
    alter(pair[1], "summary.json", lambda x: x["metrics"].update(e2e_pass=0.99))
    with pytest.raises(ValueError, match="disagrees"):
        compare_runs(*pair, GATES)


def test_continuous_gate_rejected(tmp_path):
    p = tmp_path / "gates.yaml"
    p.write_text("metrics:\n  mrr: {gate: true}\n")
    with pytest.raises(ValueError, match="binary"):
        load_gates(p)


def test_lower_is_better(pair, tmp_path):
    for path in pair:
        p = path / "results.jsonl"
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        for row in rows:
            row["scores"]["format_error_rate"] = 1 - row["scores"]["e2e_pass"]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        alter(
            path,
            "summary.json",
            lambda x: x["metrics"].update(format_error_rate=1 - x["metrics"]["e2e_pass"]),
        )
    gates = tmp_path / "lower.yaml"
    gates.write_text("metrics:\n  format_error_rate: {tolerance: 0.05, gate: true}\n")
    text, blocked = compare_runs(*pair, gates)
    assert blocked and "REGRESSION" in text


def test_promotion_no_overwrite_or_traversal(pair, tmp_path):
    target = promote_baseline(pair[0], "demo-example", tmp_path / "baselines")
    assert (target / "report.md").is_file()
    with pytest.raises(ValueError, match="already exists"):
        promote_baseline(pair[0], "demo-example", tmp_path / "baselines")
    with pytest.raises(ValueError, match="simple name"):
        promote_baseline(pair[0], "../escape", tmp_path / "baselines")


def test_test_split_cannot_be_promoted(pair, tmp_path):
    alter(pair[0], "meta.json", lambda x: x["dataset"].update(split="test"))
    with pytest.raises(ValueError, match="dev"):
        promote_baseline(pair[0], "demo-test", tmp_path / "baselines")


def test_report_and_cli(pair):
    assert "Failed cases" in report_run(pair[1])
    runner = CliRunner()
    assert (
        runner.invoke(app, ["compare", str(pair[0]), str(pair[1]), "--gates", str(GATES)]).exit_code
        == 1
    )
    assert (
        runner.invoke(app, ["compare", str(pair[0]), "missing", "--gates", str(GATES)]).exit_code
        == 2
    )
    assert runner.invoke(app, ["runs", "table", str(pair[0]), str(pair[1])]).exit_code == 0
    assert runner.invoke(app, ["runs", "list", "--runs-dir", str(pair[0].parent)]).exit_code == 0
    assert (
        runner.invoke(
            app,
            [
                "regress",
                "--candidate",
                str(pair[1]),
                "--baseline",
                str(pair[0]),
                "--gates",
                str(GATES),
            ],
        ).exit_code
        == 1
    )
