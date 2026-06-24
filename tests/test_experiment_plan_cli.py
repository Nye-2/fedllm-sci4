import json
import os
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, "scripts/run_experiment_plan.py", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_cli_writes_report(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "validation.json"

    result = run_cli(
        "validate",
        "--config",
        "configs/experiments/edgeiiot_fl_spectra_dp.json",
        "--output",
        str(report_path),
        cwd=repo,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "checks" in report
    assert any(check["name"] == "selected_split_artifact" for check in report["checks"])


def test_materialize_cli_writes_limited_queue(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "queue"

    result = run_cli(
        "materialize",
        "--config",
        "configs/experiments/edgeiiot_fl_spectra_dp.json",
        "--output-dir",
        str(out_dir),
        "--limit",
        "3",
        cwd=repo,
    )

    assert result.returncode == 0, result.stderr
    queue = out_dir / "run_queue.jsonl"
    runbook = out_dir / "RUNBOOK.md"
    assert queue.exists()
    assert runbook.exists()
    assert len(queue.read_text(encoding="utf-8").splitlines()) == 3
