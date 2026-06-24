import json
import os
import subprocess
import sys
from pathlib import Path

from spectra.experiment_plan import build_run_specs, load_experiment_config, materialize_run_queue


def test_run_single_experiment_prepare_only_marks_run_ready(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    config = load_experiment_config(repo / "configs/experiments/edgeiiot_fl_spectra_dp.json")
    spec = build_run_specs(config, final_seeds=False)[0]
    materialize_run_queue(config, [spec], output_dir=tmp_path)
    run_config = tmp_path / "runs" / spec.run_id / "config_resolved.json"

    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [sys.executable, "scripts/run_single_experiment.py", "--run-config", str(run_config), "--prepare-only"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = run_config.parent
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    request = json.loads((run_dir / "runner_request.json").read_text(encoding="utf-8"))
    assert status["status"] == "ready_for_runner"
    assert request["run"]["run_id"] == spec.run_id
    assert "qwen_peft_federated" in request["runner_contract"]["expected_runner"]
