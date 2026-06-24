import json
from pathlib import Path

from spectra.experiment_plan import (
    build_run_specs,
    derive_rare_labels,
    load_experiment_config,
    materialize_run_queue,
    validate_experiment_inputs,
)


def test_build_run_specs_creates_deterministic_p0_queue():
    config = load_experiment_config(Path("configs/experiments/edgeiiot_fl_spectra_dp.json"))

    specs = build_run_specs(config, final_seeds=False)
    main_dp = [spec for spec in specs if spec.table_id == "main_dp_fl"]

    assert [spec.method for spec in main_dp[:3]] == [
        "fedavg_lora_dp_dplora_style",
        "fedavg_lora_dp_dplora_style",
        "fedavg_lora_dp_dplora_style",
    ]
    assert [spec.epsilon for spec in main_dp[:3]] == [8.0, 4.0, 2.0]
    assert {spec.seed for spec in specs} == {20260531}
    assert "main_dp_fl__spectra_fedcore_dp__seed20260531__eps4" in {spec.run_id for spec in specs}


def test_derive_rare_labels_uses_training_counts_only():
    split = {
        "split_label_counts": {
            "Normal": {"train": 1000, "val": 1, "test": 1},
            "MITM": {"train": 10, "val": 100, "test": 100},
            "XSS": {"train": 20, "val": 1, "test": 1},
            "DDoS_TCP": {"train": 30, "val": 1, "test": 1},
        }
    }

    assert derive_rare_labels(split, policy="below_median_attack_count") == ["MITM", "XSS"]


def test_validate_experiment_inputs_reports_missing_raw_data_without_failing(tmp_path: Path):
    config = load_experiment_config(Path("configs/experiments/edgeiiot_fl_spectra_dp.json"))
    config["data"]["csv_path"] = str(tmp_path / "missing.csv")

    report = validate_experiment_inputs(config, repo_root=Path.cwd(), strict_raw_data=False)

    assert report["ok"] is True
    assert any(item["name"] == "raw_edgeiiot_csv" and not item["exists"] for item in report["checks"])


def test_materialize_run_queue_writes_per_run_configs(tmp_path: Path):
    config = load_experiment_config(Path("configs/experiments/edgeiiot_fl_spectra_dp.json"))
    specs = build_run_specs(config, final_seeds=False)[:2]

    summary = materialize_run_queue(config, specs, output_dir=tmp_path)

    queue_path = tmp_path / "run_queue.jsonl"
    assert summary["run_count"] == 2
    assert queue_path.exists()
    lines = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    first_config = tmp_path / "runs" / lines[0]["run_id"] / "config_resolved.json"
    first_status = tmp_path / "runs" / lines[0]["run_id"] / "status.json"
    assert first_config.exists()
    assert first_status.exists()
    payload = json.loads(first_config.read_text(encoding="utf-8"))
    assert payload["run"]["run_id"] == lines[0]["run_id"]
    assert payload["artifacts"]["required_files"] == [
        "config_resolved.json",
        "metrics.json",
        "metrics_by_class.json",
        "privacy_ledger.json",
        "communication.json",
        "hardware_profile.json",
        "client_label_histograms.json",
        "preprocessing_fingerprint.json",
        "budget_match.json",
        "run.log",
    ]
