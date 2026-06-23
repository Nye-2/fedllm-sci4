import importlib.util
import json
from pathlib import Path

import pytest


def load_results_module():
    script = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "results_to_latex.py"
    spec = importlib.util.spec_from_file_location("results_to_latex", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_results_table_requires_and_renders_total_uploaded_mb():
    module = load_results_module()
    records = [
        {
            "method": "spectra",
            "seed": 20260531,
            "privacy": {"epsilon": 4.0},
            "macro_f1": 0.7,
            "balanced_accuracy": 0.6,
            "rare_attack_recall": 0.5,
            "uploaded_mb_per_client_round": 1.25,
            "total_uploaded_mb": 62.5,
            "trainable_params": 4096,
        }
    ]

    table = module.make_table(records, precision=3)

    assert "Total MB" in table
    assert "62.500" in table


def test_results_table_rejects_missing_total_uploaded_mb(tmp_path):
    module = load_results_module()
    record = {
        "method": "spectra",
        "seed": 20260531,
        "privacy": {"epsilon": 4.0},
        "macro_f1": 0.7,
        "balanced_accuracy": 0.6,
        "rare_attack_recall": 0.5,
        "uploaded_mb_per_client_round": 1.25,
        "trainable_params": 4096,
    }
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    records = module.load_records(path)

    with pytest.raises(SystemExit, match="total_uploaded_mb"):
        module.make_table(records, precision=3)
