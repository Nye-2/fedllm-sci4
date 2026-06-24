#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    run_config_path = Path(args.run_config)
    payload = _load_json(run_config_path)
    run = payload["run"]
    run_dir = run_config_path.parent
    request = build_runner_request(payload)
    _write_json(run_dir / "runner_request.json", request)
    if args.prepare_only:
        _write_json(
            run_dir / "status.json",
            {
                "run_id": run["run_id"],
                "status": "ready_for_runner",
                "next_action": "Implement or dispatch the expected runner, then write all required artifacts.",
                "runner_request": "runner_request.json",
                "required_artifacts": payload["artifacts"]["required_files"],
            },
        )
        print(json.dumps({"run_id": run["run_id"], "status": "ready_for_runner"}, indent=2, sort_keys=True))
        return 0

    raise SystemExit(
        "This repository contains the experiment orchestration contract. "
        f"Run {run_config_path} with --prepare-only, then execute runner={run['expected_runner']} on the server."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or dispatch one materialized SPECTRA-FedCore run.")
    parser.add_argument("--run-config", required=True, help="Path to a materialized config_resolved.json.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write runner_request.json and mark the run ready without producing metrics.",
    )
    return parser.parse_args()


def build_runner_request(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload["run"]
    base = payload["base_config"]
    required_files = payload["artifacts"]["required_files"]
    return {
        "run": run,
        "dataset": {
            "csv_path": base["data"]["csv_path"],
            "split_artifact": "data/processed/edgeiiot/selected_ml_stratified_split_seed20260531.json",
            "client_partition_artifact": "data/processed/edgeiiot/selected_ml_clients_seed20260531_K10_alpha0.5.json",
            "prompt_version": base["data"]["prompt_version"],
            "exclude_columns": base["data"]["exclude_columns"],
        },
        "model": base["model"],
        "fl": base["fl"],
        "privacy": {
            **base["privacy"],
            "enabled_for_this_run": bool(run["privacy_enabled"]),
            "epsilon_for_this_run": run["epsilon"],
        },
        "runner_contract": {
            "expected_runner": run["expected_runner"],
            "required_outputs": required_files,
            "do_not_write_placeholder_metrics": True,
            "paper_result_policy": "Only write metrics.json after the actual model/baseline has run.",
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
