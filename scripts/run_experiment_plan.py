#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectra.experiment_plan import (
    build_run_specs,
    load_experiment_config,
    materialize_run_queue,
    validate_experiment_inputs,
    write_validation_report,
)


def main() -> int:
    args = parse_args()
    if args.command == "validate":
        return run_validate(args)
    if args.command == "materialize":
        return run_materialize(args)
    raise SystemExit(f"unknown command: {args.command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and materialize SPECTRA-FedCore experiment run queues.")
    parser.add_argument(
        "--config",
        default="configs/experiments/edgeiiot_fl_spectra_dp.json",
        help="Experiment JSON config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate local/server inputs before launching experiments.")
    validate.add_argument("--config", default=None, help="Experiment JSON config.")
    validate.add_argument("--repo-root", default=".", help="Repository root.")
    validate.add_argument("--strict-raw-data", action="store_true", help="Fail when raw data is missing.")
    validate.add_argument("--output", default=None, help="Optional JSON report path.")

    materialize = subparsers.add_parser("materialize", help="Write run_queue.jsonl and per-run config directories.")
    materialize.add_argument("--config", default=None, help="Experiment JSON config.")
    materialize.add_argument("--output-dir", default=None, help="Output directory. Defaults to config output_dir/run_queue.")
    materialize.add_argument("--final-seeds", action="store_true", help="Use all final-table seeds instead of seed only.")
    materialize.add_argument("--include-optional", action="store_true", help="Include P2 optional sensitivity runs.")
    materialize.add_argument("--limit", type=int, default=None, help="Materialize only the first N runs.")
    return parser.parse_args()


def run_validate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config or "configs/experiments/edgeiiot_fl_spectra_dp.json")
    report = validate_experiment_inputs(
        config,
        repo_root=Path(args.repo_root),
        strict_raw_data=bool(args.strict_raw_data),
    )
    if args.output:
        write_validation_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


def run_materialize(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config or "configs/experiments/edgeiiot_fl_spectra_dp.json")
    specs = build_run_specs(config, final_seeds=args.final_seeds, include_optional=args.include_optional)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        specs = specs[: args.limit]
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output_dir"]) / "run_queue"
    summary: dict[str, Any] = materialize_run_queue(config, specs, output_dir=output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
