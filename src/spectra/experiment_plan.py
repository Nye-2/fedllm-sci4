from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median_low
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_RUN_ARTIFACTS = [
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

DEFAULT_MAIN_DP_METHODS = [
    "fedavg_lora_dp_dplora_style",
    "fedsb_style_fixed_core_dp",
    "spectra_fedcore_dp",
]

DEFAULT_CONTEXT_METHODS = [
    "prompt_only_qwen35",
    "central_lora",
    "local_only_spectra_core",
    "fedavg_lora",
    "spectra_core_non_dp",
]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    tier: str
    table_id: str
    seed: int
    epsilon: float | None
    privacy_enabled: bool
    budget_match_type: str
    expected_runner: str
    notes: str

    def to_record(self, *, command: str | None = None) -> dict[str, Any]:
        record = asdict(self)
        if command is not None:
            record["command"] = command
        return record


def load_experiment_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = config_path.as_posix()
    return config


def build_run_specs(
    config: Mapping[str, Any],
    *,
    final_seeds: bool = False,
    include_optional: bool = False,
) -> list[RunSpec]:
    seeds = _seeds(config, final_seeds=final_seeds)
    epsilons = [float(value) for value in config["privacy"]["epsilon_sweep"]]
    target_epsilon = float(config["privacy"].get("target_epsilon", 4.0))
    matrix = config.get("experiment_matrix", {})
    run_groups = config.get("run_groups", {})

    main_dp_methods = list(run_groups.get("p0_dp_main", DEFAULT_MAIN_DP_METHODS))
    context_methods = list(run_groups.get("p0_context", DEFAULT_CONTEXT_METHODS))
    classical_methods = list(matrix.get("classical_baselines", []))
    ablation_methods = list(matrix.get("p1_ablation_methods", []))

    specs: list[RunSpec] = []
    for method in main_dp_methods:
        for epsilon in epsilons:
            for seed in seeds:
                specs.append(
                    _make_spec(
                        method=method,
                        tier="P0",
                        table_id="main_dp_fl",
                        seed=seed,
                        epsilon=epsilon,
                        privacy_enabled=True,
                        budget_match_type="privacy_and_schedule_matched",
                        expected_runner="qwen_peft_federated",
                        notes="Primary client-level DP-FL comparison.",
                    )
                )

    for method in context_methods:
        for seed in seeds:
            specs.append(
                _make_spec(
                    method=method,
                    tier="P0",
                    table_id="context_non_dp",
                    seed=seed,
                    epsilon=None,
                    privacy_enabled=False,
                    budget_match_type="capacity_reported",
                    expected_runner="qwen_peft_or_prompt",
                    notes="Non-DP central/local/federated context.",
                )
            )

    for method in classical_methods:
        for seed in seeds:
            specs.append(
                _make_spec(
                    method=method,
                    tier="P0",
                    table_id="classical_ids_context",
                    seed=seed,
                    epsilon=None,
                    privacy_enabled=False,
                    budget_match_type="same_split_tabular_context",
                    expected_runner="classical_tabular",
                    notes="Classical IDS context, not the main SLM contribution.",
                )
            )

    for method in ablation_methods:
        for seed in seeds:
            specs.append(
                _make_spec(
                    method=method,
                    tier="P1",
                    table_id="spectra_ablation",
                    seed=seed,
                    epsilon=target_epsilon,
                    privacy_enabled=True,
                    budget_match_type="ablation_fixed_epsilon",
                    expected_runner="qwen_peft_federated",
                    notes="Utility-recovery ablation at the primary epsilon.",
                )
            )

    if include_optional:
        for method in matrix.get("p2_optional", []):
            for seed in seeds:
                specs.append(
                    _make_spec(
                        method=str(method),
                        tier="P2",
                        table_id="optional_sensitivity",
                        seed=seed,
                        epsilon=target_epsilon if "epsilon" not in str(method).lower() else 1.0,
                        privacy_enabled="epsilon" in str(method).lower(),
                        budget_match_type="optional_sensitivity",
                        expected_runner="protocol_extension",
                        notes="Run only after P0/P1 tables are stable.",
                    )
                )

    return specs


def derive_rare_labels(split: Mapping[str, Any], *, policy: str) -> list[str]:
    if policy != "below_median_attack_count":
        raise ValueError(f"unsupported rare label policy: {policy}")
    counts_by_label = split.get("split_label_counts", {})
    attack_counts = {
        label: int(counts.get("train", 0))
        for label, counts in counts_by_label.items()
        if label != "Normal"
    }
    if not attack_counts:
        return []
    threshold = median_low(sorted(attack_counts.values()))
    return sorted(label for label, count in attack_counts.items() if count <= threshold)


def validate_experiment_inputs(
    config: Mapping[str, Any],
    *,
    repo_root: Path | str,
    strict_raw_data: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    checks = [
        _path_check("config", config.get("_config_path"), root=root, required=True),
        _path_check("paper_protocol", config.get("paper_protocol"), root=root, required=True),
        _path_check("raw_edgeiiot_csv", config["data"]["csv_path"], root=root, required=strict_raw_data),
        _path_check(
            "selected_split_artifact",
            "data/processed/edgeiiot/selected_ml_stratified_split_seed20260531.json",
            root=root,
            required=True,
        ),
        _path_check(
            "client_partition_artifact",
            "data/processed/edgeiiot/selected_ml_clients_seed20260531_K10_alpha0.5.json",
            root=root,
            required=True,
        ),
        _path_check("snli_manifest", "data/processed/snli/manifest.json", root=root, required=True),
        _path_check("fedsb_fed_wrapper", "scripts/run_fedsb_snli_fed_private.sh", root=root, required=True),
        _path_check("fedsb_central_wrapper", "scripts/run_fedsb_snli_central_private.sh", root=root, required=True),
    ]
    return {
        "ok": all(check["exists"] or not check["required"] for check in checks),
        "strict_raw_data": strict_raw_data,
        "checks": checks,
    }


def materialize_run_queue(
    config: Mapping[str, Any],
    specs: Sequence[RunSpec],
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    out = Path(output_dir)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out / "run_queue.jsonl"

    records = []
    for spec in specs:
        run_dir = runs_dir / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config_path = run_dir / "config_resolved.json"
        command = f"python3 scripts/run_single_experiment.py --run-config {run_config_path.as_posix()} --prepare-only"
        record = spec.to_record(command=command)
        records.append(record)
        _write_json(
            run_config_path,
            {
                "run": record,
                "base_config": _strip_private_keys(config),
                "artifacts": {"required_files": REQUIRED_RUN_ARTIFACTS},
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "run_id": spec.run_id,
                "status": "pending",
                "next_command": command,
                "required_artifacts": REQUIRED_RUN_ARTIFACTS,
            },
        )

    with queue_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    _write_runbook(out / "RUNBOOK.md", records)
    return {"output_dir": out.as_posix(), "run_count": len(records), "queue_path": queue_path.as_posix()}


def load_split_artifact(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_validation_report(path: Path | str, report: Mapping[str, Any]) -> None:
    _write_json(Path(path), dict(report))


def _make_spec(
    *,
    method: str,
    tier: str,
    table_id: str,
    seed: int,
    epsilon: float | None,
    privacy_enabled: bool,
    budget_match_type: str,
    expected_runner: str,
    notes: str,
) -> RunSpec:
    suffix = f"__eps{_format_epsilon(epsilon)}" if epsilon is not None else ""
    run_id = f"{_slug(table_id)}__{_slug(method)}__seed{seed}{suffix}"
    return RunSpec(
        run_id=run_id,
        method=method,
        tier=tier,
        table_id=table_id,
        seed=int(seed),
        epsilon=epsilon,
        privacy_enabled=privacy_enabled,
        budget_match_type=budget_match_type,
        expected_runner=expected_runner,
        notes=notes,
    )


def _seeds(config: Mapping[str, Any], *, final_seeds: bool) -> list[int]:
    if final_seeds:
        return [int(seed) for seed in config.get("fl", {}).get("final_table_seeds", [config["seed"]])]
    return [int(config["seed"])]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "run"


def _format_epsilon(value: float | None) -> str:
    if value is None:
        return "none"
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _path_check(name: str, path_value: object, *, root: Path, required: bool) -> dict[str, Any]:
    if not path_value:
        return {"name": name, "path": None, "exists": False, "required": required}
    path = Path(str(path_value))
    resolved = path if path.is_absolute() else root / path
    return {
        "name": name,
        "path": path.as_posix(),
        "resolved": resolved.as_posix(),
        "exists": resolved.exists(),
        "required": required,
    }


def _strip_private_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runbook(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        "# Materialized Experiment Runbook",
        "",
        "Run commands in queue order unless a gate fails. Each run directory contains `config_resolved.json` and `status.json`.",
        "",
        "```bash",
    ]
    for record in records:
        lines.append(str(record["command"]))
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
