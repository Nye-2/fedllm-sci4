#!/usr/bin/env python3
"""Convert per-run JSONL results into a LaTeX table fragment.

Expected input records follow paper/notes/result_collection_contract.md.
The script intentionally fails on missing primary metrics so paper tables are not
filled from partial or ambiguous data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PRIMARY_FIELDS = [
    "macro_f1",
    "balanced_accuracy",
    "rare_attack_recall",
    "uploaded_mb_per_client_round",
    "total_uploaded_mb",
    "trainable_params",
]


def _get_nested(record: dict[str, Any], key: str) -> Any:
    current: Any = record
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def _format_mean_std(values: list[float], precision: int) -> str:
    if len(values) == 1:
        return f"{values[0]:.{precision}f}"
    return f"{mean(values):.{precision}f} $\\pm$ {stdev(values):.{precision}f}"


def _format_number(values: list[float], precision: int) -> str:
    value = mean(values)
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.{precision}f}"


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def validate_record(record: dict[str, Any]) -> None:
    required = ["method", "seed", "privacy.epsilon", *PRIMARY_FIELDS]
    missing: list[str] = []
    for key in required:
        try:
            _get_nested(record, key)
        except KeyError:
            missing.append(key)
    if missing:
        method = record.get("method", "<unknown>")
        seed = record.get("seed", "<unknown>")
        raise SystemExit(f"record method={method} seed={seed} missing fields: {missing}")


def make_table(records: list[dict[str, Any]], precision: int) -> str:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        validate_record(record)
        key = (str(record["method"]), float(_get_nested(record, "privacy.epsilon")))
        grouped[key].append(record)

    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & $\epsilon$ & Macro-F1 & Bal. Acc. & Rare Recall & MB/round & Total MB & Trainable Params \\",
        r"\midrule",
    ]
    for (method, epsilon), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        macro_f1 = [float(record["macro_f1"]) for record in group]
        balanced_accuracy = [float(record["balanced_accuracy"]) for record in group]
        rare_recall = [float(record["rare_attack_recall"]) for record in group]
        uploaded_mb = [float(record["uploaded_mb_per_client_round"]) for record in group]
        total_uploaded_mb = [float(record["total_uploaded_mb"]) for record in group]
        params = [float(record["trainable_params"]) for record in group]
        lines.append(
            " & ".join(
                [
                    method,
                    f"{epsilon:g}",
                    _format_mean_std(macro_f1, precision),
                    _format_mean_std(balanced_accuracy, precision),
                    _format_mean_std(rare_recall, precision),
                    _format_number(uploaded_mb, 3),
                    _format_number(total_uploaded_mb, 3),
                    _format_number(params, 0),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Per-run JSONL file.")
    parser.add_argument("--precision", type=int, default=3, help="Metric precision.")
    args = parser.parse_args()

    records = load_records(args.jsonl)
    if not records:
        raise SystemExit(f"{args.jsonl} contains no records")
    print(make_table(records, args.precision))


if __name__ == "__main__":
    main()
