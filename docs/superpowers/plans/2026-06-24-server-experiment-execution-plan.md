# Server Experiment Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a fresh server-side agent validate the repository, materialize the exact experiment queue, prepare per-run directories, and then attach GPU runners without guessing the paper protocol.

**Architecture:** The paper protocol remains in `docs/superpowers/plans/2026-06-23-qwen35-final-experiment-plan.md`. The machine-readable config in `configs/experiments/edgeiiot_fl_spectra_dp.json` is converted by `scripts/run_experiment_plan.py` into `run_queue.jsonl` and one `config_resolved.json` per run. `scripts/run_single_experiment.py --prepare-only` writes a `runner_request.json` contract for the runner that will actually train/evaluate a method.

**Tech Stack:** Python 3.10+, standard library for orchestration, existing `src/fedllm_data` and `src/spectra` utilities, optional server packages from `pip install -e ".[exp]"` for GPU experiments.

---

## Task 1: Validate Repository and Data

**Files:**
- Read: `README.md`
- Read: `HANDOFF.md`
- Read: `configs/experiments/edgeiiot_fl_spectra_dp.json`
- Run: `scripts/run_experiment_plan.py`

- [ ] **Step 1: Install base package**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

- [ ] **Step 2: Validate code and processed artifacts**

```bash
python3 -m compileall src scripts paper/scripts
python3 -m pytest -q
```

Expected: all tests pass; skipped tests are acceptable only when they are explicitly environment-gated.

- [ ] **Step 3: Validate experiment inputs**

```bash
python3 scripts/run_experiment_plan.py validate \
  --config configs/experiments/edgeiiot_fl_spectra_dp.json \
  --output outputs/edgeiiot_fl_spectra_dp/gate0_validation.json
```

Expected: JSON report with `ok: true`. If raw data is available on the server, rerun with `--strict-raw-data`.

## Task 2: Materialize the Run Queue

**Files:**
- Read: `configs/experiments/edgeiiot_fl_spectra_dp.json`
- Create: `outputs/edgeiiot_fl_spectra_dp/run_queue/run_queue.jsonl`
- Create: `outputs/edgeiiot_fl_spectra_dp/run_queue/RUNBOOK.md`
- Create: `outputs/edgeiiot_fl_spectra_dp/run_queue/runs/<run_id>/config_resolved.json`

- [ ] **Step 1: Materialize one-seed queue for gate execution**

```bash
python3 scripts/run_experiment_plan.py materialize \
  --config configs/experiments/edgeiiot_fl_spectra_dp.json \
  --output-dir outputs/edgeiiot_fl_spectra_dp/run_queue
```

Expected: queue summary with a positive `run_count`.

- [ ] **Step 2: Prepare runner requests**

```bash
while read -r line; do
  python3 - <<'PY' "$line"
import json, subprocess, sys
record = json.loads(sys.argv[1])
subprocess.run(record["command"].split(), check=True)
PY
done < outputs/edgeiiot_fl_spectra_dp/run_queue/run_queue.jsonl
```

Expected: every run directory has `runner_request.json` and `status.json` with `ready_for_runner`.

## Task 3: Attach Actual Runners

**Files:**
- Read: `outputs/edgeiiot_fl_spectra_dp/run_queue/runs/<run_id>/runner_request.json`
- Create per run: all files listed in `runner_request.json.runner_contract.required_outputs`

- [ ] **Step 1: Implement or dispatch runner `qwen_peft_federated`**

Use for:

```text
fedavg_lora_dp_dplora_style
fedsb_style_fixed_core_dp
spectra_fedcore_dp
A0-A5 ablations
```

Required behavior: load Qwen3.5-2B text-only, use the frozen split/client artifacts, train only the declared adapter path, apply the declared DP upload policy when `privacy_enabled` is true, and write real metrics only after evaluation completes.

- [ ] **Step 2: Implement or dispatch runner `qwen_peft_or_prompt`**

Use for:

```text
prompt_only_qwen35
central_lora
local_only_spectra_core
fedavg_lora
spectra_core_non_dp
```

Required behavior: use the same prompt renderer and split artifacts as the main DP runs.

- [ ] **Step 3: Implement or dispatch runner `classical_tabular`**

Use for:

```text
random_forest
xgboost_or_lightgbm
mlp
```

Required behavior: use numeric/tabular features, exclude the same identifier-like columns, fit preprocessing on train only, and report context metrics without treating these baselines as SLM methods.

## Task 4: Final Tables

**Files:**
- Read: per-seed run result JSONL
- Run: `paper/scripts/results_to_latex.py`

- [ ] **Step 1: Generate result table fragments**

```bash
cd paper
python scripts/results_to_latex.py ../outputs/edgeiiot_fl_spectra_dp/results_main_dp.jsonl \
  > tables/generated_main_results.tex
```

Expected: the generated table includes Macro-F1, balanced accuracy, rare recall, MB/round, total MB, and trainable parameters.

## Non-Negotiable Rules

- Do not write placeholder `metrics.json`.
- Do not report estimated numbers in the paper.
- Do not change dataset split artifacts after a run has started.
- Do not claim record-level DP; this protocol is client-level upload release DP.
- Do not treat Fed-SB-style as optional; it is the closest baseline.
