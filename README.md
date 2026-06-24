# SPECTRA-FedCore

SPECTRA-FedCore is a research codebase for a public-spectral, client-level differentially private federated core-adaptation method for Edge-IIoT intrusion detection.

The repository is currently structured to make the server handoff reproducible before large GPU experiments start:

- `src/fedllm_data/`: dataset manifests, label normalization, selected-CSV split generation, prompt smoke samples, and client partitions.
- `src/spectra/`: CPU-testable NumPy core utilities for public SVD bases, spectral core adapters, client-side Gaussian release accounting, FL aggregation, and metrics.
- `scripts/prepare_datasets.py`: regenerates portable dataset artifacts.
- `scripts/run_experiment_plan.py`: validates server inputs and materializes the exact run queue from the paper protocol.
- `scripts/run_single_experiment.py`: prepares one materialized run directory and writes the runner contract for the server-side trainer.
- `HANDOFF.md`: server execution guide.
- `docs/superpowers/specs/2026-05-31-spectra-dp-fedcore-design.md`: paper-level design.
- `docs/superpowers/plans/2026-06-23-qwen35-final-experiment-plan.md`: final Qwen3.5-2B experiment protocol.
- `docs/superpowers/plans/2026-06-24-server-experiment-execution-plan.md`: server-side execution plan for another agent.

## Quick Check

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python3 -m compileall src scripts
python3 -m pytest -q
```

## Server Experiment Queue

After data is in place, a server-side agent should start with:

```bash
python3 scripts/run_experiment_plan.py validate \
  --config configs/experiments/edgeiiot_fl_spectra_dp.json \
  --output outputs/edgeiiot_fl_spectra_dp/gate0_validation.json

python3 scripts/run_experiment_plan.py materialize \
  --config configs/experiments/edgeiiot_fl_spectra_dp.json \
  --output-dir outputs/edgeiiot_fl_spectra_dp/run_queue
```

This writes:

- `outputs/edgeiiot_fl_spectra_dp/run_queue/run_queue.jsonl`
- `outputs/edgeiiot_fl_spectra_dp/run_queue/RUNBOOK.md`
- `outputs/edgeiiot_fl_spectra_dp/run_queue/runs/<run_id>/config_resolved.json`

Each queue command runs `scripts/run_single_experiment.py --prepare-only`, which creates `runner_request.json`. The GPU runner must then write the required real artifacts; it must not write placeholder `metrics.json`.

## Data Preparation

Raw datasets are intentionally not committed. After preparing `data/raw/edgeiiotset/full_dataset` and `data/raw/snli/current`, regenerate artifacts with:

```bash
PYTHONPATH=src python scripts/prepare_datasets.py \
  --edge-root data/raw/edgeiiotset/full_dataset \
  --snli-root data/raw/snli/current \
  --out-dir data/processed \
  --count-rows \
  --relative-paths
```

Important generated artifacts:

- `data/processed/edgeiiot/file_manifest.json`
- `data/processed/edgeiiot/label_inventory.json`
- `data/processed/edgeiiot/selected_ml_stratified_split_seed20260531.json`
- `data/processed/edgeiiot/selected_ml_clients_seed20260531_K10_alpha0.5.json`
- `data/processed/snli/manifest.json`

## Experiment Protocol

Main paper protocol:

- Backbone: `Qwen/Qwen3.5-2B` from ModelScope, text-only telemetry prompts
- Dataset: `ML-EdgeIIoT-dataset.csv`
- Task: normalized 15-class closed-set intrusion classification
- Split: 80/10/10 stratified row-level split, fixed seed `20260531`
- FL simulation: one server process, `K=10` clients, IID sanity plus Dirichlet label-skew `alpha=0.5`
- Privacy: client-level release DP, not record-level DP; main epsilon sweep `{8, 4, 2}` with epsilon `1` optional

Minimum paper comparison:

- classical IDS context: Random Forest, XGBoost/LightGBM, MLP
- adapter baselines: Prompt-only Qwen3.5-2B, Central LoRA, FedAvg-LoRA, FedAvg-LoRA-DP/DP-LoRA-style, Fed-SB-style fixed-core
- method variants: SPECTRA-Core non-DP, SPECTRA-FedCore DP, and the A0-A5 utility-recovery ablation
- optional external validation: WUSTL-IIOT-2021 only after the main Edge-IIoTset P0/P1 tables are stable

The method novelty should be framed around public-backbone spectral bases, budget-aware rank/noise allocation, and client-side DP utility recovery. Do not claim that training a square core alone is novel; Fed-SB is the closest baseline and must be evaluated seriously. See the final protocol before launching GPU runs.
