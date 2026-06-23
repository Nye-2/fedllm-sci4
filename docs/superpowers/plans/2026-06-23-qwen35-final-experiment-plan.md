# SPECTRA-FedCore Final Experiment Plan

Date: 2026-06-23
Status: execution-ready protocol for Qwen3.5-2B experiments

## 1. Fixed Main Decision

The paper uses one primary backbone:

```text
Qwen/Qwen3.5-2B
```

Model source:

```text
ModelScope: https://www.modelscope.ai/models/Qwen/Qwen3.5-2B
Hugging Face mirror: https://huggingface.co/Qwen/Qwen3.5-2B
```

Use only the text/language pathway. Edge-IIoT telemetry is rendered as compact text prompts. No image input, vision encoder signal, or multimodal supervision is used in the main benchmark.

Why this choice:

- 2B parameters is small enough to support the edge-intelligence story.
- The model is recent and recognizable.
- The model is large enough to avoid looking like a toy edge model.
- A single fixed backbone keeps the method comparison clean.

Do not run a model-zoo comparison unless the main paper is already complete.

## 2. Research Questions

RQ1: Under client-side release DP, does SPECTRA-FedCore improve the privacy-utility-communication trade-off over FedAvg-LoRA and a Fed-SB-style fixed-core baseline?

RQ2: Under the same Qwen3.5-2B backbone and uploaded-coordinate budget, which utility-recovery component matters most: public SVD basis, spectral rank allocation, layer-adaptive DP noise, local residual, or shrinkage?

RQ3: Are the conclusions stable under the mainstream Edge-IIoTset closed-set protocol, duplicate diagnostics, and a raw-file robustness check?

The main paper should not claim that LLMs beat all tabular IDS models. Classical IDS baselines calibrate the task; the main claim is about private federated edge assistant adaptation.

## 3. Data Protocol

Primary dataset:

```text
data/raw/edgeiiotset/full_dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv
```

Primary task:

```text
15-class closed-set intrusion classification
```

Main split:

```text
80/10/10 stratified row-level split
seed = 20260531
artifact = data/processed/edgeiiot/selected_ml_stratified_split_seed20260531.json
```

Current split sizes:

```text
train = 126,233
val   = 15,774
test  = 15,793
```

FL partition:

```text
K = 10 clients
artifact = data/processed/edgeiiot/selected_ml_clients_seed20260531_K10_alpha0.5.json
primary non-IID = Dirichlet label-skew, alpha = 0.5
```

Evaluation rules:

- Keep validation and test sets global and fixed across all methods.
- Partition only the training split into clients.
- Fit scalers, quantile bins, feature filters, balancing, and prompt bins only on training data.
- Do not use validation or test labels to tune prompt fields, DP parameters, clipping radii, or rank budgets.
- Report duplicate or feature-hash overlap after removing excluded identifier-like fields.

## 4. Prompt Protocol

Use the existing compact prompt renderer as the main prompt family:

```text
prompt_version = compact_v2
target = exact 15-class label string
max prompt target = <= 512 tokens whenever practical
```

Exclude high-leakage fields from the main prompt:

```text
frame.time
ip.src_host
ip.dst_host
http.file_data
http.request.uri.query
http.referer
http.request.full_uri
Attack_label
Attack_type
```

Report:

- mean, p50, p95 prompt token length
- exact label extraction failure rate
- any labels dropped or normalized

## 5. FL and DP Settings

Main FL setting:

```text
single-server simulation
clients = 10
rounds = 50
local_epochs = 1
participation = 1.0 for main reproducibility
optimizer = AdamW unless implementation dictates otherwise
aggregation = sample-count weighted FedAvg over uploaded core coordinates
seeds = 3 for final tables
```

Debug setting:

```text
rounds = 3 or 5
one seed
one epsilon
no ablations
goal = catch shape, tokenizer, adapter, and accountant errors
```

Main DP setting:

```text
privacy unit = client-level upload
adjacency = replace-one
delta = 1e-5
epsilon sweep = {8, 4, 2}
epsilon = 1 only if epsilon=2 remains usable and compute allows
accountant = conservative RDP without sampling amplification
```

Report every run with:

- epsilon and delta
- RDP order grid
- clipping radius or per-layer radii
- noise scale or per-layer scales
- clipping rate by round and by layer
- participation schedule

Do not claim record-level DP.

## 6. Method Variants

P0 minimum method set:

```text
Prompt-only Qwen3.5-2B
Central LoRA
Local-only SPECTRA-Core
FedAvg-LoRA
Fed-SB-style fixed-core
SPECTRA-Core non-DP
SPECTRA-FedCore DP
```

P1 ablation set:

```text
A0 Random orthogonal basis + uniform rank + global DP
A1 Public SVD basis + uniform rank + global DP
A2 Public SVD basis + spectral rank + global DP
A3 Public SVD basis + spectral rank + layer-adaptive DP
A4 A3 + local residual
A5 A4 + shrinkage post-processing
```

P2 optional extensions:

```text
K = 20 sensitivity
participation = 0.5 sensitivity
epsilon = 1
source/protocol skew if metadata is stable
cross-backbone check if reviewers ask
```

## 7. Baseline Rules

Classical IDS context:

```text
Random Forest
XGBoost or LightGBM
MLP
```

Classical baselines use numeric/tabular features, not prompts. They answer: how strong is the tabular task itself?

Adapter baselines:

```text
FedAvg-LoRA
Fed-SB-style fixed-core
SPECTRA-FedCore
```

Fed-SB handling:

1. Reproduce official Fed-SB SNLI private FL first.
2. Try faithful adaptation to Qwen3.5-2B and Edge-IIoT prompts.
3. If direct adaptation is brittle, run a clearly labeled `Fed-SB-style fixed-core` baseline in our code path under matched budgets.

Fairness constraints:

- same Qwen3.5-2B backbone
- same train/val/test split
- same client partition
- same rounds and local epochs
- same epsilon/delta and accountant convention for DP comparisons
- report same uploaded MB/client/round and trainable parameter count

## 8. Result Tables

Table 1: Main DP-FL result.

```text
Rows: FedAvg-LoRA-DP, Fed-SB-style-DP, SPECTRA-FedCore-DP
Columns: epsilon, macro-F1, balanced accuracy, rare recall, uploaded MB/round, trainable params
Settings: K=10 Dirichlet alpha=0.5, Qwen3.5-2B
```

Table 2: Non-DP and central/local context.

```text
Rows: Prompt-only, Central LoRA, Local-only SPECTRA, FedAvg-LoRA non-DP, SPECTRA non-DP
Columns: macro-F1, balanced accuracy, rare recall, wall-clock, peak VRAM
```

Table 3: Classical IDS context.

```text
Rows: Random Forest, XGBoost/LightGBM, MLP
Columns: macro-F1, balanced accuracy, rare recall, inference latency
```

Table 4: SPECTRA ablation.

```text
Rows: A0-A5
Columns: macro-F1, rare recall, clipping rate, uploaded MB/round
Primary epsilon: 4
```

Table 5: Diagnostics and robustness.

```text
Rows: main split, deduplicated sensitivity if needed, raw per-file 8/1/1 robustness
Columns: duplicate rate, macro-F1 shift, method ranking stable yes/no
```

Table 6: Edge cost.

```text
Rows: main adapter methods
Columns: trainable params, uploaded MB/client/round, total MB, peak VRAM, latency, prompt length p95
```

## 9. Figures

Figure 1: System architecture.

```text
public Qwen3.5-2B backbone -> public spectral bases -> client local core training -> client-side DP release -> server aggregation -> global core broadcast
```

Figure 2: Privacy-utility curve.

```text
x = epsilon
y = macro-F1
methods = FedAvg-LoRA-DP, Fed-SB-style-DP, SPECTRA-FedCore-DP
```

Figure 3: Communication Pareto.

```text
x = uploaded MB/client/round
y = macro-F1 or rare recall
markers = method and epsilon
```

Figure 4: Ablation waterfall.

```text
A0 -> A1 -> A2 -> A3 -> A4 -> A5
```

## 10. Run Order

Run experiments in this order:

1. Environment and data verification.
2. Fed-SB official SNLI private FL reproduction.
3. Qwen3.5-2B load and target-module inspection.
4. Prompt-only and tokenizer smoke test.
5. One local non-DP SPECTRA-Core run.
6. One 3-round FL non-DP smoke run.
7. One DP accountant smoke run at epsilon 4.
8. Full non-DP central/local/federated context.
9. Full DP-FL main table at epsilon 8, 4, 2.
10. SPECTRA ablations at epsilon 4.
11. Classical IDS context.
12. Diagnostics and robustness.
13. Final three-seed reruns for paper tables.

## 11. Per-Run Artifact Contract

Every run must write:

```text
config_resolved.json
metrics.json
metrics_by_class.json
privacy_ledger.json
communication.json
hardware_profile.json
client_label_histograms.json
run.log
```

Every row in final tables must be traceable to run IDs and seeds. No hand-entered estimates in the paper.
