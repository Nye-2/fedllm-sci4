# SPECTRA-FedCore LaTeX Draft

This directory contains a template-neutral SCI manuscript scaffold for SPECTRA-FedCore.

## Build

Use a standard TeX distribution with `latexmk`, `pdflatex`, and BibTeX:

```bash
cd paper
latexmk -pdf main.tex
```

or:

```bash
cd paper
make
```

Clean generated files:

```bash
make clean
```

The local machine used to create this scaffold did not have TeX installed, so the first compile should be done on a TeX-enabled workstation or server.

## Drafting Rules

- Do not fill result tables with estimates.
- Keep all result claims tied to completed server runs.
- Main backbone is fixed to `Qwen/Qwen3.5-2B`; all main experiments use text-only telemetry prompts.
- Keep the collision boundary with Fed-SB explicit: square-core training and exact core averaging are not claimed as novel.
- Main novelty should stay focused on public-backbone spectral bases, budget-aware rank/noise allocation, client-side DP utility recovery, and Edge-IIoT reproducibility.
- After the target journal is chosen, migrate `main.tex` from `article` to the journal template and keep the section files unchanged where possible.

## Expected Result Inputs

The results section expects run-level artifacts with:

- method name
- seed
- dataset protocol id
- table id
- split id
- client partition id
- baseline source
- budget-match type
- backbone checkpoint and revision
- model source and input modality
- experiment tier (`P0`, `P1`, or `P2`)
- adapter target modules
- trainable parameter count
- uploaded MB per round
- total uploaded MB
- privacy ledger path
- preprocessing fingerprint path
- macro-F1
- balanced accuracy
- per-class recall
- rare attack recall
- peak VRAM
- wall-clock time
- inference latency

## Result Table Helper

After server runs export per-seed JSONL records, generate a LaTeX table fragment with:

```bash
cd paper
python3 scripts/results_to_latex.py path/to/results.jsonl > tables/generated_main_results.tex
```

The expected JSONL schema is documented in `notes/result_collection_contract.md`.
