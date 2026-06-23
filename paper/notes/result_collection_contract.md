# Result Collection Contract

Every server run should export a JSON or JSONL record with these fields so tables can be filled reproducibly:

- `run_id`
- `method`
- `seed`
- `dataset`
- `split_artifact`
- `client_partition_artifact`
- `backbone`
- `model_source`
- `modelscope_id`
- `backbone_revision`
- `input_modality`
- `experiment_tier`
- `target_modules`
- `rank_allocation`
- `privacy.enabled`
- `privacy.epsilon`
- `privacy.delta`
- `privacy.accountant`
- `privacy.rdp_orders`
- `privacy.clipping`
- `privacy.noise_allocation`
- `privacy.clipping_rate_by_round`
- `trainable_params`
- `uploaded_mb_per_client_round`
- `total_uploaded_mb`
- `macro_f1`
- `balanced_accuracy`
- `per_class_recall`
- `rare_attack_recall`
- `confusion_matrix_path`
- `privacy_ledger_path`
- `peak_vram_gb`
- `wall_clock_minutes`
- `inference_latency_ms`
- `prompt_token_length_mean`
- `prompt_token_length_p95`
- `label_extraction_failure_rate`

Do not aggregate away per-seed records. Paper tables should be generated from per-seed run files.
