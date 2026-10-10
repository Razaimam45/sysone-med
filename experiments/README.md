# Medical SysOne experiment exports

The runner is evaluation-only: it loads existing checkpoints, performs inference on the saved full evaluation splits, and emits CSV/PDF/PNG assets under `experiment_results/`. It does not train or update model weights.

```bash
bash experiments/run_saved_experiments.sh
```

It evaluates constrained one-token MedGemma generation on the shared manifest, reruns Laya-Med and MedGemma-LoRA on test/calibration and correct/missing/shuffled image conditions, evaluates the frozen-head checkpoint, audits candidate/probability rows, computes paired comparisons, recomputes routing/conformal policies from matched calibration/test predictions, and runs a candidate-count latency profile. Set `LAYA_CHECKPOINT`, `HEAD_CHECKPOINT`, `LORA_CHECKPOINT`, `MANIFEST`, or `OUT_DIR` to override defaults. `RUN_CONSTRAINED=0` skips constrained generation only when its saved prediction ledger already exists; `RUN_SCALING=0` skips the additional Laya-Med timing run.

## Outputs

- `candidate_audit.csv`, `candidate_rows.csv`: manifest alignment, ordered choices, target/prediction index, and probability validity.
- `main_results.csv`: per-model/task accuracy, ECE, NLL, Brier score, p50/p95 latency, and bootstrap intervals.
- `ll_vs_head_paired.csv`, `head_vs_lora_paired.csv`: paired accuracy/NLL/Brier differences with bootstrap intervals; examples must align exactly.
- `visual_dependence/visual_dependence.csv`, `.png`, `.pdf`: correct-image, missing-image, and shuffled-image accuracy with paired intervals.
- `posthoc/posthoc_policy_results.csv`, `risk_coverage_points.csv`, `risk_coverage.png`, `.pdf`: routing and split-conformal results recomputed from matched calibration/test logits.
- `observed_latency/latency_by_candidate_count.csv`, `.png`, `.pdf`: observed per-example timings grouped by task and number of options.
- `candidate_scaling/candidate_scaling.csv`, `.png`, `.pdf`: controlled candidate-count timing and Laya-Med text/image encoder component timings.

For just the analysis layer, run commands from the repository root:

```bash
python analysis/run_experiments.py summary --predictions NAME=path/to/predictions.jsonl ...
python analysis/run_experiments.py compare --reference NAME=reference.jsonl --candidate NAME=candidate.jsonl
python analysis/run_experiments.py audit --manifest-test data/prepared/test.jsonl --predictions NAME=predictions.jsonl
```

Prediction JSONL records must contain `id`, `dataset`, `label`, and either normalized `probabilities` or raw `logits`. New evaluator outputs include the checkpoint temperature, ordered candidate options, and per-example latency. Existing probability vectors are used as written. Calibration and test predictions must be generated from the same checkpoint and manifest; the post-hoc command rejects overlapping IDs and never uses test labels to fit conformal thresholds.

## Scope notes

The runner compares existing trained formulations; it does not train new model checkpoints. The adapted-backbone native-likelihood condition is not currently implemented because it requires a specified candidate prompt/likelihood protocol for the saved LoRA checkpoint. Once that condition emits a JSONL ledger, include it in `summary` and `compare` without changing the analysis code. Visual perturbation is implemented for Laya-Med and MedGemma-LoRA; the frozen-head checkpoint uses cached image features and is not rerun under image perturbations. The component profiler reports encoder and remaining forward time, not GPU kernel-level FLOPs. Preserve the saved run settings and checkpoint/manifest hashes alongside any exported paper table.
