# Multimodal medical decision training

A Laya/Jev-inspired non-autoregressive decision framework: PubMedBERT encodes
question/choice inputs, BiomedCLIP supplies visual features, and a fusion MLP
scores supplied candidate answers. This is a supervised option-scoring baseline,
not a fine-tuned Laya checkpoint, report generator, or claim of research novelty.

[Research positioning and related work](../RELATED_WORK.md) records the
Q2ATransformer, MedProb and calibrated surgical VQA overlaps, and distinguishes
implemented capabilities from untested generalization and robustness claims.

## Environment and assets

```bash
python -m pip install -e '.[train,dev]'
python training/download_assets.py --all-medical
```

The cached assets and normalized datasets in this checkout are already present.
Default paths are relative to the repository, independent of the working directory.
Models are exposed as `data/models/text` and `data/models/vision`; their exact
snapshot directories and file hashes are saved with every training run. The
BiomedCLIP loader uses the local visual weights without constructing its text tower.

See [data layout](../data/README.md) for source, cache, and normalized locations.
PubMedQA and VQA-RAD source data are retained separately under `data/sources/`.

## Prepare and audit data

```bash
python training/prepare_data.py
```

This checkout already has an audited `data/prepared/manifest.json`. Preparation
refuses to overwrite an existing prepared directory. For a new dataset version:

```bash
python training/prepare_data.py --output-dir data/prepared-v2
```

All training, calibration, evaluation, and baseline runs consume that same
manifest. It contains exact split IDs, file hashes, source hashes, tokenizer
identity, and the token budget. Prepared rows contain decoded-image hashes.

Preparation:

- rejects legacy MedMCQA inputs containing answer explanations; the updated
  normalizer emits question/choice-only inputs; SLAKE organ/location annotations
  are also excluded;
- groups exact text questions, decoded image contents, supplied patient/article/
  group IDs, and PMC article identifiers when present in image filenames;
- preserves existing test partitions and quarantines lower-priority records
  overlapping protected source groups;
- divides existing validation groups into selection and calibration sets;
- creates separate validation/calibration/test groups from training sources when
  those partitions are absent;
- quarantines corrupt images, invalid/duplicate choices, conflicting labels,
  duplicate inputs, and questions whose full choices cannot fit the token budget.

Read `data/prepared/audit.json` before interpreting results. Quarantining test
records changes the evaluated benchmark subset: report the retained counts and
exclusions. The current PMC-VQA and MedMCQA test partitions are internal holdouts
from downloaded training data, not their official benchmark tests. VQA-RAD uses
a new shared group split; historical trial results use different partitions.

This prevents the checked overlaps; it does not establish independence of
unidentified patients, near-duplicate images, or pretrained model corpora.
Supply patient/article IDs where available and audit dataset provenance.

## Train

```bash
python training/train.py \
  --manifest data/prepared/manifest.json \
  --batch-size 8 --grad-accumulation 4 --epochs 3 \
  --output-dir models/laya-med
```

Defaults: LoRA rank 8, frozen base text/vision encoders, 512-token inputs,
equal-dataset sampling with replacement, CUDA FP16 training when available,
and checkpoint selection by macro average dataset accuracy. `--sampling
proportional` instead samples the natural dataset mixture. `--train-vision`
enables visual encoder training (LoRA when `--lora-rank > 0`, full fine-tuning otherwise).

To adapt both biomedical encoders with LoRA in a new run:

```bash
python training/train.py \
  --manifest data/prepared/manifest.json \
  --train-vision --lora-rank 8 \
  --batch-size 8 --grad-accumulation 4 --epochs 15 \
  --output-dir models/laya-med-vision-lora
```

Visual LoRA targets attention QKV and output projections; pretrained visual
weights remain frozen. Startup and `run.json` report `vision_trainable` so
this is directly verifiable (442,368 for the local BiomedCLIP at rank 8).
Evaluation reconstructs the saved adapter architecture before strict checkpoint
loading. With `--train-vision --lora-rank 0`, the visual encoder is fully
fine-tuned, but the text encoder is frozen because the same rank also controls
text LoRA. Only batches containing images update visual parameters.

Do not add `--train-vision` when resuming a frozen-vision run: strict resume
checks correctly reject that optimizer/architecture change. The command above
starts a new run from pretrained encoder assets and preserves the existing run.


Question, full option set, and candidate text are always preserved; only context
is truncated. Image tensors contain one image per visual example. Candidate
indices reuse its embedding; absent images always receive zero visual features.

`--no-images` trains a text-only baseline on the same cases and splits. It does
not read images in the collator, though startup still verifies the shared data
manifest and image integrity for fair comparisons.

## MedGemma-Direct baseline

MedGemma-Direct uses the `google/medgemma-4b-it` multimodal checkpoint with its
vision tower and pretrained backbone frozen, LoRA adapters on the language
attention projections, and a learned candidate-scoring head on final fused
hidden states. It does not call `generate()`: each supplied answer is scored and
the scores are normalized over that question's variable-size choice set. It
uses the same manifest rows, group splits, context/choice fields, and calibration
partition as Laya-Med; MedGemma's own tokenizer and image processor are used for
its backbone-specific input encoding.

```bash
python training/train_medgemma_direct.py \
  --manifest data/prepared/manifest.json \
  --medgemma-model google/medgemma-4b-it \
  --output-dir models/medgemma-direct \
  --epochs 3 --batch-size 1 --grad-accumulation 8 \
  --max-length 512 --lora-rank 8 --device cuda

python training/evaluate_medgemma_direct.py \
  --checkpoint models/medgemma-direct/best.pt \
  --manifest data/prepared/manifest.json \
  --batch-size 1 --device cuda \
  --append-results results
```

The smaller default batch reflects candidate expansion over a 4B multimodal
backbone. Checkpoints store only trainable LoRA/head tensors plus optimizer and
run state; the base MedGemma snapshot is verified by file fingerprint when
resuming or evaluating. `MedGemma-Generate` remains the existing inference-only
`conventional_llm` baseline, while MedGemma-Direct is trained on the shared
training split; this is a backbone-controlled comparison, not a training-budget
matched generation-versus-decision comparison.

The last incomplete accumulation window is normalized by its actual number of
examples. The scheduler advances only after successful AMP updates. Validation
is sample-correct and reports per-dataset metrics. Temperature calibration uses
separate calibration data and supports variable numbers of choices.

Artifacts: `best.pt`, `last.pt`, `run.json`, `metadata.json`, and
`calibration.json`. Checkpoints contain model/optimizer/scheduler/scaler state,
RNG and sampler state, epoch, history, assets, and the manifest fingerprint.
Resume at the last completed epoch with the same arguments and output directory:

```bash
python training/train.py --manifest data/prepared/manifest.json \
  --batch-size 8 --grad-accumulation 4 --epochs 3 \
  --output-dir models/laya-med --resume models/laya-med/last.pt
```

Epoch-boundary resume is supported; interruption within an epoch replays that
epoch. Changing the planned epoch count on resume is rejected to preserve the
original learning-rate schedule. Deterministic replay across different devices
or library versions is not guaranteed.

## Evaluate and ablate

```bash
python training/evaluate.py --checkpoint models/laya-med/best.pt
python training/evaluate.py --checkpoint models/laya-med/best.pt --ablation no-images
python training/evaluate.py --checkpoint models/laya-med/best.pt --ablation shuffle-images
python scripts/run_trial.py --manifest data/prepared/manifest.json
```

Evaluation restores the saved architecture, input length, tokenizer and fitted
temperature. It reports raw/calibrated per-dataset accuracy, NLL, Brier and ECE,
macro metrics, synchronized forward latency, pipeline throughput, and peak CUDA
allocation. It writes JSON reports and per-example logits next to the checkpoint.
Use `--batch-size 1` and your target batch size for comparable latency reporting;
forward timings exclude loading, and pipeline throughput includes preprocessing.
Evaluation currently uses FP32; do not directly compare it to an FP16 baseline
without matching precision and measurement boundaries.

`--max-examples N` explicitly marks evaluation as partial. Image shuffling is
within each dataset and maps each image to a different image identity; the image
ablations use the original temperature as a diagnostic, without fitting on test.

Laya baseline batches only examples sharing the exact question/choice schema;
its requested batch size is an upper bound. Distinct questions run separately
because the upstream API accepts a single schema per call.

## Custom rows

Pass `--extra-jsonl path` to preparation, not to training. Every row requires
`id`, `dataset`, `question`, `options`, an integer `label`, and optional `state`.
Image state uses `{"text": "...", "image": "/absolute/path/image.png"}`.
Include `patient_id`, `article_id`, or `group_id` when known. Relative image paths
are interpreted from the preparation process's working directory and saved as
absolute paths. Additional records enter the training source pool and undergo
the same grouping and validation checks.

## Verification

```bash
python -m pytest -q
```

Regression coverage includes split leakage, tokenizer truncation, image masking,
frozen gradients, ragged calibration, exact CPU interrupted/resumed training,
text-only checkpoint evaluation, and heterogeneous Laya batching. The actual
cached encoder smoke-check result is recorded in `verification.json`.
