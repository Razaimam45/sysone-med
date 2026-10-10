# Inference-time decision policies

Run the selected policies across the enabled backbone/checkpoint configurations
(currently nine of ten; #5 MedGemma-Direct head-only is commented out)
from the project root. Six policies are available; your current defaults are
permutation, memory, selective, and calibrated (routing/conformal stay disabled):

```bash
bash tta/run_all_tta.sh
```

The script uses the `ttw` Conda Python when available, otherwise Python on PATH.
It runs sequentially, loads each model once, reuses original test predictions,
and stops with a nonzero exit status if a configuration fails. Local trained
checkpoints, the prepared manifest/images, model dependencies, and access to
any required Hugging Face weights must be available. Full runs need suitable
model memory; this script does not install dependencies or download datasets.

| Output | Policy | Mechanism |
| --- | --- | --- |
| `routing.json` | Routing (existing) | Confidence thresholds with abstention |
| `conformal.json` | Conformal (existing) | Prediction sets using held-out calibration labels |
| `permutation.json` | Permutation Consistency (PC) | Score distinct option orders, map probabilities back to original option indices, average, and take argmax |
| `memory.json` | Dynamic Memory (DM) | Retrieve previous confident pseudo-decisions, add compact text context, rescore, and take argmax |
| `calibrated.json` | Calibration-guided refinement | Choose a policy using separate labeled calibration blocks; otherwise retain base |
| `selective.json` | Selective Permutation Refinement | Probe reversed option order; if aligned distribution cosine similarity is below the threshold, average extra permutations; otherwise retain the original prediction |

All three new policies use `torch.inference_mode()`: no backward pass, optimizer,
gradient-based update, or checkpoint mutation. They always emit one candidate;
retaining the base answer is not abstention. Legacy `method_drift_cal.py` and
`method_rlcd.py` remain available but are **not run** by either runner, and old
outputs from those methods are excluded from consolidated reports.

The selective policy is a **gradient-free CAS-inspired extension**, not a
reproduction of [Cross-Augmentation Similarity](https://arxiv.org/abs/2609.08367).
Its views permute answer options rather than augmenting images, and its
refinement averages predictions rather than optimizing weights. Probe inference
is still necessary even when refinement is skipped. For binary questions,
there are only two distinct option orders, so selecting refinement requires no
additional views beyond the probe. Report actual scored-example counts rather
than assuming the paper's compute savings apply here.

DM is a simplified retrieval policy inspired by
[Dynamic Cheatsheet](https://aclanthology.org/2026.eacl-long.333/), not its full
strategy-generation machinery. Retrieval uses lexical Jaccard similarity over
question, context, and candidate words. Memory starts empty for each run,
is partitioned by dataset, and follows manifest order. Only prior predictions
meeting the confidence threshold enter a bounded FIFO memory; truth labels and
future cases are never used. Retrieved entries explicitly identify guesses as
unverified. Image paths are preserved, while retrieved text is prepended to the
current context and subject to each backend's existing context truncation.
Zero-overlap retrieval and empty memory keep the base prediction. These policies
can change predictions but do not guarantee an accuracy improvement.

## Configuration

Defaults: 2,000 test examples **per dataset**, four distinct permutations
(including the original, capped at the number of possible orders), seed 0,
CAS threshold 0.95, memory top-k 3, confidence threshold 0.8, capacity 256 per
dataset, and at most 800 characters of retrieved context.

```bash
# Small run across the enabled configurations, with separate output directories:
MAX_SAMPLES=2 OUTPUT_ROOT=tta_smoke bash tta/run_all_tta.sh

# Override inference-policy settings for the enabled configurations:
bash tta/run_all_tta.sh --permutations 6 --cas-threshold 0.9 \
  --memory-k 2 --memory-confidence 0.85

# Run one configuration:
python tta/run_tta.py --baseline medgemma_direct_lora \
  --checkpoint models/qwen_direct_lora --backbone qwen \
  --datasets pubmedqa --max-samples 10 --output-dir tta_results/qwen_direct_lora_tta
```

Environment overrides: `TTW_PYTHON`, `CUDA_VISIBLE_DEVICES`, `DEVICE`,
`MAX_SAMPLES`, and `OUTPUT_ROOT`. Extra command-line arguments are forwarded to
the enabled invocations. Avoid overriding `--output-dir` globally, since that would
send different checkpoints to the same directory.

Configurations #3 (Laya) and #4 (MedJev) use the three text datasets. The other
eight use all seven datasets. The ten output directories are `laya_med_tta`,
`laya_med_vl_tta`, `laya_tta`, `medjev_tta`, `medgemma_direct_headonly_tta`,
`medgemma_direct_lora_tta`, `medgemma_ll_tta`, `qwen_direct_headonly_tta`,
`qwen_direct_lora_tta`, and `qwen_ll_tta` under `OUTPUT_ROOT`.

Each contains JSON files for the selected policies plus `summary.json`. Existing
files for unselected policies remain untouched. New-policy reports
include base/refined accuracy and ECE, per-dataset accuracy, prediction changes,
per-example probabilities, and extra scored-example counts. Consolidated
`tta_summary.json` reports live in the corresponding directory without `_tta`;
checkpoint variants have separate reports. Routing and conformal retain their
existing metrics and thresholds.

## Validation without model downloads

```bash
python -m pytest -q tests/test_tta_policies.py
bash -n tta/run_all_tta.sh
```

Tests cover remapping and probability averaging, distinct-order caps, prediction
changes, selective gate branches, chronological memory and dataset isolation,
label independence, image preservation, disabled gradients, all five reports,
and the enabled shell invocations. These do not substitute for a hardware-backed
run with the real checkpoints.

## Experimental evidence-prior correction

`method_prior.py` adds a separate, opt-in gradient-free candidate. For base
confidence below 0.6, it scores the same question/options after removing state
text and image, and proposes `softmax(log(p_base) - 0.25 * log(p_null))`. It
accepts the correction only when the corrected top-two margin is at least
0.05. High-confidence cases and cases without removable evidence retain the
base distribution. These are fixed heuristic settings, not a guarantee of
improvement or a claim of established research novelty. Questions containing
all the clinical evidence are not made content-free by this operation.

```bash
CUDA_VISIBLE_DEVICES=3 python -m tta.method_prior \
  --baseline laya_med --checkpoint models/laya-med/best.pt \
  --device cuda:0 --max-samples 10 --output /tmp/prior_smoke.json
```

For MedGemma-Direct LoRA use `--baseline medgemma_direct_lora --checkpoint
models/medgemma-direct`. Here `--max-samples` is a **total** cap, selected
round-robin across the requested datasets, not a per-dataset cap. This
experimental candidate is not automatically added to `run_all_tta.sh`.

## Experimental calibration-guided refinement

`method_calibrated.run(args, adapter=None, test=None)` tries base prediction,
subtracting/reinforcing the evidence-free prior, one retrieved labeled reference,
and two weak answer-frequency corrections. It uses **labeled calibration data**:
the first 64 cases per dataset form a reference bank, the next eight select a
policy, and the next eight guard against a one-block gain. A refinement must
beat base on both eight-case blocks; ties retain base. Datasets with fewer than
80 calibration examples retain base. Reference cases never come from test.
This is calibration-assisted inference, not unlabeled online learning, and its
small empirical guard provides no formal guarantee of test accuracy.

It is integrated as the sixth available policy, `calibrated`, and writes
selection/guard counts alongside its accuracy report. All scoring remains
gradient-free. Run only this new policy, preserving your existing five results:

```bash
bash tta/run_all_tta.sh --methods calibrated
```

Or explicitly run all six:

```bash
bash tta/run_all_tta.sh --methods routing conformal permutation memory selective calibrated
```

The same `--methods` option works with `python tta/run_tta.py` for a single
checkpoint. Only the names in `METHODS` in `run_tta.py` control defaults; a second
module tuple no longer needs to be edited. The new policy saves `calibrated.json`
and is included in `tta_summary.json`. It uses up to 80 calibration rows per
dataset independently of the test `--max-samples` cap.
