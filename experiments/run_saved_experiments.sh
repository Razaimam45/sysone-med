#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT_DIR:-$ROOT/experiment_results}"
MANIFEST="${MANIFEST:-$ROOT/data/prepared/manifest.json}"
TEST_ROWS="${TEST_ROWS:-$ROOT/data/prepared/test.jsonl}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2000}"
START_AT="${START_AT:-constrained}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=3

STAGES=(constrained laya-test laya-calibration laya-no-images laya-shuffled-images \
    head-test head-calibration lora-test lora-calibration lora-no-images lora-shuffled-images \
    audit summary compare-head compare-constrained compare-lora visual-laya visual-lora \
    posthoc latency scaling)
declare -A STAGE_INDEX
for index in "${!STAGES[@]}"; do
    STAGE_INDEX["${STAGES[$index]}"]="$index"
done
if [[ ! -v "STAGE_INDEX[$START_AT]" ]]; then
    printf 'Unknown START_AT stage: %s\nValid stages: %s\n' "$START_AT" "${STAGES[*]}" >&2
    exit 2
fi
START_INDEX="${STAGE_INDEX[$START_AT]}"

run_stage() {
    local stage="$1"
    shift
    if (( ${STAGE_INDEX[$stage]} < START_INDEX )); then
        printf 'Skipping completed stage: %s\n' "$stage"
        return 0
    fi
    printf '\n==> %s\n' "$stage"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'Dry run: would execute %s\n' "$*"
        return 0
    fi
    "$@"
}

LAYA_CHECKPOINT="${LAYA_CHECKPOINT:-$ROOT/models/laya-med/best.pt}"
HEAD_CHECKPOINT="${HEAD_CHECKPOINT:-$ROOT/models/medgemma-direct-headonly/best.pt}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-$ROOT/models/medgemma-direct/best.pt}"

for checkpoint in "$LAYA_CHECKPOINT" "$HEAD_CHECKPOINT" "$LORA_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
        printf 'Missing required checkpoint: %s\n' "$checkpoint" >&2
        exit 2
    fi
done

if [[ "${RUN_CONSTRAINED:-1}" == "1" ]]; then
    run_stage constrained python scripts/run_trial.py --models google/medgemma-4b-it --modes constrained_generate \
        --tasks pubmedqa vqarad med_qa medmcqa slake path_vqa pmc_vqa \
        --max-examples "$MAX_EXAMPLES" --manifest "$MANIFEST" \
        --output-dir "$ROOT/results/medgemma_constrained"
fi

run_stage laya-test python training/evaluate.py --checkpoint "$LAYA_CHECKPOINT" --manifest "$MANIFEST" \
    --split test --max-examples "$MAX_EXAMPLES" --ablation none --output "$OUT/laya/correct-test.json"
run_stage laya-calibration python training/evaluate.py --checkpoint "$LAYA_CHECKPOINT" --manifest "$MANIFEST" \
    --split calibration --max-examples "$MAX_EXAMPLES" --ablation none --output "$OUT/laya/correct-calibration.json"
run_stage laya-no-images python training/evaluate.py --checkpoint "$LAYA_CHECKPOINT" --manifest "$MANIFEST" \
    --split test --max-examples "$MAX_EXAMPLES" --ablation no-images --output "$OUT/laya/missing-image-test.json"
run_stage laya-shuffled-images python training/evaluate.py --checkpoint "$LAYA_CHECKPOINT" --manifest "$MANIFEST" \
    --split test --max-examples "$MAX_EXAMPLES" --ablation shuffle-images --seed 42 --output "$OUT/laya/shuffled-image-test.json"

run_stage head-test python training/evaluate_medgemma_direct.py --checkpoint "$HEAD_CHECKPOINT" --manifest "$MANIFEST" \
    --split test --max-examples "$MAX_EXAMPLES" --output "$OUT/medgemma-head/test.json"
run_stage head-calibration python training/evaluate_medgemma_direct.py --checkpoint "$HEAD_CHECKPOINT" --manifest "$MANIFEST" \
    --split calibration --max-examples "$MAX_EXAMPLES" --output "$OUT/medgemma-head/calibration.json"
run_stage lora-test python training/evaluate_medgemma_direct.py --checkpoint "$LORA_CHECKPOINT" --manifest "$MANIFEST" \
    --split test --max-examples "$MAX_EXAMPLES" --output "$OUT/medgemma-lora/test.json"
run_stage lora-calibration python training/evaluate_medgemma_direct.py --checkpoint "$LORA_CHECKPOINT" --manifest "$MANIFEST" \
    --split calibration --max-examples "$MAX_EXAMPLES" --output "$OUT/medgemma-lora/calibration.json"
run_stage lora-no-images python training/evaluate_medgemma_direct.py --checkpoint "$LORA_CHECKPOINT" --manifest "$MANIFEST" \
    --split test --max-examples "$MAX_EXAMPLES" --ablation no-images --output "$OUT/medgemma-lora/missing-image-test.json"
run_stage lora-shuffled-images python training/evaluate_medgemma_direct.py --checkpoint "$LORA_CHECKPOINT" --manifest "$MANIFEST" \
    --split test --max-examples "$MAX_EXAMPLES" --ablation shuffle-images --seed 42 --output "$OUT/medgemma-lora/shuffled-image-test.json"

run_stage audit python analysis/run_experiments.py audit --manifest-test "$TEST_ROWS" \
    --predictions \
    MedGemma-AR="$ROOT/results/medgemma_ar/details/predictions.jsonl" \
    MedGemma-LL="$ROOT/results/medgemma_ll/details/predictions.jsonl" \
    MedGemma-Constrained="$ROOT/results/medgemma_constrained/details/predictions.jsonl" \
    MedGemma-Head="$OUT/medgemma-head/test.predictions.jsonl" \
    MedGemma-LoRA="$OUT/medgemma-lora/test.predictions.jsonl" \
    Laya-Med="$OUT/laya/correct-test.predictions.jsonl" \
    --max-per-dataset "$MAX_EXAMPLES" \
    --output "$OUT/candidate_audit.csv" --rows-output "$OUT/candidate_rows.csv"

run_stage summary python analysis/run_experiments.py summary --predictions \
    MedGemma-AR="$ROOT/results/medgemma_ar/details/predictions.jsonl" \
    MedGemma-LL="$ROOT/results/medgemma_ll/details/predictions.jsonl" \
    MedGemma-Constrained="$ROOT/results/medgemma_constrained/details/predictions.jsonl" \
    MedGemma-Head="$OUT/medgemma-head/test.predictions.jsonl" \
    MedGemma-LoRA="$OUT/medgemma-lora/test.predictions.jsonl" \
    Laya-Med="$OUT/laya/correct-test.predictions.jsonl" \
    --max-per-dataset "$MAX_EXAMPLES" \
    --output "$OUT/main_results.csv"

run_stage compare-head python analysis/run_experiments.py compare \
    --reference "MedGemma-LL=$ROOT/results/medgemma_ll/details/predictions.jsonl" \
    --candidate "MedGemma-Head=$OUT/medgemma-head/test.predictions.jsonl" \
    --max-per-dataset "$MAX_EXAMPLES" \
    --output "$OUT/ll_vs_head_paired.csv"
run_stage compare-constrained python analysis/run_experiments.py compare \
    --reference "MedGemma-LL=$ROOT/results/medgemma_ll/details/predictions.jsonl" \
    --candidate "MedGemma-Constrained=$ROOT/results/medgemma_constrained/details/predictions.jsonl" \
    --max-per-dataset "$MAX_EXAMPLES" \
    --output "$OUT/ll_vs_constrained_paired.csv"
run_stage compare-lora python analysis/run_experiments.py compare \
    --reference "MedGemma-Head=$OUT/medgemma-head/test.predictions.jsonl" \
    --candidate "MedGemma-LoRA=$OUT/medgemma-lora/test.predictions.jsonl" \
    --max-per-dataset "$MAX_EXAMPLES" \
    --output "$OUT/head_vs_lora_paired.csv"

run_stage visual-laya python analysis/run_experiments.py visual \
    --correct-image "$OUT/laya/correct-test.predictions.jsonl" \
    --missing-image "$OUT/laya/missing-image-test.predictions.jsonl" \
    --shuffled-image "$OUT/laya/shuffled-image-test.predictions.jsonl" \
    --output-dir "$OUT/visual_dependence"
run_stage visual-lora python analysis/run_experiments.py visual \
    --correct-image "$OUT/medgemma-lora/test.predictions.jsonl" \
    --missing-image "$OUT/medgemma-lora/missing-image-test.predictions.jsonl" \
    --shuffled-image "$OUT/medgemma-lora/shuffled-image-test.predictions.jsonl" \
    --output-dir "$OUT/visual_dependence_medgemma_lora"
run_stage posthoc python analysis/run_experiments.py posthoc \
    --calibration "$OUT/laya/correct-calibration.predictions.jsonl" \
    --test "$OUT/laya/correct-test.predictions.jsonl" \
    --output-dir "$OUT/posthoc"
run_stage latency python analysis/run_experiments.py latency --predictions \
    MedGemma-AR="$ROOT/results/medgemma_ar/details/predictions.jsonl" \
    MedGemma-LL="$ROOT/results/medgemma_ll/details/predictions.jsonl" \
    MedGemma-Head="$OUT/medgemma-head/test.predictions.jsonl" \
    MedGemma-LoRA="$OUT/medgemma-lora/test.predictions.jsonl" \
    Laya-Med="$OUT/laya/correct-test.predictions.jsonl" \
    --max-per-dataset "$MAX_EXAMPLES" \
    --output-dir "$OUT/observed_latency"

if [[ "${RUN_SCALING:-1}" == "1" ]]; then
    run_stage scaling python training/benchmark_laya_scaling.py --checkpoint "$LAYA_CHECKPOINT" \
        --manifest "$MANIFEST" --max-examples "$MAX_EXAMPLES" \
        --output-dir "$OUT/candidate_scaling"
fi

printf 'Experiment assets written under %s\n' "$OUT"
