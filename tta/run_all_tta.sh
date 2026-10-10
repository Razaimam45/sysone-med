#!/usr/bin/env bash
# Run selected policies for enabled configurations; pass --methods calibrated to run only the new policy.
set -euo pipefail
cd "$(dirname "$0")/.."

# Respect explicit interpreter/device overrides; otherwise prefer the ttw env.
if [[ -z "${TTW_PYTHON:-}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    TTW_PYTHON="${CONDA_BASE}/envs/ttw/bin/python"
    if [[ ! -x "$TTW_PYTHON" ]]; then
        TTW_PYTHON="$(command -v python3 || command -v python)"
    fi
fi
export CUDA_VISIBLE_DEVICES=3
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-tta_results}"

DATASETS="pubmedqa med_qa medmcqa vqarad slake path_vqa pmc_vqa"

# 1. Laya-Med (multimodal NAR) — all 7 datasets
echo "=== Laya-Med ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline laya_med \
    --checkpoint models/laya-med/best.pt \
    --datasets $DATASETS \
    --device "$DEVICE" --batch-size 8 --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_ROOT/laya_med_tta" "$@"

# 2. Laya-Med Vision LoRA — same pipeline, different checkpoint
echo "=== Laya-Med Vision LoRA ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline laya_med \
    --checkpoint models/laya-med-vision-lora/best.pt \
    --datasets $DATASETS \
    --device "$DEVICE" --batch-size 8 --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_ROOT/laya_med_vl_tta" "$@"

# 3. Laya (text-only NAR) — text datasets only
echo "=== Laya ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline laya \
    --checkpoint convaiinnovations/laya \
    --datasets pubmedqa med_qa medmcqa \
    --device "$DEVICE" --batch-size 16 --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_ROOT/laya_tta" "$@"

# 4. MedJev (text-only NAR) — text datasets only
echo "=== MedJev ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline medjev \
    --checkpoint junma/MedJev-Qwen3.5-0.8B \
    --datasets pubmedqa med_qa medmcqa \
    --device "$DEVICE" --batch-size 1 --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_ROOT/medjev_tta" "$@"

# 5. MedGemma-Direct head-only — all 7 datasets
# echo "=== MedGemma-Direct head-only ==="
# "$TTW_PYTHON" -u tta/run_tta.py \
#     --baseline medgemma_direct_headonly \
#     --checkpoint models/medgemma-direct-headonly \
#     --datasets $DATASETS \
#     --device "$DEVICE" --batch-size 8 --max-samples "$MAX_SAMPLES" \
#     --output-dir "$OUTPUT_ROOT/medgemma_direct_headonly_tta" "$@"

# 6. MedGemma-Direct LoRA — all 7 datasets
echo "=== MedGemma-Direct LoRA ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline medgemma_direct_lora \
    --checkpoint models/medgemma-direct \
    --datasets $DATASETS \
    --device "$DEVICE" --batch-size 8 --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_ROOT/medgemma_direct_lora_tta" "$@"

# 7. MedGemma-LL — native candidate-label likelihood (b), all 7 datasets
echo "=== MedGemma-LL ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline medgemma_ll \
    --checkpoint google/medgemma-4b-it \
    --datasets $DATASETS \
    --device "$DEVICE" --batch-size 1 --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_ROOT/medgemma_ll_tta" "$@"

# 8. Qwen2.5-VL-3B Direct head-only — all 7 datasets
echo "=== Qwen2.5-VL-3B Direct head-only ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline medgemma_direct_headonly \
    --checkpoint models/qwen_direct_headonly \
    --datasets $DATASETS \
    --device "$DEVICE" --batch-size 8 --max-samples "$MAX_SAMPLES" \
    --backbone qwen \
    --output-dir "$OUTPUT_ROOT/qwen_direct_headonly_tta" "$@"

# 9. Qwen2.5-VL-3B Direct LoRA — all 7 datasets
echo "=== Qwen2.5-VL-3B Direct LoRA ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline medgemma_direct_lora \
    --checkpoint models/qwen_direct_lora \
    --datasets $DATASETS \
    --device "$DEVICE" --batch-size 8 --max-samples "$MAX_SAMPLES" \
    --backbone qwen \
    --output-dir "$OUTPUT_ROOT/qwen_direct_lora_tta" "$@"

# 10. Qwen2.5-VL-3B LL — native candidate-label likelihood
echo "=== Qwen2.5-VL-3B LL ==="
"$TTW_PYTHON" -u tta/run_tta.py \
    --baseline medgemma_ll \
    --checkpoint Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets $DATASETS \
    --device "$DEVICE" --batch-size 1 --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_ROOT/qwen_ll_tta" "$@"

echo ""
echo "Done. Selected policies per baseline in $OUTPUT_ROOT/<baseline>_tta/."
