#!/usr/bin/env bash
# Run all baselines.
# Uses the ttw env which has open_clip, jevify, laya, torch, transformers, medjev.
set -e
cd "$(dirname "$0")/.."

# Resolve the ttw env python explicitly so this works regardless of PATH.
TTW_PYTHON="$(conda info --base 2>/dev/null)/envs/ttw/bin/python"
if [ ! -x "$TTW_PYTHON" ]; then
    TTW_PYTHON="$(command -v python)"
fi
export CUDA_VISIBLE_DEVICES=3

# 1. MedGemma-AR (conventional 32-token) — text + image
# $TTW_PYTHON baselines/run_ar.py --tasks pubmedqa med_qa medmcqa vqarad slake path_vqa pmc_vqa --max-samples 99999

# 2. MedGemma-LL (native 1-token) — text + image
# $TTW_PYTHON baselines/run_ll.py --tasks pubmedqa med_qa medmcqa vqarad slake path_vqa pmc_vqa --max-samples 99999

# 3. Jev
# export OPENROUTER_API_KEY=<your-openrouter-key>   # set in your environment; never commit real keys
# python baselines/run_jev.py --max-samples 1000 --append

# 4. MedJev (local checkpoint) — text only
# $TTW_PYTHON baselines/run_medjev.py --max-samples 2000 --device auto --dtype bf16

# 5. MedGemma-Direct head-only — text + image (all 7 datasets)
# $TTW_PYTHON baselines/run_direct_headonly.py --max-samples 2000

# 6. MedGemma-Direct LoRA — text + image (all 7 datasets)
# $TTW_PYTHON baselines/run_direct_lora.py --max-samples 2000

# 7. Laya (text-only NAR) — text only
# $TTW_PYTHON baselines/run_laya.py --max-samples 2000

# 8. Laya-Med (multimodal NAR) — text + image, calibrated only
# $TTW_PYTHON baselines/run_laya_med.py --max-samples 2000 --append

# 9. Laya-Med Vision LoRA — text + image, calibrated only
# $TTW_PYTHON baselines/run_laya_med.py \
#     --run models/laya-med-vision-lora \
#     --output-dir results/laya_med_vl \
#     --max-samples 2000 \
#     --append

# 10. Qwen2.5-VL-3B AR (conventional 32-token) — text + image
# $TTW_PYTHON baselines/run_ar.py --model Qwen/Qwen2.5-VL-3B-Instruct \
#     --output-dir results/qwen_ar \
#     --tasks pubmedqa med_qa medmcqa vqarad slake path_vqa pmc_vqa --max-samples 2000

# 11. Qwen2.5-VL-3B LL (native 1-token) — text + image
# $TTW_PYTHON baselines/run_ll.py --model Qwen/Qwen2.5-VL-3B-Instruct \
#     --output-dir results/qwen_ll \
#     --tasks pubmedqa med_qa medmcqa vqarad slake path_vqa pmc_vqa --max-samples 2000

# 12. Qwen2.5-VL-3B Direct head-only — text + image (all 7 datasets)
# $TTW_PYTHON baselines/run_qwen_direct_headonly.py --max-samples 2000 \
#     --output-dir results/qwen_direct_headonly

# 13. Qwen2.5-VL-3B Direct LoRA — text + image (all 7 datasets)
$TTW_PYTHON baselines/run_qwen_direct_lora.py --max-samples 2000 \
    --output-dir results/qwen_direct_lora