#!/usr/bin/env bash
# Run the five SysOne-Med decision formulations on the medical datasets.
#
#   1. Jev        — hosted JEV API baseline (text tasks)
#   2. AR-Gen     — MedGemma autoregressive, 32-token generation
#   3. AR-LL      — MedGemma autoregressive, native one-token
#   4. AR-Head    — MedGemma frozen backbone + decision head (head-only)
#   5. AR-Adapt   — MedGemma frozen backbone + decision head + LoRA adapters
#   6. NAR        — LayaMed multimodal non-autoregressive decision
#
# Uses the `ttw` conda env (open_clip, jevify, laya, torch, transformers).
# Set your dataset/model paths via the flags below (see README "Datasets").
set -e
cd "$(dirname "$0")/.."

# Resolve the ttw env python explicitly so this works regardless of PATH.
TTW_PYTHON="$(conda info --base 2>/dev/null)/envs/ttw/bin/python"
if [ ! -x "$TTW_PYTHON" ]; then
    TTW_PYTHON="$(command -v python)"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# All seven datasets (text + image). Trim to a subset to run faster.
TASKS="pubmedqa med_qa medmcqa vqarad slake path_vqa pmc_vqa"
MAX="${MAX:-99999}"

# 1. Jev (hosted API, text tasks only) — needs OPENROUTER_API_KEY
# export OPENROUTER_API_KEY=<your-openrouter-key>   # set in your environment; never commit real keys
# $TTW_PYTHON baselines/run_jev.py --tasks pubmedqa med_qa medmcqa --max-samples "$MAX" --append

# 2. AR-Gen (MedGemma, conventional 32-token generation) — text + image
$TTW_PYTHON baselines/run_ar.py --tasks $TASKS --max-samples "$MAX"

# 3. AR-LL (MedGemma, native one-token) — text + image
$TTW_PYTHON baselines/run_ll.py --tasks $TASKS --max-samples "$MAX"

# 4. AR-Head (MedGemma frozen backbone + decision head) — text + image
$TTW_PYTHON baselines/run_direct_headonly.py --tasks $TASKS --max-samples "$MAX"

# 5. AR-Adapt (MedGemma frozen backbone + decision head + LoRA) — text + image
$TTW_PYTHON baselines/run_direct_lora.py --tasks $TASKS --max-samples "$MAX"

# 6. NAR (LayaMed multimodal non-autoregressive) — text + image
$TTW_PYTHON baselines/run_laya_med.py --tasks $TASKS --max-samples "$MAX" --append