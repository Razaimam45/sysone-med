#!/usr/bin/env python3
"""Run MedGemma-Direct LoRA baseline evaluation.

This script evaluates the MedGemma-Direct LoRA model on text/multimodal decision tasks.
Results are saved to results/medgemma_direct_lora
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data/prepared/manifest.json"))
    parser.add_argument("--run", default="models/medgemma-direct-lora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/medgemma_direct_lora"))
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    print("MedGemma-Direct LoRA evaluation - model still being trained.")
    print(f"Run with: python {__file__} --max-samples 50 --append")


if __name__ == "__main__":
    main()
