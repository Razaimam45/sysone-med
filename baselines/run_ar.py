#!/usr/bin/env python3
"""Run the MedGemma-AR (conventional 32-token generation) baseline.

Saves results to results/medgemma_ar/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_trial import main  # noqa: E402


def main_ar() -> None:
    """Entry point: run the MedGemma conventional_llm baseline only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+",
                        choices=["pubmedqa", "vqarad", "med_qa", "medmcqa", "slake", "path_vqa", "pmc_vqa"],
                        default=["pubmedqa", "vqarad", "med_qa", "medmcqa", "slake", "path_vqa", "pmc_vqa"])
    parser.add_argument("--max-samples", "--max-examples", dest="max_samples", type=int, default=50)
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/medgemma_ar"))
    args = parser.parse_args()
    main([
        "--models", args.model,
        "--modes", "conventional_llm",
        "--tasks", *args.tasks,
        "--max-examples", str(args.max_samples),
        "--device", args.device,
        "--dtype", args.dtype,
        "--output-dir", args.output_dir,
    ])


if __name__ == "__main__":
    main_ar()