#!/usr/bin/env python3
"""Run the Laya (text-only NAR decision backbone) baseline.

Saves results to results/laya/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_trial import main  # noqa: E402


def main_laya() -> None:
    """Entry point: run the Laya non_ar_decision baseline only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+",
                        choices=["pubmedqa", "vqarad", "med_qa", "medmcqa", "slake", "path_vqa", "pmc_vqa"],
                        default=["pubmedqa", "vqarad", "med_qa", "medmcqa", "slake", "path_vqa", "pmc_vqa"])
    parser.add_argument("--max-samples", "--max-examples", dest="max_samples", type=int, default=50)
    parser.add_argument("--non-ar-model", default="convaiinnovations/laya")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/laya"))
    parser.add_argument("--calibrate", action="store_true",
                        help="also run the calibrated_non_ar_decision mode")
    args = parser.parse_args()

    modes = ["non_ar_decision"]
    if args.calibrate:
        modes.append("calibrated_non_ar_decision")
    main([
        "--modes", *modes,
        "--tasks", *args.tasks,
        "--max-examples", str(args.max_samples),
        "--non-ar-model", args.non_ar_model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--batch-size", str(args.batch_size),
        "--output-dir", args.output_dir,
    ])


if __name__ == "__main__":
    main_laya()