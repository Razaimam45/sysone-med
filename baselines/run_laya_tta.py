#!/usr/bin/env python3
"""Run TTA on Laya (text-only NAR decision backbone).

Saves results to tta_results/laya_tta/ as CSV/JSON/Markdown reports.
Uses LayaTtaAdapter which plugs into the standard run_trial framework.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_trial import main as run_trial_main  # noqa: E402

def main_laya() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+",
                        choices=["pubmedqa", "vqarad", "med_qa", "medmcqa", "slake", "path_vqa", "pmc_vqa"],
                        default=["pubmedqa", "med_qa", "medmcqa"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--non-ar-model", default="convaiinnovations/laya")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=str(ROOT / "results/laya"))
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