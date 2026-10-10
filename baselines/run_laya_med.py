#!/usr/bin/env python3
"""Run the Laya-Med (multimodal NAR decision) baseline.

Saves results to results/laya_med/

Delegates to training/evaluate.py, which is the same evaluation path used for
the previous Laya-Med runs. Only the calibrated row is emitted (the checkpoint
holds a held-out calibration temperature).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from evaluate import main as evaluate_main  # noqa: E402

ALL_TASKS = ("pubmedqa", "med_qa", "medmcqa", "vqarad", "slake", "path_vqa", "pmc_vqa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=list(ALL_TASKS))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--run", default="models/laya-med")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/laya_med"))
    parser.add_argument("--append", action="store_true",
                        help="merge result rows into results.json and regenerate csv/md reports")
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    output_dir = Path(args.output_dir)
    if args.append:
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluate_main([
            "--checkpoint", str(Path(args.run) / "best.pt"),
            "--datasets", *args.tasks,
            "--max-examples", str(args.max_samples),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--ablation", "none",
            "--append-results", str(output_dir),
            "--model", "laya-med",
        ])
    else:
        evaluate_main([
            "--checkpoint", str(Path(args.run) / "best.pt"),
            "--datasets", *args.tasks,
            "--max-examples", str(args.max_samples),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--ablation", "none",
            "--output", str(output_dir / "evaluation.json"),
            "--model", "laya-med",
        ])


if __name__ == "__main__":
    main()