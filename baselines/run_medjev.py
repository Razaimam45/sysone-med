#!/usr/bin/env python3
"""Run the local MedJev checkpoint baseline on the text tasks and save to results/medjev/

Loads the audited manifest, scores --max-samples text decision rows per task
through the MedJev checkpoint, and reports accuracy / confidence / ECE / NLL /
Brier / latency using the same metric definitions as the local trial.

Results are saved to results/medjev/results.json with same columns for cross-baseline analysis.

    python baselines/run_medjev.py --tasks pubmedqa --max-samples 5
    python baselines/run_medjev.py --max-samples 50 --append
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from med_jevify.decision_data import load_manifest
from med_jevify.evaluate import run_probability_backend
from med_jevify.medjev_adapter import MedJevAdapter, free_device

TEXT_TASKS = ("pubmedqa", "med_qa", "medmcqa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TEXT_TASKS, default=list(TEXT_TASKS))
    parser.add_argument("--max-samples", "--max-examples", dest="max_samples", type=int, default=50)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data/prepared/manifest.json"))
    parser.add_argument("--run", default="junma/MedJev-Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto",
                        help='cuda device, or "auto" to pick the most free GPU')
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent requests (default 1; the pointer head is per-record)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/medjev"))
    parser.add_argument("--append", action="store_true",
                        help="merge result rows into results.json and regenerate csv/md reports")
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    manifest, prepared = load_manifest(args.manifest)
    run_timestamp = datetime.now(timezone.utc).isoformat()
    mode = "medjev"
    adapter = MedJevAdapter(run=args.run, device=free_device(args.device), dtype=args.dtype)

    records = []
    for task in args.tasks:
        examples = [r for r in prepared["test"] if r["dataset"] == task][:args.max_samples]
        if not examples:
            parser.error(f"No text examples found for task {task}.")
        print(f"Scoring {len(examples)} {task} examples through {args.run}...", flush=True)
        record, _ = run_probability_backend(
            examples, adapter, args.run, task, mode, args.max_samples, None,
            architecture="non-AR"
        )
        row = record.as_dict() | {"timestamp": run_timestamp}
        records.append(row)
        print(json.dumps(row, indent=2))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    existing = json.loads(result_path.read_text()) if result_path.exists() else []
    # Drop rows from this exact run (same model/mode/timestamp) before merging,
    # so re-runs replace rather than duplicate.
    existing = [row for row in existing
                if not (row.get("model") == args.run and row.get("mode") == mode
                        and row.get("timestamp") == run_timestamp)]
    existing.extend(records)
    sys.path.insert(0, str(PROJECT_ROOT / "utils"))  # make_report.py lives here
    from make_report import write_report
    write_report(existing, output)
    print(f"\nSaved {len(records)} rows to {result_path}; reports regenerated.")


if __name__ == "__main__":
    main()