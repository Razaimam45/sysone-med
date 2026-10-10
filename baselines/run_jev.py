#!/usr/bin/env python3
"""Run the hosted JEV API baseline on the text tasks and save to results/jev/

Loads the audited manifest, scores --max-samples text decision rows per task
through the OpenRouter JEV endpoint, and reports accuracy / confidence / ECE /
NLL / Brier / latency using the same metric definitions as the local trial.

Results are saved to results/jev/results.json with same columns for cross-baseline analysis.

    export OPENROUTER_API_KEY=sk-or-v1-...
    python baselines/run_jev.py --max-samples 50 --append
    python baselines/run_jev.py --tasks pubmedqa --max-samples 20
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
from med_jevify.jev_api import JevApiAdapter

TEXT_TASKS = ("pubmedqa", "med_qa", "medmcqa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TEXT_TASKS, default=list(TEXT_TASKS))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data/prepared/manifest.json"))
    parser.add_argument("--model", default="typesafe/jev-1.13")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--calibrate", action="store_true",
                        help="fit a per-task temperature on the calibration split first")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/jev"))
    parser.add_argument("--append", action="store_true",
                        help="merge result rows into results.json and regenerate csv/md reports")
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    manifest, prepared = load_manifest(args.manifest)
    run_timestamp = datetime.now(timezone.utc).isoformat()
    mode = "jev_api_calibrated" if args.calibrate else "jev_api"
    adapter = JevApiAdapter(model=args.model, timeout=args.timeout, max_workers=args.workers)

    records = []
    for task in args.tasks:
        examples = [r for r in prepared["test"] if r["dataset"] == task][:args.max_samples]
        if not examples:
            parser.error(f"No text examples found for task {task}.")
        scaler = None
        if args.calibrate:
            cal = [r for r in prepared["calibration"] if r["dataset"] == task][:args.max_samples]
            from med_jevify.calibration import fit_temperature
            cal_probs, _ = adapter.score(cal, args.max_samples)
            cal_logits = np.log(np.clip(np.asarray(cal_probs, dtype=float), 1e-9, 1.0))
            scaler = fit_temperature(cal_logits, [int(r["label"]) for r in cal])
            print(f"[{task}] calibration temperature = {scaler.temperature:.6f}", flush=True)
        print(f"Scoring {len(examples)} {task} examples through {args.model}...", flush=True)
        record, _ = run_probability_backend(
            examples, adapter, args.model, task, mode, args.max_samples, scaler,
            architecture="non-AR"
        )
        row = record.as_dict() | {"timestamp": run_timestamp}
        records.append(row)
        print(json.dumps(row, indent=2))

    if args.append:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "results.json"
        existing = json.loads(result_path.read_text()) if result_path.exists() else []
        existing = [row for row in existing
                    if not (row.get("model") == args.model and row.get("mode") == mode)]
        existing.extend(records)
        sys.path.insert(0, str(PROJECT_ROOT / "utils"))  # make_report.py lives here
        from make_report import write_report
        write_report(existing, output)
        print(f"\nAppended {len(records)} rows to {result_path}; reports regenerated.")


if __name__ == "__main__":
    main()