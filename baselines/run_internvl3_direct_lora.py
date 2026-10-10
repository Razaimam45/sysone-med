#!/usr/bin/env python3
"""Run InternVL3-2B Direct LoRA baseline evaluation (TEXT-ONLY).

NOTE: InternVL3-2B's processor under transformers 5.x is a bare Qwen2 tokenizer
that strips image blocks. Multimodal evaluation is NOT supported.
This script evaluates on text-only datasets only (PubMedQA, MedQA, MedMCQA).

The LoRA checkpoint holds the trained decision-head weights plus rank-`rank`
LoRA adapters on the q/k/v/o projections of the InternVL language model.

Results are saved to results/internvl3_direct_lora/
"""

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
from med_jevify.medgemma_direct import MedGemmaDirectAdapter

# Text-only datasets (InternVL3 processor doesn't support multimodal)
TEXT_TASKS = ("pubmedqa", "med_qa", "medmcqa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TEXT_TASKS, default=list(TEXT_TASKS))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data/prepared/manifest.json"))
    parser.add_argument("--run", default="models/internvl3_direct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/internvl3_direct_lora"))
    parser.add_argument("--append", action="store_true",
                        help="merge result rows into results.json and regenerate csv/md reports")
    parser.add_argument("--backbone", default="internvl3", help="Backbone: internvl3 (InternVL3-2B)")
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    manifest, prepared = load_manifest(args.manifest)
    run_timestamp = datetime.now(timezone.utc).isoformat()
    mode = "internvl3_direct_lora"
    adapter = MedGemmaDirectAdapter(run=args.run, device=args.device, dtype=args.dtype, lora=True, backbone=args.backbone)

    records = []
    for task in args.tasks:
        examples = [r for r in prepared["test"] if r["dataset"] == task][:args.max_samples]
        if not examples:
            parser.error(f"No examples found for task {task}.")
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
    from make_report import write_report  # noqa: E402
    write_report(existing, output)
    print(f"\nSaved {len(records)} rows to {result_path}; reports regenerated.")


if __name__ == "__main__":
    main()