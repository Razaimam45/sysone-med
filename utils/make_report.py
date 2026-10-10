#!/usr/bin/env python3
"""Create CSV/JSON/Markdown summaries from structured result records."""

import argparse
import csv
import json
from pathlib import Path


def write_report(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, indent=2))
    with (output_dir / "results.csv").open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    columns = ["model", "architecture", "task", "mode", "n", "accuracy", "mean_confidence", "ece", "nll", "brier", "p50_latency", "p95_latency", "throughput", "timestamp"]
    headings = ["Model", "Architecture", "Dataset", "Mode", "N", "Accuracy", "Confidence", "ECE", "NLL", "Brier", "p50 (ms)", "p95 (ms)", "Examples/s", "Timestamp (UTC)"]
    lines = ["# Final comparison", "",
             "p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.", "",
             "Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.", "",
             "Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.", "",
             "| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for record in records:
        values = [record.get(column) for column in columns]
        lines.append("| " + " | ".join(f"{x:.4f}" if isinstance(x,float) else str(x) for x in values) + " |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    write_report(json.loads(args.input.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
