#!/usr/bin/env python3
"""Measure Laya-Med latency versus candidate count on matched examples.

Writes candidate_scaling.csv and total/component latency figures (PNG/PDF).
This is a timing experiment only: labels are replaced with a valid dummy label
because no accuracy is computed for the truncated candidate sets.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tta"))
from laya_med.data import MedicalCollator, MedicalDecisionDataset
from train import forward, move
from tta.common import load_bundle, torch_device


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class ModuleTimer:
    def __init__(self, model, device):
        self.device = device
        self.events = {"text_encoder": [], "vision_encoder": [], "fusion": [], "option_head": []}
        self.handles = []
        for name in self.events:
            module = getattr(model, name, None)
            if module is not None:
                self.handles.append((name, module.register_forward_pre_hook(self._start(name))))
                self.handles.append((name, module.register_forward_hook(self._end(name))))
        self.starts = {}

    def _start(self, name):
        def hook(module, inputs):
            if self.device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.starts.setdefault(name, []).append(event)
            else:
                self.starts.setdefault(name, []).append(time.perf_counter())
        return hook

    def _end(self, name):
        def hook(module, inputs, output):
            start = self.starts[name].pop()
            if self.device.type == "cuda":
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.events[name].append((start, end))
            else:
                self.events[name].append((start, time.perf_counter()))
        return hook

    def reset(self):
        for values in self.events.values():
            values.clear()
        self.starts.clear()

    def milliseconds(self):
        sync(self.device)
        result = {}
        for name, pairs in self.events.items():
            if self.device.type == "cuda":
                result[name] = sum(start.elapsed_time(end) for start, end in pairs)
            else:
                result[name] = sum((end - start) * 1000 for start, end in pairs)
        return result

    def close(self):
        for _, handle in self.handles:
            handle.remove()


def build_rows(rows, candidate_count):
    selected = []
    for row in rows:
        if len(row["options"]) < candidate_count:
            continue
        copied = dict(row)
        copied["options"] = list(row["options"][:candidate_count])
        copied["label"] = 0
        selected.append(copied)
    return selected


def benchmark(model, tokenizer, processor, saved, rows, dataset_name, candidate_counts,
              batch_size, device, output_dir):
    timers = ModuleTimer(model, device)
    rows_out = []
    supported_counts = [count for count in candidate_counts
                        if count <= max(len(row["options"]) for row in rows)]
    if not supported_counts:
        return rows_out
    max_count = max(supported_counts)
    matched_rows = build_rows(rows, max_count)
    collator = MedicalCollator(tokenizer, processor, saved["max_length"])
    for count in supported_counts:
        subset = build_rows(matched_rows, count)
        loader = DataLoader(MedicalDecisionDataset(subset), batch_size=batch_size,
                            collate_fn=collator, num_workers=0)
        first = next(iter(loader))
        with torch.inference_mode():
            forward(model, move(first, device))
        sync(device)
        prep_ms, transfer_ms, forward_ms = [], [], []
        component_ms = {name: [] for name in timers.events}
        iterator = iter(loader)
        progress = tqdm(total=len(loader), desc=f"{dataset_name}, {count} candidates",
                       unit="batch", leave=False)
        with torch.inference_mode():
            while True:
                start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                progress.update(1)
                prep_ms.append((time.perf_counter() - start) * 1000 / len(batch["rows"]))
                start = time.perf_counter()
                moved = move(batch, device)
                sync(device)
                transfer_ms.append((time.perf_counter() - start) * 1000 / len(batch["rows"]))
                timers.reset()
                start = time.perf_counter()
                forward(model, moved)
                sync(device)
                elapsed = (time.perf_counter() - start) * 1000 / len(batch["rows"])
                forward_ms.append(elapsed)
                components = timers.milliseconds()
                for name, value in components.items():
                    component_ms[name].append(value / len(batch["rows"]))
            progress.close()
        text = float(np.median(component_ms["text_encoder"]))
        vision = float(np.median(component_ms["vision_encoder"]))
        fusion_head = (float(np.median(component_ms["fusion"])) +
                       float(np.median(component_ms["option_head"])))
        row = {"dataset": dataset_name, "candidate_count": count, "n": len(subset),
               "batch_size": batch_size, "device": str(device),
               "preprocess_p50_ms_per_example": float(np.median(prep_ms)),
               "transfer_p50_ms_per_example": float(np.median(transfer_ms)),
               "text_encoder_p50_ms_per_example": text,
               "image_encoder_p50_ms_per_example": vision,
               "fusion_head_p50_ms_per_example": fusion_head,
               "other_forward_p50_ms_per_example": max(0.0, float(np.median(forward_ms)) -
                                                       text - vision - fusion_head),
               "forward_p50_ms_per_example": float(np.median(forward_ms)),
               "forward_p95_ms_per_example": float(np.percentile(forward_ms, 95)),
               "examples_per_second_forward_only": 1000 / float(np.median(forward_ms))}
        rows_out.append(row)
    timers.close()
    return rows_out


def write_outputs(rows, output_dir: Path, checkpoint: str, manifest: str):
    if not rows:
        raise ValueError("No candidate-count measurements were collected")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_scaling.json").write_text(json.dumps({
        "checkpoint": str(Path(checkpoint).resolve()), "manifest": str(Path(manifest).resolve()),
        "python": platform.python_version(), "torch": torch.__version__, "results": rows,
        "timing_note": "CUDA module timings use events; preprocessing includes DataLoader collation/image decode; warmup excluded.",
    }, indent=2))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    import csv
    with (output_dir / "candidate_scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted({row["dataset"] for row in rows})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4.3 * len(datasets)),
                             squeeze=False, constrained_layout=True)
    for index, dataset in enumerate(datasets):
        selected = sorted((row for row in rows if row["dataset"] == dataset),
                          key=lambda row: row["candidate_count"])
        x = [row["candidate_count"] for row in selected]
        axes[index, 0].plot(x, [row["forward_p50_ms_per_example"] for row in selected], marker="o")
        axes[index, 0].set(title=dataset, xlabel="Candidate count",
                           ylabel="Forward p50 (ms/example)")
        bottom = np.zeros(len(selected))
        for key, label in (("text_encoder_p50_ms_per_example", "Text encoder"),
                            ("image_encoder_p50_ms_per_example", "Image encoder"),
                            ("fusion_head_p50_ms_per_example", "Fusion + head"),
                            ("other_forward_p50_ms_per_example", "Other forward")):
            values = np.asarray([row[key] for row in selected])
            axes[index, 1].bar(x, values, bottom=bottom, width=.7, label=label)
            bottom += values
        axes[index, 1].set(title=dataset, xlabel="Candidate count",
                           ylabel="Component time (ms/example)")
        axes[index, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "candidate_scaling.png", dpi=220)
    fig.savefig(output_dir / "candidate_scaling.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "models/laya-med/best.pt"))
    parser.add_argument("--manifest")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Maximum matched test examples per dataset")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiment_results/latency")
    args = parser.parse_args()
    if (any(count < 2 for count in args.candidate_counts) or args.batch_size < 1
            or args.max_examples < 1):
        parser.error("Candidate counts must be >=2; batch size and sample cap must be positive")
    device = torch_device(args.device)
    model, tokenizer, saved, processor, manifest, splits = load_bundle(
        args.checkpoint, device, args.manifest)
    datasets = args.datasets or sorted({row["dataset"] for row in splits["test"]})
    rows = []
    for dataset in datasets:
        source = [row for row in splits["test"] if row["dataset"] == dataset][:args.max_examples]
        if not source:
            raise ValueError(f"No test rows for {dataset}")
        rows.extend(benchmark(model, tokenizer, processor, saved, source, dataset,
                              sorted(set(args.candidate_counts)), args.batch_size,
                              device, args.output_dir))
    write_outputs(rows, args.output_dir, args.checkpoint, args.manifest or saved["manifest"])
    print(f"Wrote candidate scaling assets to {args.output_dir}")


if __name__ == "__main__":
    main()
