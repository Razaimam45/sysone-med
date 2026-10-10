#!/usr/bin/env python3
"""Evaluate a trained MedGemma-Direct checkpoint on shared held-out splits."""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "src"), str(ROOT / "training"), str(ROOT / "scripts")])
from laya_med.data import MedicalDecisionDataset
from laya_med.medgemma_direct import MedGemmaDirectCollator
from med_jevify.decision_data import file_hash, load_manifest
from med_jevify.metrics import latency_summary
from train import asset_fingerprint, resolve_asset
from train_medgemma_direct import (forward_batch, load_direct_model,
                                   load_trainable_state_dict, select_dtype)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize(device)


def prepare_image_ablation_rows(rows, ablation: str, seed: int = 42):
    if ablation == "none":
        return rows
    copied = []
    for row in rows:
        state = row.get("state", "")
        normalized = (dict(state) if isinstance(state, dict)
                      else {"text": state, "image": None})
        copied.append(dict(row, state=normalized))
    if ablation == "no-images":
        for row in copied:
            row["state"]["image"] = None
        return copied
    if ablation != "shuffle-images":
        raise ValueError(f"Unknown image ablation: {ablation}")
    rng = random.Random(seed)
    for dataset_name in sorted({row["dataset"] for row in copied}):
        selected = [row for row in copied if row["dataset"] == dataset_name and
                    row["state"].get("image")]
        if not selected:
            continue
        images = list(dict.fromkeys(row["state"]["image"] for row in selected))
        if len(images) == 1:
            raise ValueError(f"{dataset_name}: need at least two distinct images to shuffle")
        rng.shuffle(images)
        mapping = dict(zip(images, images[1:] + images[:1]))
        for row in selected:
            row["state"]["image"] = mapping[row["state"]["image"]]
    return copied


def probabilities(logits, temperature):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def summarize_probabilities(probabilities_by_row, labels):
    if not probabilities_by_row or len(probabilities_by_row) != len(labels):
        raise ValueError("Nonempty probability rows and matching labels are required")
    probabilities_by_row = [np.asarray(row, dtype=np.float64) for row in probabilities_by_row]
    for probs, label in zip(probabilities_by_row, labels):
        if probs.ndim != 1 or not len(probs) or not np.isfinite(probs).all():
            raise ValueError("Each probability row must be a finite nonempty vector")
        if not 0 <= label < len(probs) or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
            raise ValueError("Probability rows must be normalized and contain their target class")

    correct = np.asarray([int(probs.argmax() == label)
                          for probs, label in zip(probabilities_by_row, labels)], dtype=float)
    confidence = np.asarray([probs.max() for probs in probabilities_by_row])
    ece_value = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        mask = ((confidence > edges[index]) & (confidence <= edges[index + 1]))
        if mask.any():
            ece_value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())

    class_errors = []
    for class_index in range(max(len(probs) for probs in probabilities_by_row)):
        included = [index for index, probs in enumerate(probabilities_by_row)
                    if class_index < len(probs)]
        class_confidence = np.asarray([probabilities_by_row[index][class_index] for index in included])
        class_correct = np.asarray([int(labels[index] == class_index) for index in included], dtype=float)
        class_error = 0.0
        for bin_index in range(15):
            mask = ((class_confidence > edges[bin_index]) &
                    (class_confidence <= edges[bin_index + 1]))
            if mask.any():
                class_error += mask.mean() * abs(class_confidence[mask].mean() - class_correct[mask].mean())
        class_errors.append(class_error)

    losses, briers, divergences = [], [], []
    for probs, label in zip(probabilities_by_row, labels):
        target_probability = max(float(probs[label]), 1e-9)
        losses.append(-np.log(target_probability))
        briers.append(float(np.square(probs - np.eye(1, len(probs), label)[0]).sum()))
        midpoint = 0.5 * probs.copy()
        midpoint[label] += 0.5
        divergences.append(float(0.5 * np.log(1.0 / midpoint[label]) +
                                0.5 * np.sum(probs * np.log(np.clip(probs / midpoint, 1e-9, None)))))
    return {
        "n": len(labels), "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()), "ece": float(ece_value),
        "classwise_ece": float(np.mean(class_errors)), "nll": float(np.mean(losses)),
        "brier": float(np.mean(briers)), "jsd": float(np.mean(divergences)),
    }


def summarize_records(records, temperature=1.0):
    groups = {}
    for row in records:
        groups.setdefault(row["dataset"], []).append(row)
    per_dataset = {}
    for dataset, rows in groups.items():
        probs = [probabilities(row["logits"], temperature) for row in rows]
        labels = [row["label"] for row in rows]
        per_dataset[dataset] = summarize_probabilities(probs, labels)
    return {
        "per_dataset": per_dataset,
        "macro_accuracy": float(np.mean([row["accuracy"] for row in per_dataset.values()])),
        "macro_nll": float(np.mean([row["nll"] for row in per_dataset.values()])),
    }


def result_rows(records, durations, temperature, calibration_n, timestamp):
    rows = []
    for dataset in sorted({row["dataset"] for row in records}):
        selected = [row for row in records if row["dataset"] == dataset]
        selected_durations = [duration for row, duration in zip(records, durations)
                              if row["dataset"] == dataset]
        labels = [row["label"] for row in selected]
        p50, p95, throughput = latency_summary(selected_durations)
        for mode, scale in (("medgemma_direct_raw", 1.0),
                            ("medgemma_direct_calibrated", temperature)):
            probs = [probabilities(row["logits"], scale) for row in selected]
            metrics = summarize_probabilities(probs, labels)
            rows.append({
                "model": "google/medgemma-4b-it-direct", "architecture": "AR",
                "task": dataset, "mode": mode, "accuracy": metrics["accuracy"],
                "mean_confidence": metrics["mean_confidence"], "ece": metrics["ece"],
                "classwise_ece": metrics["classwise_ece"], "nll": metrics["nll"],
                "brier": metrics["brier"], "p50_latency": p50, "p95_latency": p95,
                "throughput": throughput, "n": len(labels), "temperature": scale,
                "calibration_n": 0 if mode.endswith("_raw") else calibration_n,
                "jsd": metrics["jsd"], "mean_label_mass": None,
                "invalid_predictions": 0,
                "probability_basis": "softmax over direct MedGemma hidden-state candidate scores",
                "timing": "synchronized forward latency; pipeline throughput includes preprocessing; loading excluded",
                "timestamp": timestamp,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "models/medgemma-direct/best.pt"))
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=["test", "calibration"], default="test")
    parser.add_argument("--ablation", choices=["none", "no-images", "shuffle-images"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", help="Default: all datasets with held-out test cases")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--append-results", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.max_examples is not None and args.max_examples <= 0):
        parser.error("Batch size and example cap must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_family") not in {"MedGemma-Direct", "MedGemma-Direct-HeadOnly"}:
        raise ValueError("Checkpoint is not a MedGemma-Direct run")
    calibration = checkpoint.get("calibration")
    if not calibration:
        raise ValueError("Checkpoint has no held-out calibration temperature")
    temperature = float(calibration["temperature"])
    saved = checkpoint["args"]
    manifest_path = args.manifest or saved["manifest"]
    if file_hash(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the training manifest")
    manifest, splits = load_manifest(manifest_path, verify_images=True)
    if manifest["encoding"]["max_length"] != saved["max_length"]:
        raise ValueError("Prepared manifest token budget differs from the training run")
    names = args.datasets or sorted({row["dataset"] for row in splits[args.split]})
    rows = []
    for name in names:
        selected = [row for row in splits[args.split] if row["dataset"] == name]
        if not selected:
            raise ValueError(f"No {args.split} partition for requested dataset {name}")
        rows.extend(selected[:args.max_examples] if args.max_examples else selected)

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    model_path = resolve_asset(saved["medgemma_model"], saved.get("revision"))
    assets = asset_fingerprint(model_path)
    if assets["files"] != checkpoint["assets"]["medgemma"]["files"]:
        raise ValueError("MedGemma assets differ from the training checkpoint")
    dtype = select_dtype(args.dtype, device)
    if device.type != "cuda" and dtype != torch.float32:
        parser.error("Low-precision MedGemma-Direct evaluation currently requires CUDA")

    is_head_only = checkpoint.get("model_family") == "MedGemma-Direct-HeadOnly"
    if is_head_only and args.ablation != "none":
        parser.error("Image ablations require MedGemma-LoRA; head-only checkpoints use cached image features")
    if args.ablation != "none" and saved.get("no_images"):
        parser.error("Image ablations require a multimodal checkpoint")
    processor, model = load_direct_model(model_path, device, saved["lora_rank"], dtype,
                                         saved["head_size"], lora=not is_head_only)
    load_trainable_state_dict(model, checkpoint["trainable_model"])
    model.eval()

    if args.ablation != "none":
        rows = prepare_image_ablation_rows(rows, args.ablation, args.seed)

    collator = MedGemmaDirectCollator(processor, saved["max_length"])
    loader = DataLoader(MedicalDecisionDataset(rows), batch_size=args.batch_size, collate_fn=collator)

    records, durations = [], []
    amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode():
        first_batch = next(iter(loader))
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            forward_batch(model, first_batch, device)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for batch in tqdm(loader, desc=f"{args.split}/{args.ablation}", unit="batch"):
            before = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                logits = forward_batch(model, batch, device)
            synchronize(device)
            elapsed = time.perf_counter() - before
            durations.extend([elapsed / len(batch["rows"])] * len(batch["rows"]))
            for scores, row in zip(logits, batch["rows"]):
                records.append({"id": row.id, "dataset": row.dataset, "label": row.label,
                                "options": list(row.options), "logits": scores.float().cpu().tolist(),
                                "latency_seconds": elapsed / len(batch["rows"]),
                                "temperature": temperature, "ablation": args.ablation})
        synchronize(device)
        pipeline_seconds = time.perf_counter() - start

    timestamp = datetime.now(timezone.utc).isoformat()
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "manifest_sha256": checkpoint["manifest_sha256"],
        "split": args.split,
        "ablation": args.ablation, "seed": args.seed,
        "datasets": names, "batch_size": args.batch_size, "max_examples": args.max_examples,
        "partial": args.max_examples is not None, "device": str(device), "precision": str(dtype),
        "examples": len(records), "temperature": temperature,
        "raw": summarize_records(records), "calibrated": summarize_records(records, temperature),
        "timing": {
            "definition": "FP32/BF16/FP16 synchronized forward latency; pipeline throughput includes preprocessing and transfers; loading excluded",
            "forward_ms_per_example_p50": float(np.percentile(durations, 50) * 1000),
            "forward_ms_per_example_p95": float(np.percentile(durations, 95) * 1000),
            "pipeline_examples_per_second": len(records) / pipeline_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
        "timestamp": timestamp,
    }
    output = args.output or Path(args.checkpoint).parent / f"evaluation-medgemma-direct-{args.split}-{args.ablation}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    output.with_suffix(".predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    if args.append_results:
        results_dir = args.append_results
        result_path = results_dir / "results.json"
        existing = json.loads(result_path.read_text()) if result_path.exists() else []
        from make_report import write_report
        write_report(existing + result_rows(records, durations, temperature,
                                             calibration["examples"], timestamp), results_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
