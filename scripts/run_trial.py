#!/usr/bin/env python3
"""Run the MedJevify trial (model execution is intentionally explicit)."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Support direct execution from a source checkout, including IDE run buttons.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))  # make_report.py lives here

from med_jevify import JEVIFY_COMMIT
from med_jevify.calibration import TemperatureScaler, fit_temperature
from med_jevify.evaluate import (conventional_generate, constrained_generate, label_logits, native_generate,
                                 label_logits_batch, run_conventional, run_constrained, run_native,
                                 run_probability_backend, summarize, timed_batch_call,
                                 timed_call)
from med_jevify.non_ar import LayaAdapter
from med_jevify.decision_data import load_manifest, file_hash
from med_jevify.upstream import verify_jevify


MODEL_DIR = {
    # AR generation / native scoring baselines are keyed by model + mode so each
    # lands in its own results/<baseline>/ directory with identical columns.
    ("google/gemma-3-4b-it", "conventional_llm"): "medgemma_ar",
    ("google/medgemma-4b-it", "conventional_llm"): "medgemma_ar",
    ("kushalpatil/jevify-gemma4-e4b", "conventional_llm"): "medgemma_ar",
    ("google/gemma-3-4b-it", "native_generate"): "medgemma_ll",
    ("google/medgemma-4b-it", "native_generate"): "medgemma_ll",
    ("kushalpatil/jevify-gemma4-e4b", "native_generate"): "medgemma_ll",
    ("google/medgemma-4b-it", "constrained_generate"): "medgemma_constrained",
    ("google/gemma-3-4b-it", "constrained_generate"): "gemma_constrained",
    ("convaiinnovations/laya", "non_ar_decision"): "laya",
    ("convaiinnovations/laya", "calibrated_non_ar_decision"): "laya",
    ("convaiinnovations/laya", "non_ar_decision_b16"): "laya",
    ("convaiinnovations/laya", "calibrated_non_ar_decision_b16"): "laya",
    ("laya-med", "laya_med_raw"): "laya_med",
    ("laya-med", "laya_med_calibrated"): "laya_med",
    ("laya-med-vision-lora", "laya_med_raw"): "laya_med",
    ("laya-med-vision-lora", "laya_med_calibrated"): "laya_med",
}


def model_dir(model: str, mode: str) -> str:
    """Map a (model, mode) pair to its results/<baseline>/ directory name."""
    return MODEL_DIR.get((model, mode), model.replace("/", "_"))


def persist_results(records, output, run_timestamp):
    """Write each record to its own results/<baseline>/ subdirectory.

    Each baseline keeps its own results.json + report.md/csv with the same
    column schema, so cross-baseline analysis is a simple directory walk.
    """
    from make_report import write_report
    output = Path(output)
    grouped: dict[str, list[dict]] = {}
    for row in records:
        baseline = model_dir(row.get("model", "unknown"), row.get("mode", ""))
        grouped.setdefault(baseline, []).append(dict(row, timestamp=run_timestamp))
    for baseline, rows in grouped.items():
        # If the caller already pointed at results/<baseline>/, don't nest again.
        sub = output if output.name == baseline else output / baseline
        result_path = sub / "results.json"
        existing = json.loads(result_path.read_text()) if result_path.exists() else []
        # Preserve legacy rows (no timestamp) and drop only rows from the same run.
        existing = [row for row in existing if row.get("timestamp") != run_timestamp]
        write_report(existing + rows, sub)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["google/gemma-3-4b-it", "google/medgemma-4b-it", "kushalpatil/jevify-gemma4-e4b"])
    parser.add_argument("--modes", nargs="+", choices=["conventional_llm", "native_generate", "constrained_generate", "raw_jevify", "calibrated_jevify", "non_ar_decision", "calibrated_non_ar_decision"], default=["conventional_llm", "native_generate", "raw_jevify", "calibrated_jevify", "non_ar_decision", "calibrated_non_ar_decision"])
    parser.add_argument("--tasks", nargs="+",
                        choices=["pubmedqa", "vqarad", "med_qa", "medmcqa", "slake", "path_vqa", "pmc_vqa"],
                        default=["pubmedqa", "vqarad"])
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data/prepared/manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--non-ar-model", default="convaiinnovations/laya")
    parser.add_argument("--non-ar-fast", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--jevify-commit", default=JEVIFY_COMMIT)
    args = parser.parse_args(argv)
    if args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    verify_jevify(args.jevify_commit)
    manifest, prepared = load_manifest(args.manifest, verify_images=True)
    datasets = {}
    for task in args.tasks:
        calibration = [r for r in prepared['calibration'] if r['dataset'] == task]
        test = [r for r in prepared['test'] if r['dataset'] == task]
        if not calibration or not test:
            parser.error(f"Manifest lacks calibration/test data for {task}")
        datasets[f'{task}_calibration'] = calibration[:args.max_examples]
        datasets[task] = test[:args.max_examples]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    details_dir = output / "details"
    details_dir.mkdir(exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).isoformat()
    settings = vars(args) | {"jevify_commit": JEVIFY_COMMIT, "timestamp": run_timestamp}
    (details_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
    from importlib.metadata import version
    settings.update({
        "evaluation_version": 4,
        "manifest_sha256": file_hash(args.manifest),
        "package_versions": {name: version(name) for name in ["torch", "transformers", "tokenizers", "jevify", "numpy"]},
        "scoring": "pinned train/eval.py: log(matched top-64 label mass + 1e-9)",
        "temperature_grid": {"min": 0.3, "max": 30.0, "count": 120},
        "confidence_definition": "maximum class probability (upstream eval), not API entropy confidence",
        "timing": "cold prompt cache per example, synchronized, one warm-up per mode/task; model loading excluded",
        "calibrated_timing": "same raw inference latency + measured CPU temperature transform; fit excluded",
        "native_generation": "same upstream prompt; greedy, unconstrained max_new_tokens=1; invalid tokens count as wrong",
        "conventional_generation": "same upstream prompt; greedy, unconstrained max_new_tokens=32; parsed label prefixes count as answers",
        "native_probability_basis": "unrounded full-vocabulary logits from generate; allowed labels plus invalid outcome",
        "native_confidence": "mass of emitted label (or individual token probability for invalid output)",
        "native_ece": "15 bins of emitted-answer confidence versus actual correctness",
        "native_distribution_metrics": "NLL/Brier/JSD/classwise ECE over allowed labels plus non-label mass",
    "batching": "AR inference and calibration always use batch size 1; --batch-size applies only to non-AR scoring",
        "batch_latency": "non-AR batched wall time divided across examples in the batch for per-example summaries",
        "models_note": "raw_jevify means no post-hoc scaling; upstream Jevify checkpoint already includes trained LoRA",
        "non_ar_model": args.non_ar_model,
        "non_ar_batching": "Only identical question/choice schemas share a batch; requested batch size is an upper bound; heterogeneous schemas run separately",
        "status": "running",
        "splits": {key: [row["id"] for row in rows] for key, rows in datasets.items()},
    })
    (details_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
    records, predictions, calibration_report = [], [], []
    if any(mode.startswith("non_ar") or mode.startswith("calibrated_non_ar") for mode in args.modes):
        non_ar = LayaAdapter(args.non_ar_model, args.device, args.non_ar_fast)
        for task in args.tasks:
            examples = datasets[task]
            if any(isinstance(row.get("state"), dict) and row["state"].get("image") for row in examples):
                print(f"{args.non_ar_model}: skipping {task}; non-AR backend is text-only", flush=True)
                continue
            calibration_rows = datasets[f"{task}_calibration"]
            temperature = None
            calibration_n = 0
            if "calibrated_non_ar_decision" in args.modes:
                print(f"{args.non_ar_model}: calibrating {task} on {len(calibration_rows)} examples...", flush=True)
                calibration_probs, _ = non_ar.score(calibration_rows, args.batch_size)
                calibration_logits = np.log(np.clip(np.asarray(calibration_probs), 1e-9, 1.0))
                temperature = fit_temperature(
                    calibration_logits,
                    [int(row["label"]) for row in calibration_rows],
                )
                calibration_n = len(calibration_rows)
                print(f"{args.non_ar_model}: {task} fitted non-AR temperature = {temperature.temperature:.6f}", flush=True)
            non_ar_specs = []
            if "non_ar_decision" in args.modes:
                non_ar_specs.append(("non_ar_decision", 1, None))
                if args.batch_size > 1:
                    non_ar_specs.append((f"non_ar_decision_b{args.batch_size}", args.batch_size, None))
            if "calibrated_non_ar_decision" in args.modes:
                non_ar_specs.append(("calibrated_non_ar_decision", 1, temperature))
                if args.batch_size > 1:
                    non_ar_specs.append((f"calibrated_non_ar_decision_b{args.batch_size}", args.batch_size, temperature))
            for mode, batch_size, scaler in non_ar_specs:
                print(f"{args.non_ar_model}: {task} / {mode} ({len(examples)} examples, batch {batch_size})...", flush=True)
                record, details = run_probability_backend(
                    examples, non_ar, args.non_ar_model, task, mode, batch_size, scaler)
                record.calibration_n = calibration_n if mode.startswith("calibrated_") else 0
                records.append(record.as_dict())
                predictions.extend(dict(row, model=args.non_ar_model, task=task, mode=mode) for row in details)
            persist_results(records, output, run_timestamp)
    jevify_modes = {"conventional_llm", "native_generate", "constrained_generate", "raw_jevify", "calibrated_jevify"}
    for model_id in (args.models if jevify_modes.intersection(args.modes) else []):
        import jevify
        print(f"Loading {model_id}...", flush=True)
        from transformers import AutoModelForCausalLM, AutoProcessor
        # Qwen2.5-VL and InternVL3 use non-standard config classes that
        # AutoModelForCausalLM cannot auto-resolve under transformers 5.x.
        # The jevify TransformersBackend hard-codes AutoModelForCausalLM in its
        # __init__, so we temporarily route from_pretrained through the registry
        # for the duration of the load; Gemma/MedGemma fall through unchanged.
        from med_jevify.backbones import resolve_backbone
        _orig_from_pretrained = AutoModelForCausalLM.from_pretrained

        def _patched_from_pretrained(model_id, **kw):
            try:
                spec = resolve_backbone(model_id)
                if spec.model_cls is not None:
                    return spec.model_cls.from_pretrained(model_id, **kw)
            except ValueError:
                pass
            return _orig_from_pretrained(model_id, **kw)

        AutoModelForCausalLM.from_pretrained = _patched_from_pretrained
        try:
            jev = jevify.Jevify.from_transformers(
                model_id,
                device=None if args.device == "auto" else args.device,
                dtype=args.dtype,
                no_think=model_id not in {"google/gemma-3-4b-it", "google/medgemma-4b-it"},
            )
        finally:
            AutoModelForCausalLM.from_pretrained = _orig_from_pretrained
        for task in args.tasks:
            examples = datasets[task]
            calibration_rows = datasets[f"{task}_calibration"]
            temperature = TemperatureScaler()
            calibration_n = 0
            if "calibrated_jevify" in args.modes:
                print(f"{model_id}: calibrating {task} on {len(calibration_rows)} examples...", flush=True)
                calibration_logits, calibration_details = [], []
                calibration_batch_size = 1
                for start in range(0, len(calibration_rows), calibration_batch_size):
                    batch = calibration_rows[start:start + calibration_batch_size]
                    if len(batch) == 1:
                        scored, _ = timed_call(jev, label_logits, batch[0])
                        scored = [scored]
                    else:
                        scored, _ = timed_batch_call(jev, label_logits_batch, batch)
                    for row, (logits, mass) in zip(batch, scored):
                        calibration_logits.append(logits)
                        calibration_details.append({"id": row["id"], "label": int(row["label"]),
                                                    "logits": logits.tolist(), "label_mass": mass})
                temperature = fit_temperature(calibration_logits, [int(row["label"]) for row in calibration_rows])
                calibration_n = len(calibration_rows)
                calibration_report.append({"model": model_id, "task": task,
                                           "temperature": temperature.temperature,
                                           "examples": calibration_details})
                print(f"{model_id}: {task} fitted temperature = {temperature.temperature:.6f}", flush=True)

            if "conventional_llm" in args.modes:
                print(f"{model_id}: {task} / conventional_llm ({len(examples)} examples)...", flush=True)
                timed_call(jev, conventional_generate, examples[0])  # untimed in the reported sample
                record, details = run_conventional(examples, jev, model_id, task, 1)
                records.append(record.as_dict())
                predictions.extend(dict(row, model=model_id, task=task, mode="conventional_llm") for row in details)

            if "native_generate" in args.modes:
                print(f"{model_id}: {task} / native_generate ({len(examples)} examples)...", flush=True)
                timed_call(jev, native_generate, examples[0])  # untimed in the reported sample
                record, details = run_native(examples, jev, model_id, task, 1)
                records.append(record.as_dict())
                predictions.extend(dict(row, model=model_id, task=task, mode="native_generate") for row in details)

            if "constrained_generate" in args.modes:
                print(f"{model_id}: {task} / constrained_generate ({len(examples)} examples)...", flush=True)
                timed_call(jev, constrained_generate, examples[0])  # untimed warm-up
                record, details = run_constrained(examples, jev, model_id, task)
                records.append(record.as_dict())
                predictions.extend(dict(row, model=model_id, task=task, mode="constrained_generate")
                                   for row in details)

            mode_specs = []
            if "raw_jevify" in args.modes:
                mode_specs.append(("raw_jevify", 1))
            if "calibrated_jevify" in args.modes:
                mode_specs.append(("calibrated_jevify", 1))
            for mode, batch_size in mode_specs:
                print(f"{model_id}: {task} / {mode} ({len(examples)} examples, batch {batch_size})...", flush=True)
                timed_call(jev, label_logits, examples[0])
                probabilities, durations, masses = [], [], []
                for start in range(0, len(examples), batch_size):
                    batch = examples[start:start + batch_size]
                    if len(batch) == 1:
                        scored, elapsed = timed_call(jev, label_logits, batch[0])
                        scored = [scored]
                    else:
                        scored, elapsed = timed_batch_call(jev, label_logits_batch, batch)
                    for row, (logits, mass) in zip(batch, scored):
                        begin = time.perf_counter()
                        scaler = TemperatureScaler() if mode.startswith("raw_") else temperature
                        probs = scaler.transform(logits[None, :])[0]
                        duration = elapsed / len(batch) + time.perf_counter() - begin
                        probabilities.append(probs)
                        durations.append(duration)
                        masses.append(mass)
                        predictions.append({"model": model_id, "task": task, "mode": mode,
                                            "id": row["id"], "label": int(row["label"]),
                                            "options": list(row["options"]),
                                            "prediction": int(probs.argmax()), "probabilities": probs.tolist(),
                                            "logits": logits.tolist(), "label_mass": mass,
                                            "temperature": scaler.temperature, "latency_seconds": duration})
                record = summarize(model_id, task, mode, probabilities,
                                   [int(row["label"]) for row in examples], durations)
                record.temperature = scaler.temperature
                record.calibration_n = calibration_n if mode.startswith("calibrated_") else 0
                record.mean_label_mass = float(np.mean(masses))
                if mode.startswith("calibrated_"):
                    record.timing += "; shared raw inference + CPU transform, fit excluded"
                records.append(record.as_dict())
            # Save after each task so completed work survives later failures.
            persist_results(records, output, run_timestamp)
            (details_dir / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
            (details_dir / "calibration.json").write_text(json.dumps(calibration_report, indent=2))
        # Release references before loading the next model.
        del jev
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    settings["status"] = "complete"
    (details_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
    persist_results(records, output, run_timestamp)
    print(f"Finished. Results saved to {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
