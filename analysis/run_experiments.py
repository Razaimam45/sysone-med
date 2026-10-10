#!/usr/bin/env python3
"""Export reproducible tables and figures from per-example prediction ledgers.

Prediction files are JSONL emitted by the project evaluators. This script never
truncates them: paired analyses require identical IDs and labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def limit_per_dataset(records: list[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
    if maximum is None:
        return records
    if maximum <= 0:
        raise ValueError("max_per_dataset must be positive")
    counts: dict[str, int] = {}
    selected = []
    for record in records:
        dataset = str(record.get("dataset", record.get("task")))
        if counts.get(dataset, 0) >= maximum:
            continue
        selected.append(record)
        counts[dataset] = counts.get(dataset, 0) + 1
    return selected


def index(records: list[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for record in records:
        dataset = record.get("dataset", record.get("task"))
        if dataset is None or "id" not in record or "label" not in record:
            raise ValueError(f"{name}: every row needs dataset/task, id, and label")
        key = (str(dataset), str(record["id"]))
        if key in output:
            raise ValueError(f"{name}: duplicate ID {key}")
        output[key] = record
    return output


def probs(record: dict[str, Any], temperature: float | None = None) -> np.ndarray:
    if "probabilities" in record:
        values = np.asarray(record["probabilities"], dtype=np.float64)
    elif "logits" in record:
        values = np.asarray(record["logits"], dtype=np.float64)
        temperature = float(record.get("temperature", 1.0)) if temperature is None else temperature
        if temperature <= 0 or not np.isfinite(temperature):
            raise ValueError("temperature must be finite and positive")
        values /= temperature
        values -= values.max()
        values = np.exp(values)
        values /= values.sum()
    else:
        raise ValueError(f"{record.get('id')}: missing probabilities/logits")
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-5)):
        raise ValueError(f"{record.get('id')}: invalid probability vector")
    label = int(record["label"])
    if not 0 <= label < len(values):
        raise ValueError(f"{record.get('id')}: label outside probability vector")
    return values


def metric(record: dict[str, Any], temperature: float | None = None) -> dict[str, float]:
    values = probs(record, temperature)
    label = int(record["label"])
    one_hot = np.eye(1, len(values), label)[0]
    prediction = int(record.get("prediction", values.argmax()))
    return {"accuracy": float(prediction == label),
            "nll": float(-np.log(max(values[label], 1e-12))),
            "brier": float(np.square(values - one_hot).sum()),
            "confidence": float(record.get("confidence", values.max())),
            "prediction": prediction}


def paired_probability_vectors(left: dict[str, Any], right: dict[str, Any],
                               left_temp: float | None,
                               right_temp: float | None) -> tuple[np.ndarray, np.ndarray, bool]:
    left_probs, right_probs = probs(left, left_temp), probs(right, right_temp)
    left_options = list(left.get("options", []))
    right_options = list(right.get("options", []))
    left_has_invalid = bool(left_options and left_options[-1] == "__invalid__")
    right_has_invalid = bool(right_options and right_options[-1] == "__invalid__")
    if left_has_invalid:
        left_options = left_options[:-1]
    if right_has_invalid:
        right_options = right_options[:-1]
    same_candidates = bool(left_options and right_options and left_options == right_options)
    if len(left_probs) == len(right_probs):
        return left_probs, right_probs, True
    if same_candidates and left_has_invalid and not right_has_invalid and len(left_probs) == len(right_probs) + 1:
        right_probs = np.append(right_probs, 0.0)
        return left_probs, right_probs, True
    if same_candidates and right_has_invalid and not left_has_invalid and len(right_probs) == len(left_probs) + 1:
        left_probs = np.append(left_probs, 0.0)
        return left_probs, right_probs, True
    return left_probs, right_probs, False


def require_same(reference: list[dict[str, Any]], candidate: list[dict[str, Any]],
                 ref_name: str, cand_name: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left, right = index(reference, ref_name), index(candidate, cand_name)
    if left.keys() != right.keys():
        raise ValueError(f"{ref_name} and {cand_name} have different IDs: "
                         f"missing={list(left.keys() - right.keys())[:4]}, "
                         f"extra={list(right.keys() - left.keys())[:4]}")
    pairs = []
    for key in sorted(left):
        if int(left[key]["label"]) != int(right[key]["label"]):
            raise ValueError(f"Label mismatch for {key}")
        left_options = [option for option in left[key].get("options", []) if option != "__invalid__"]
        right_options = [option for option in right[key].get("options", []) if option != "__invalid__"]
        if left_options and right_options and left_options != right_options:
            raise ValueError(f"Candidate order/content mismatch for {key}")
        pairs.append((left[key], right[key]))
    return pairs


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 replicates: int) -> tuple[float, float]:
    means = np.empty(replicates, dtype=np.float64)
    chunk_size = max(1, min(256, replicates))
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indices].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def paired_table(reference: list[dict[str, Any]], candidate: list[dict[str, Any]],
                 ref_name: str, cand_name: str, ref_temp: float, cand_temp: float,
                 replicates: int, seed: int) -> list[dict[str, Any]]:
    pairs = require_same(reference, candidate, ref_name, cand_name)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in pairs:
        grouped.setdefault(str(pair[0].get("dataset", pair[0].get("task"))), []).append(pair)
    grouped["__pooled__"] = pairs
    rng = np.random.default_rng(seed)
    output = []
    for dataset, group in grouped.items():
        row: dict[str, Any] = {"reference": ref_name, "candidate": cand_name,
                               "dataset": dataset, "n": len(group)}
        raw_dimensions_match = all(len(probs(a, ref_temp)) == len(probs(b, cand_temp))
                      for a, b in group)
        aligned = [paired_probability_vectors(a, b, ref_temp, cand_temp) for a, b in group]
        row["probability_dimensions_match"] = raw_dimensions_match
        row["brier_comparable"] = all(comparable for _, _, comparable in aligned)
        for key in ("accuracy", "nll", "brier"):
            if key == "brier" and not row["brier_comparable"]:
                row.update({"brier_reference": np.nan, "brier_candidate": np.nan,
                            "brier_delta": np.nan, "brier_ci_low": np.nan,
                            "brier_ci_high": np.nan})
                continue
            if key == "brier":
                values = []
                for (a, b), (left_probs, right_probs, _) in zip(group, aligned):
                    label = int(a["label"])
                    target = np.eye(1, len(left_probs), label)[0]
                    values.append((float(np.square(right_probs - target).sum()),
                                   float(np.square(left_probs - target).sum())))
                deltas = np.asarray([candidate_value - reference_value
                                     for candidate_value, reference_value in values])
            else:
                deltas = np.asarray([metric(b, cand_temp)[key] - metric(a, ref_temp)[key]
                                     for a, b in group])
            lo, hi = bootstrap_ci(deltas, rng, replicates)
            if key == "brier":
                reference_value = float(np.mean([pair[1] for pair in values]))
                candidate_value = float(np.mean([pair[0] for pair in values]))
            else:
                reference_value = float(np.mean([metric(a, ref_temp)[key] for a, _ in group]))
                candidate_value = float(np.mean([metric(b, cand_temp)[key] for _, b in group]))
            row.update({f"{key}_reference": reference_value,
                        f"{key}_candidate": candidate_value,
                        f"{key}_delta": float(deltas.mean()), f"{key}_ci_low": lo,
                        f"{key}_ci_high": hi})
        output.append(row)
    return output


def candidate_audit(predictions: list[dict[str, Any]], manifest_rows: list[dict[str, Any]],
                    model: str, temperature: float | None = None) -> list[dict[str, Any]]:
    expected = index(manifest_rows, "manifest")
    observed = index(predictions, model)
    rows = []
    for dataset in sorted({key[0] for key in observed}):
        selected = [(key, row) for key, row in observed.items() if key[0] == dataset]
        missing = label_bad = options_bad = vector_bad = extra_class = 0
        for key, record in selected:
            source = expected.get(key)
            if source is None:
                missing += 1
            else:
                label_bad += int(int(source["label"]) != int(record["label"]))
                if record.get("options") is not None:
                    options = list(source["options"])
                    seen = record["options"]
                    options_bad += int(seen != options and seen != options + ["__invalid__"])
            try:
                vector = probs(record, temperature)
                if source is not None:
                    allowed = len(source["options"])
                    extra_class += int(len(vector) not in (allowed, allowed + 1))
            except (ValueError, TypeError):
                vector_bad += 1
        rows.append({"model": model, "dataset": dataset, "n": len(selected),
                     "unknown_ids": missing, "label_mismatches": label_bad,
                     "option_mismatches": options_bad, "invalid_vectors": vector_bad,
                     "unexpected_class_count": extra_class})
    return rows


def candidate_rows(predictions: list[dict[str, Any]], manifest_rows: list[dict[str, Any]],
                   model: str) -> list[dict[str, Any]]:
    manifest = index(manifest_rows, "manifest")
    output = []
    for record in predictions:
        key = (str(record.get("dataset", record.get("task"))), str(record["id"]))
        source = manifest.get(key)
        options = list(record.get("options", source.get("options", []) if source else []))
        values = probs(record)
        prediction = int(record.get("prediction", values.argmax()))
        output.append({"model": model, "dataset": key[0], "id": key[1],
                       "candidate_options_json": json.dumps(options, ensure_ascii=True),
                       "label_index": int(record["label"]), "prediction_index": prediction,
                       "prediction_text": options[prediction] if prediction < len(options) else "__out_of_set__",
                       "probability_basis": record.get("probability_basis", "saved per-candidate scores"),
                       "temperature": record.get("temperature", 1.0)})
    return output


def visual_analysis(conditions: dict[str, list[dict[str, Any]]], temperatures: dict[str, float],
                    replicates: int, seed: int, output_dir: Path) -> list[dict[str, Any]]:
    baseline = conditions["correct_image"]
    for name in ("missing_image", "shuffled_image"):
        require_same(baseline, conditions[name], "correct_image", name)
    by_condition = {name: index(records, name) for name, records in conditions.items()}
    datasets = sorted({key[0] for key in by_condition["correct_image"]})
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in datasets:
        keys = [key for key in by_condition["correct_image"] if key[0] == dataset]
        base_records = [by_condition["correct_image"][key] for key in keys]
        base_scores = [metric(record, temperatures["correct_image"])["accuracy"]
                       for record in base_records]
        rows.append({"dataset": dataset, "condition": "correct_image", "n": len(keys),
                     "accuracy": float(np.mean(base_scores)), "delta": 0.0,
                     "ci_low": 0.0, "ci_high": 0.0})
        for condition in ("missing_image", "shuffled_image"):
            deltas = np.asarray([
                metric(by_condition[condition][key], temperatures[condition])["accuracy"] -
                metric(by_condition["correct_image"][key], temperatures["correct_image"])["accuracy"]
                for key in keys])
            lo, hi = bootstrap_ci(deltas, rng, replicates)
            rows.append({"dataset": dataset, "condition": condition, "n": len(keys),
                         "accuracy": float(np.mean([metric(by_condition[condition][key],
                                                              temperatures[condition])["accuracy"]
                                                    for key in keys])),
                         "delta": float(deltas.mean()), "ci_low": lo, "ci_high": hi})
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "visual_dependence.csv", rows)
    plot_grouped(rows, output_dir / "visual_dependence", ylabel="Accuracy (%)", scale=100)
    return rows


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    rank = int(math.ceil((len(scores) + 1) * (1 - alpha)))
    return float("inf") if rank > len(scores) else float(np.sort(scores)[rank - 1])


def conformal_mask(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return probabilities.max() - probabilities <= threshold


def posthoc_analysis(calibration: list[dict[str, Any]], test: list[dict[str, Any]],
                     temperature: float | None, alphas: list[float], taus: list[float],
                     output_dir: Path, replicates: int = 5000,
                     seed: int = 42) -> list[dict[str, Any]]:
    calib, testing = index(calibration, "calibration"), index(test, "test")
    if calib.keys() & testing.keys():
        raise ValueError("Calibration and test IDs overlap")
    rows = []
    curve = []
    rng = np.random.default_rng(seed)
    for dataset in sorted({key[0] for key in testing}):
        cal = [row for key, row in calib.items() if key[0] == dataset]
        test_rows = [row for key, row in testing.items() if key[0] == dataset]
        if not cal:
            raise ValueError(f"Missing calibration predictions for {dataset}")
        scores = np.asarray([probs(row, temperature).max() -
                             probs(row, temperature)[int(row["label"])] for row in cal])
        base = [metric(row, temperature)["accuracy"] for row in test_rows]
        base_acc_ci = bootstrap_ci(np.asarray(base), rng, replicates)
        full_coverage_ci = bootstrap_ci(np.ones(len(test_rows)), rng, replicates)
        rows.append({"dataset": dataset, "policy": "baseline", "parameter": "none",
                     "n": len(test_rows), "coverage": 1.0,
                     "accepted_accuracy": float(np.mean(base)), "empirical_label_coverage": np.nan,
                     "accepted_accuracy_ci_low": base_acc_ci[0],
                     "accepted_accuracy_ci_high": base_acc_ci[1],
                     "coverage_ci_low": full_coverage_ci[0], "coverage_ci_high": full_coverage_ci[1],
                     "mean_set_size": np.nan})
        for tau in taus:
            accepted = [row for row in test_rows if probs(row, temperature).max() >= tau]
            acc = float(np.mean([metric(row, temperature)["accuracy"] for row in accepted])) if accepted else np.nan
            accepted_accuracy_ci = (bootstrap_ci(np.asarray([metric(row, temperature)["accuracy"]
                                                             for row in accepted]), rng, replicates)
                                    if accepted else (np.nan, np.nan))
            coverage_values = np.asarray([int(probs(row, temperature).max() >= tau)
                                          for row in test_rows])
            coverage_ci = bootstrap_ci(coverage_values, rng, replicates)
            row = {"dataset": dataset, "policy": "routing", "parameter": tau,
                   "n": len(test_rows), "accepted_n": len(accepted),
                   "coverage": len(accepted) / len(test_rows), "accepted_accuracy": acc,
                                     "accepted_accuracy_ci_low": accepted_accuracy_ci[0],
                                     "accepted_accuracy_ci_high": accepted_accuracy_ci[1],
                                     "coverage_ci_low": coverage_ci[0], "coverage_ci_high": coverage_ci[1],
                   "empirical_label_coverage": np.nan, "mean_set_size": np.nan}
            rows.append(row)
            curve.append({**row, "risk": 1 - acc if accepted else np.nan})
        for alpha in alphas:
            threshold = conformal_quantile(scores, alpha)
            sizes, covered, correct = [], [], []
            for row in test_rows:
                vector = probs(row, temperature)
                mask = conformal_mask(vector, threshold)
                sizes.append(int(mask.sum()))
                covered.append(bool(mask[int(row["label"])]))
                if mask.sum() == 1:
                    correct.append(int(vector.argmax() == int(row["label"])))
            acc = float(np.mean(correct)) if correct else np.nan
            accepted_accuracy_ci = (bootstrap_ci(np.asarray(correct), rng, replicates)
                                    if correct else (np.nan, np.nan))
            singleton = np.asarray([int(size == 1) for size in sizes])
            coverage_ci = bootstrap_ci(singleton, rng, replicates)
            label_coverage_ci = bootstrap_ci(np.asarray(covered, dtype=float), rng, replicates)
            result = {"dataset": dataset, "policy": "conformal", "parameter": alpha,
                      "n": len(test_rows), "accepted_n": len(correct),
                      "coverage": len(correct) / len(test_rows), "accepted_accuracy": acc,
                      "accepted_accuracy_ci_low": accepted_accuracy_ci[0],
                      "accepted_accuracy_ci_high": accepted_accuracy_ci[1],
                      "coverage_ci_low": coverage_ci[0], "coverage_ci_high": coverage_ci[1],
                      "empirical_label_coverage": float(np.mean(covered)),
                      "empirical_label_coverage_ci_low": label_coverage_ci[0],
                      "empirical_label_coverage_ci_high": label_coverage_ci[1],
                      "mean_set_size": float(np.mean(sizes))}
            rows.append(result)
            curve.append({**result, "risk": 1 - acc if correct else np.nan})
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "posthoc_policy_results.csv", rows)
    save_csv(output_dir / "risk_coverage_points.csv", curve)
    plot_risk_coverage(curve, output_dir / "risk_coverage")
    return rows


def plot_grouped(rows: list[dict[str, Any]], prefix: Path, ylabel: str, scale: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted({row["dataset"] for row in rows})
    conditions = sorted({row["condition"] for row in rows})
    x = np.arange(len(datasets))
    width = .8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.2), 4.5), constrained_layout=True)
    for i, condition in enumerate(conditions):
        lookup = {row["dataset"]: row["accuracy"] * scale for row in rows
                  if row["condition"] == condition}
        ax.bar(x + (i - (len(conditions) - 1) / 2) * width, [lookup.get(d, np.nan) for d in datasets],
               width, label=condition.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, datasets, rotation=25, ha="right")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_risk_coverage(rows: list[dict[str, Any]], prefix: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted({row["dataset"] for row in rows})
    cols = min(4, len(datasets))
    fig, axes = plt.subplots(math.ceil(len(datasets) / cols), cols,
                             figsize=(4 * cols, 3.1 * math.ceil(len(datasets) / cols)),
                             squeeze=False, constrained_layout=True)
    for ax, dataset in zip(axes.flat, datasets):
        for policy in ("routing", "conformal"):
            points = sorted((r for r in rows if r["dataset"] == dataset and r["policy"] == policy
                             and np.isfinite(r["risk"])), key=lambda r: r["coverage"])
            if points:
                ax.plot([r["coverage"] for r in points], [r["risk"] for r in points],
                        marker="o", label=policy)
        ax.set(title=dataset, xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def latency_table(named_files: list[tuple[str, Path]],
                  maximum: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for model, path in named_files:
        groups: dict[tuple[str, int], list[float]] = {}
        for record in limit_per_dataset(load_jsonl(path), maximum):
            duration = record.get("latency_seconds", record.get("latency_s"))
            if duration is None:
                continue
            options = record.get("options")
            count = (sum(option != "__invalid__" for option in options) if options else
                     len(record.get("logits", record.get("probabilities", []))))
            dataset = str(record.get("dataset", record.get("task")))
            groups.setdefault((dataset, count), []).append(float(duration))
        for (dataset, count), values in sorted(groups.items()):
            rows.append({"model": model, "dataset": dataset, "candidate_count": count,
                         "n": len(values), "p50_ms": float(np.percentile(values, 50) * 1000),
                         "p95_ms": float(np.percentile(values, 95) * 1000)})
    return rows


def ece(records: list[dict[str, Any]], bins: int = 15) -> float:
    results = [metric(record) for record in records]
    confidence = np.asarray([row["confidence"] for row in results])
    correct = np.asarray([row["accuracy"] for row in results])
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for bin_index in range(bins):
        mask = ((confidence >= edges[bin_index]) & (confidence < edges[bin_index + 1])
                if bin_index < bins - 1 else
                (confidence >= edges[bin_index]) & (confidence <= edges[bin_index + 1]))
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def summary_table(named_files: list[tuple[str, Path]], replicates: int,
                  seed: int, maximum: int | None = None) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for model_name, path in named_files:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in limit_per_dataset(load_jsonl(path), maximum):
            dataset = str(record.get("dataset", record.get("task")))
            groups.setdefault(dataset, []).append(record)
        for dataset, records in sorted(groups.items()):
            metrics = [metric(record) for record in records]
            accuracies = np.asarray([row["accuracy"] for row in metrics])
            losses = np.asarray([row["nll"] for row in metrics])
            accuracy_ci = bootstrap_ci(accuracies, rng, replicates)
            nll_ci = bootstrap_ci(losses, rng, replicates)
            durations = [record.get("latency_seconds", record.get("latency_s"))
                         for record in records]
            durations = [float(value) for value in durations if value is not None]
            rows.append({"model": model_name, "dataset": dataset, "n": len(records),
                         "accuracy": float(accuracies.mean()),
                         "accuracy_ci_low": accuracy_ci[0], "accuracy_ci_high": accuracy_ci[1],
                         "ece": ece(records), "nll": float(losses.mean()),
                         "nll_ci_low": nll_ci[0], "nll_ci_high": nll_ci[1],
                         "brier": float(np.mean([row["brier"] for row in metrics])),
                         "p50_latency_ms": float(np.percentile(durations, 50) * 1000) if durations else np.nan,
                         "p95_latency_ms": float(np.percentile(durations, 95) * 1000) if durations else np.nan})
    return rows


def named_paths(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {value}")
        result.append((name, Path(path)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    audit = subs.add_parser("audit")
    audit.add_argument("--manifest-test", required=True)
    audit.add_argument("--predictions", nargs="+", required=True, metavar="NAME=JSONL")
    audit.add_argument("--max-per-dataset", type=int)
    audit.add_argument("--temperature", type=float)
    audit.add_argument("--output", default="experiment_results/candidate_audit.csv")
    audit.add_argument("--rows-output", default="experiment_results/candidate_rows.csv")
    compare = subs.add_parser("compare")
    compare.add_argument("--reference", required=True, metavar="NAME=JSONL")
    compare.add_argument("--candidate", required=True, metavar="NAME=JSONL")
    compare.add_argument("--max-per-dataset", type=int)
    compare.add_argument("--reference-temperature", type=float)
    compare.add_argument("--candidate-temperature", type=float)
    compare.add_argument("--replicates", type=int, default=5000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--output", default="experiment_results/paired_comparison.csv")
    visual = subs.add_parser("visual")
    visual.add_argument("--correct-image", required=True)
    visual.add_argument("--missing-image", required=True)
    visual.add_argument("--shuffled-image", required=True)
    for name in ("correct-image", "missing-image", "shuffled-image"):
        visual.add_argument(f"--{name}-temperature", type=float)
    visual.add_argument("--replicates", type=int, default=5000)
    visual.add_argument("--seed", type=int, default=42)
    visual.add_argument("--output-dir", default="experiment_results/visual")
    posthoc = subs.add_parser("posthoc")
    posthoc.add_argument("--calibration", required=True)
    posthoc.add_argument("--test", required=True)
    posthoc.add_argument("--temperature", type=float)
    posthoc.add_argument("--alphas", type=float, nargs="+", default=[.05, .1, .2, .3])
    posthoc.add_argument("--taus", type=float, nargs="+", default=np.linspace(.2, 1, 17).tolist())
    posthoc.add_argument("--replicates", type=int, default=5000)
    posthoc.add_argument("--seed", type=int, default=42)
    posthoc.add_argument("--output-dir", default="experiment_results/posthoc")
    latency = subs.add_parser("latency")
    latency.add_argument("--predictions", nargs="+", required=True, metavar="NAME=JSONL")
    latency.add_argument("--max-per-dataset", type=int)
    latency.add_argument("--output-dir", default="experiment_results/latency")
    summary = subs.add_parser("summary", help="per-model results with bootstrap intervals")
    summary.add_argument("--predictions", nargs="+", required=True, metavar="NAME=JSONL")
    summary.add_argument("--max-per-dataset", type=int)
    summary.add_argument("--replicates", type=int, default=5000)
    summary.add_argument("--seed", type=int, default=42)
    summary.add_argument("--output", default="experiment_results/main_results.csv")
    args = parser.parse_args()
    if args.command == "audit":
        manifests = load_jsonl(args.manifest_test)
        rows, row_details = [], []
        for name, path in named_paths(args.predictions):
            predictions = limit_per_dataset(load_jsonl(path), args.max_per_dataset)
            rows.extend(candidate_audit(predictions, manifests, name, args.temperature))
            row_details.extend(candidate_rows(predictions, manifests, name))
        save_csv(args.output, rows)
        save_csv(args.rows_output, row_details)
    elif args.command == "compare":
        ref_name, ref_path = named_paths([args.reference])[0]
        cand_name, cand_path = named_paths([args.candidate])[0]
        rows = paired_table(limit_per_dataset(load_jsonl(ref_path), args.max_per_dataset),
                    limit_per_dataset(load_jsonl(cand_path), args.max_per_dataset),
                    ref_name, cand_name,
                            args.reference_temperature, args.candidate_temperature,
                            args.replicates, args.seed)
        save_csv(args.output, rows)
    elif args.command == "visual":
        conditions = {"correct_image": load_jsonl(args.correct_image),
                      "missing_image": load_jsonl(args.missing_image),
                      "shuffled_image": load_jsonl(args.shuffled_image)}
        temps = {"correct_image": args.correct_image_temperature,
                 "missing_image": args.missing_image_temperature,
                 "shuffled_image": args.shuffled_image_temperature}
        visual_analysis(conditions, temps, args.replicates, args.seed, Path(args.output_dir))
    elif args.command == "posthoc":
        posthoc_analysis(load_jsonl(args.calibration), load_jsonl(args.test), args.temperature,
                         args.alphas, args.taus, Path(args.output_dir),
                         args.replicates, args.seed)
    elif args.command == "latency":
        rows = latency_table(named_paths(args.predictions), args.max_per_dataset)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_csv(out / "latency_by_candidate_count.csv", rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for model in sorted({row["model"] for row in rows}):
            points = sorted((row for row in rows if row["model"] == model),
                            key=lambda row: row["candidate_count"])
            ax.plot([r["candidate_count"] for r in points], [r["p50_ms"] for r in points],
                    marker="o", label=model)
        ax.set(xlabel="Candidate count", ylabel="p50 latency (ms)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(out / "latency_by_candidate_count.png", dpi=220)
        fig.savefig(out / "latency_by_candidate_count.pdf")
        plt.close(fig)
    else:
        save_csv(args.output, summary_table(named_paths(args.predictions),
                                            args.replicates, args.seed,
                                            args.max_per_dataset))


if __name__ == "__main__":
    main()
