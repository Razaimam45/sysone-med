"""Load the saved model summaries for the notebook's three landscape renderers."""

import csv
import math
from pathlib import Path
from statistics import mean


# Explicit modes avoid mixing the two batch-size runs in Laya's results.csv.
# Laya-Med denotes the vision-LoRA implementation used by the paper.
RESULT_SOURCES = {
    "MedGemma-AR": ("medgemma_ar", "conventional_llm"),
    "MedGemma-LL": ("medgemma_ll", "native_generate"),
    "Jev": ("jev", "jev_api"),
    "MedJev": ("medjev", "medjev"),
    "Laya": ("laya", "non_ar_decision_b16"),
    "MedGemma-Head": ("medgemma_direct_headonly", "medgemma_direct_headonly"),
    "MedGemma-Adapt": ("medgemma_direct_lora", "medgemma_direct_lora"),
    "Laya-Med": ("laya_med_vl", "laya_med_calibrated"),
}
DATASET_NAMES = {
    "pubmedqa": "PubMedQA",
    "med_qa": "MedQA",
    "medmcqa": "MedMCQA",
    "vqarad": "VQA-RAD",
    "slake": "SLAKE",
    "path_vqa": "PathVQA",
    "pmc_vqa": "PMC-VQA",
}
TEXT_TASKS = ("pubmedqa", "med_qa", "medmcqa")
TEXT_ONLY = {"Jev", "MedJev", "Laya"}
METRIC_COLUMNS = {
    "accuracy": ("accuracy", 100.0),
    "latency": ("p50_latency", 1.0),  # Already milliseconds in results.csv.
    "ece": ("ece", 100.0),
    "nll": ("nll", 1.0),
}


def load_paper_results():
    """Return metric -> dataset/group -> model, in the renderers' original units.

    Averages equally weight supported datasets. Latency averages are means of
    dataset medians, not a pooled median. Unsupported image tasks remain None.
    Each call rereads the CSVs so rerunning a cell picks up refreshed results.
    """
    root = Path(__file__).resolve().parents[1]
    metrics = {
        metric: {name: dict.fromkeys(RESULT_SOURCES) for name in DATASET_NAMES.values()}
        for metric in METRIC_COLUMNS
    }
    for model, (folder, mode) in RESULT_SOURCES.items():
        path = root / "results" / folder / "results.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["mode"] == mode]
        by_task = {row["task"]: row for row in rows}
        expected = set(TEXT_TASKS) if model in TEXT_ONLY else set(DATASET_NAMES)
        if len(by_task) != len(rows) or set(by_task) != expected:
            raise ValueError(f"{path}: expected one {mode!r} row per supported dataset")
        for task, row in by_task.items():
            for metric, (column, scale) in METRIC_COLUMNS.items():
                value = float(row[column])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path}: invalid {column} for {task}: {row[column]!r}")
                if metric in ("accuracy", "ece") and value > 1:
                    raise ValueError(f"{path}: {column} must be a fraction, not a percentage")
                if metric == "latency" and value == 0:
                    raise ValueError(f"{path}: latency must be positive for the log-scale plot")
                metrics[metric][DATASET_NAMES[task]][model] = value * scale

    groups = {
        "Text Average": [DATASET_NAMES[task] for task in TEXT_TASKS],
        "Image Average": [name for task, name in DATASET_NAMES.items() if task not in TEXT_TASKS],
        "Overall Average": list(DATASET_NAMES.values()),
    }
    for datasets in metrics.values():
        for group, members in groups.items():
            datasets[group] = {}
            for model in RESULT_SOURCES:
                values = [datasets[name][model] for name in members if datasets[name][model] is not None]
                datasets[group][model] = mean(values) if values else None
    return metrics


def quality_panel_config(data, dataset_order):
    """Fit the existing ECE/NLL panels to their measured values, with clean ticks."""
    def axis(values):
        low, high = min(values), max(values)
        padding = max((high - low) * 0.08, high * 0.03, 0.001)
        lower, upper = max(0.0, low - padding), high + padding
        raw_step = (upper - lower) / 4
        magnitude = 10 ** math.floor(math.log10(raw_step))
        step = next(factor * magnitude for factor in (1, 2, 2.5, 5, 10)
                    if factor * magnitude >= raw_step)
        ticks = [float(f"{i * step:.10g}")
                 for i in range(math.ceil(lower / step), math.floor(upper / step) + 1)]
        return (lower, upper), ticks

    panels = []
    for dataset in dataset_order:
        available = [(ece, nll) for ece, nll in data[dataset]
                     if ece is not None and nll is not None]
        xr, xt = axis([ece for ece, _ in available])
        yr, yt = axis([nll for _, nll in available])
        panels.append((dataset, xr, xt, yr, yt))
    return panels
