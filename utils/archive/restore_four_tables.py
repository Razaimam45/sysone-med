#!/usr/bin/env python3
"""Restore the four-table paper layout from current evaluation artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PREDICTIONS = ROOT / "results/details/predictions.jsonl"
LAYA_PREDICTIONS = ROOT / "models/laya-med/evaluation-best-50.predictions.jsonl"
LAYA_EVAL = ROOT / "models/laya-med/evaluation-best-50.json"
OUT = ROOT / "latex/tables"
DATASETS = ["pubmedqa", "med_qa", "medmcqa", "vqarad", "slake", "path_vqa", "pmc_vqa"]
TEXT_TASKS = ["pubmedqa", "med_qa", "medmcqa"]
LABELS = {"pubmedqa": "PubMedQA", "med_qa": "MedQA", "medmcqa": "MedMCQA",
          "vqarad": "VQA-RAD", "slake": "SLAKE", "path_vqa": "PathVQA", "pmc_vqa": "PMC-VQA"}
MODELS = {"gemma": "google/gemma-3-4b-it", "medgemma": "google/medgemma-4b-it",
          "ours": "laya-med"}


def latest_rows():
    from merge_results import merge_results, latest_rows as _latest
    return _latest(merge_results(ROOT / "results"))


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def get(rows, model, task, mode):
    return rows.get((model, task, mode))


def mean_metric(rows, model, mode, metric, tasks=DATASETS):
    values = [get(rows, model, task, mode) for task in tasks]
    values = [row[metric] for row in values if row is not None and row.get(metric) is not None]
    return float(np.mean(values)) if values else None


def percent(value, digits=1):
    return "--" if value is None else f"{100 * value:.{digits}f}"


def decimal(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def start_table(caption, label, columns):
    return [
        r"% Preamble: \usepackage{booktabs,graphicx,amssymb} and \usepackage[table]{xcolor}",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4.2pt}", r"\renewcommand{\arraystretch}{1.14}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
    ]


def end_table(lines):
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def table1(rows):
    methods = [
        ("Gemma-3", "AR VLM", "Autoregressive generation", MODELS["gemma"], "conventional_llm"),
        ("MedGemma", "AR VLM", "Autoregressive generation", MODELS["medgemma"], "conventional_llm"),
        ("MedGemma + candidate likelihood", "AR VLM", "Answer-label likelihood scoring", MODELS["medgemma"], "raw_jevify"),
        ("MedGemma-Direct", "AR VLM", "Learned direct candidate scoring", None, None),
        (r"\textbf{Laya-Med (ours)}", r"\textbf{NAR multimodal}", r"\textbf{Direct decision modeling}", MODELS["ours"], "laya_med_raw"),
    ]
    values_by_method = []
    for _, _, _, model, mode in methods:
        values = [get(rows, model, dataset, mode)["accuracy"]
                  if model is not None and get(rows, model, dataset, mode) else None for dataset in DATASETS]
        observed = [value for value in values if value is not None]
        values.append(float(np.mean(observed)) if observed else None)
        values_by_method.append(values)
    best = [max((values[col] for values in values_by_method if values[col] is not None), default=None)
            for col in range(len(DATASETS) + 1)]
    lines = start_table(
        "Top-1 accuracy (\\%) on the seven available medical QA/VQA test subsets (n=50 per dataset). Text-1/2/3 are PubMedQA/MedQA/MedMCQA; MM-1/2/3/4 are VQA-RAD/SLAKE/PathVQA/PMC-VQA. Macro average is unweighted. Candidate likelihood is next-token answer-label mass, not full-option sequence likelihood. Dashes indicate unmeasured methods.",
        "tab:main-predictive-performance", "lll" + "c" * 8)
    lines += [r"\rowcolor{gray!13}\multicolumn{3}{c}{Method} & \multicolumn{3}{c}{Text-only} & \multicolumn{4}{c}{Image + text} & Macro " + r"\\",
              r"\cmidrule(lr){1-3}\cmidrule(lr){4-6}\cmidrule(lr){7-10}\cmidrule(lr){11-11}",
              r"Method & Backbone & Prediction paradigm & Text-1 & Text-2 & Text-3 & MM-1 & MM-2 & MM-3 & MM-4 & Acc. $\uparrow$ " + r"\\",
              r"\midrule"]
    for index, ((name, backbone, paradigm, _, _), values) in enumerate(zip(methods, values_by_method)):
        cells = []
        for col, value in enumerate(values):
            cell = percent(value)
            if value is not None and value == best[col]:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        color = r"\rowcolor{teal!9}" if index == len(methods) - 1 else (r"\rowcolor{gray!6}" if index == 3 else "")
        lines.append(color + " " + " & ".join([name, backbone, paradigm, *cells]) + r" \\")
    return end_table(lines)


def table2(rows):
    methods = [
        ("MedGemma-Generate", MODELS["medgemma"], "conventional_llm", "conventional_llm", "AR; sequential"),
        ("MedGemma-Likelihood", MODELS["medgemma"], "calibrated_jevify", "raw_jevify", "AR; no decode"),
        ("MedGemma-Direct", None, None, None, "AR; learned head"),
        (r"\textbf{Laya-Med (ours)}", MODELS["ours"], "laya_med_calibrated", "laya_med_raw", r"\textbf{NAR; no decode}"),
    ]
    lines = start_table(
        "Macro averages across seven 50-example dataset subsets. Accuracy/ECE are percentages; Brier/NLL are unitless. Likelihood and Laya-Med calibration metrics use held-out temperature scaling; generation preserves invalid-answer outcomes. Probability supports differ by paradigm. Latency/throughput are means of dataset p50/examples-per-second. Memory was not collected under a common protocol.",
        "tab:accuracy-calibration-efficiency", "llccccccc")
    lines += [r"\rowcolor{gray!13}Method & Backbone / decoding & Acc. $\uparrow$ & ECE $\downarrow$ & Brier $\downarrow$ & NLL $\downarrow$ & p50 (ms) $\downarrow$ & Ex./s $\uparrow$ & Memory $\downarrow$ " + r"\\", r"\midrule"]
    for index, (label, model, quality_mode, speed_mode, paradigm) in enumerate(methods):
        if model is None:
            vals = [None] * 6
        else:
            vals = [mean_metric(rows, model, quality_mode, metric) for metric in ("accuracy", "ece", "brier", "nll")]
            vals += [mean_metric(rows, model, speed_mode, "p50_latency"),
                     mean_metric(rows, model, speed_mode, "throughput")]
        cells = [percent(vals[0]), percent(vals[1]), decimal(vals[2]), decimal(vals[3]),
                 decimal(vals[4], 1), decimal(vals[5], 1), "--"]
        color = r"\rowcolor{teal!9}" if model == MODELS["ours"] else (r"\rowcolor{gray!6}" if model is None else (r"\rowcolor{gray!5}" if index % 2 else ""))
        lines.append(color + " " + " & ".join([label, f"{paradigm}; MedGemma VLM" if model != MODELS["ours"] else f"{paradigm}; PubMedBERT + BiomedCLIP", *cells]) + r" \\")
    return end_table(lines)


def load_selective_predictions(rows):
    grouped = {}
    for row in load_jsonl(SUMMARY_PREDICTIONS):
        grouped.setdefault((row.get("model"), row.get("task"), row.get("mode")), []).append(row)
    if LAYA_PREDICTIONS.exists() and LAYA_EVAL.exists():
        temperature = float(json.loads(LAYA_EVAL.read_text())["temperature"])
        for row in load_jsonl(LAYA_PREDICTIONS):
            logits = np.asarray(row["logits"], dtype=float)
            for mode, values in (("laya_med_raw", logits), ("laya_med_calibrated", logits / temperature)):
                exp = np.exp(values - values.max()); probs = exp / exp.sum()
                grouped.setdefault((MODELS["ours"], row["dataset"], mode), []).append({
                    "label": row["label"], "prediction": int(probs.argmax()),
                    "probabilities": probs.tolist(), "confidence": float(probs.max())})
    return grouped


def selective_summary(items):
    if not items:
        return None
    confidence = np.asarray([float(row.get("confidence", max(row["probabilities"]))) for row in items])
    correct = np.asarray([int(row.get("prediction", int(np.argmax(row["probabilities"]))) == int(row["label"])) for row in items], dtype=float)
    order = np.argsort(-confidence, kind="stable")
    coverage = np.arange(1, len(order) + 1) / len(order)
    risk = 1 - np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    aurc = float(np.trapezoid(np.r_[0.0, risk], np.r_[0.0, coverage]))
    return {"aurc": aurc,
            "acc90": float(correct[order[:max(1, math.ceil(.90 * len(order)))]] .mean()),
            "acc75": float(correct[order[:max(1, math.ceil(.75 * len(order)))]] .mean())}


def table3(rows, predictions):
    methods = [("MedGemma-Generate", MODELS["medgemma"], "conventional_llm", "conventional_llm", DATASETS),
               ("MedGemma-Likelihood (cal.)", MODELS["medgemma"], "calibrated_jevify", "calibrated_jevify", DATASETS),
               ("MedGemma-Direct", None, None, None, DATASETS),
               ("Laya-Med, uncalibrated", MODELS["ours"], "laya_med_raw", "laya_med_raw", DATASETS),
               (r"\textbf{Laya-Med + temperature scaling}", MODELS["ours"], "laya_med_calibrated", "laya_med_calibrated", DATASETS)]
    lines = start_table(
        "Calibration and selective prediction on available n=50/dataset prediction records. ECE is 15-bin fixed-width; adaptive ECE is 10 equal-frequency bins. AURC is macro dataset area under risk--coverage. Coverage accuracy retains the highest-confidence 90\\% or 75\\%. Dashes mark absent predictions.",
        "tab:calibration-selective", "lccccccc")
    lines += [r"\rowcolor{gray!13}Method & ECE $\downarrow$ & Adaptive ECE $\downarrow$ & NLL $\downarrow$ & Brier $\downarrow$ & AURC $\downarrow$ & Acc.@90\% $\uparrow$ & Acc.@75\% $\uparrow$ " + r"\\", r"\midrule"]
    for index, (label, model, mode, pred_mode, datasets) in enumerate(methods):
        if model is None:
            lines.append(r"\rowcolor{gray!6}" + " " + " & ".join([label] + ["--"] * 7) + r" \\")
            continue
        ece_val = mean_metric(rows, model, mode, "ece", datasets)
        nll_val = mean_metric(rows, model, mode, "nll", datasets)
        brier_val = mean_metric(rows, model, mode, "brier", datasets)
        selective = [selective_summary(predictions.get((model, d, pred_mode), [])) for d in datasets]
        selective = [val for val in selective if val is not None]
        aurc = float(np.mean([v["aurc"] for v in selective])) if selective else None
        acc90 = float(np.mean([v["acc90"] for v in selective])) if selective else None
        acc75 = float(np.mean([v["acc75"] for v in selective])) if selective else None
        adaptive = None
        # Adaptive ECE is derived from prediction probabilities and computed dataset-wise below.
        adaptive_values = []
        for dataset in datasets:
            items = predictions.get((model, dataset, pred_mode), [])
            if items:
                conf = np.asarray([float(item.get("confidence", max(item["probabilities"]))) for item in items])
                corr = np.asarray([int(item.get("prediction", int(np.argmax(item["probabilities"]))) == item["label"]) for item in items], dtype=float)
                order = np.argsort(conf, kind="stable")
                chunks = np.array_split(order, min(10, len(order)))
                adaptive_values.append(float(sum(len(chunk)/len(order)*abs(corr[chunk].mean()-conf[chunk].mean()) for chunk in chunks if len(chunk))))
        if adaptive_values: adaptive = float(np.mean(adaptive_values))
        values = [percent(ece_val), percent(adaptive), decimal(nll_val), decimal(brier_val), decimal(aurc,3), percent(acc90), percent(acc75)]
        shade = r"\rowcolor{teal!9}" if "temperature scaling" in label else (r"\rowcolor{gray!5}" if index % 2 else "")
        lines.append(shade + " " + " & ".join([label, *values]) + r" \\")
    return end_table(lines)


def table4(rows):
    variants = [("Text-only", "PubMedBERT", "--", r"\checkmark", r"\checkmark", r"\checkmark", None),
                ("Vision-only", "--", "BiomedCLIP", r"\checkmark", r"\checkmark", r"\checkmark", None),
                ("No LoRA", "PubMedBERT", "BiomedCLIP", r"\checkmark", r"$\times$", r"\checkmark", None),
                ("No temperature scaling", "PubMedBERT", "BiomedCLIP", r"\checkmark", r"\checkmark", r"$\times$", "laya_med_raw"),
                ("Generic encoders", "BERT", "CLIP", r"\checkmark", r"\checkmark", r"\checkmark", None),
                (r"\textbf{Laya-Med (full)}", r"\textbf{PubMedBERT}", r"\textbf{BiomedCLIP}", r"\textbf{\checkmark}", r"\textbf{\checkmark}", r"\textbf{\checkmark}", "laya_med_calibrated")]
    lines = start_table(
        "Laya-Med component ablations. Only full-model calibrated evaluation and the no-temperature-scaling diagnostic are measured; other components have not been ablated and are left blank. Metrics are macro means over seven n=50 test subsets.",
        "tab:layamed-ablations", "lcccccccc")
    lines += [r"\rowcolor{gray!13}Variant & Text encoder & Vision encoder & Direct head & LoRA & Temp. scaling & Accuracy $\uparrow$ & ECE $\downarrow$ & p50 latency (ms) $\downarrow$ " + r"\\", r"\midrule"]
    for i, (label, text, vision, head, lora, scaled, mode) in enumerate(variants):
        if mode:
            acc = mean_metric(rows, MODELS["ours"], mode, "accuracy")
            ece_val = mean_metric(rows, MODELS["ours"], mode, "ece")
            latency = mean_metric(rows, MODELS["ours"], mode, "p50_latency")
            metrics = [percent(acc), percent(ece_val), decimal(latency, 1)]
        else:
            metrics = ["--"] * 3
        shade = r"\rowcolor{teal!9}" if "full" in label else (r"\rowcolor{gray!5}" if i % 2 else "")
        lines.append(shade + " " + " & ".join([label, text, vision, head, lora, scaled, *metrics]) + r" \\")
    return end_table(lines)


def main():
    rows = latest_rows()
    predictions = load_selective_predictions(rows)
    tables = {"table1_main_performance.tex": table1(rows),
              "table2_tradeoff.tex": table2(rows),
              "table3_calibration_selective.tex": table3(rows, predictions),
              "table4_layamed_ablations.tex": table4(rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.tex"):
        old.unlink()
    for name, content in tables.items():
        (OUT / name).write_text(content)
    (OUT / "tables.tex").write_text("\n".join(r"\input{latex/tables/" + Path(name).stem + "}" for name in tables) + "\n")
    print(f"Restored the previous four tables in {OUT}")


if __name__ == "__main__":
    main()
