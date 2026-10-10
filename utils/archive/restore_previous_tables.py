#!/usr/bin/env python3
"""Restore the previous six-table paper set from current evaluation summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex/tables"
DATASETS = ["pubmedqa", "vqarad", "med_qa", "medmcqa", "slake", "path_vqa", "pmc_vqa"]
LABELS = {"pubmedqa": "PubMedQA", "vqarad": "VQA-RAD", "med_qa": "MedQA",
          "medmcqa": "MedMCQA", "slake": "SLAKE", "path_vqa": "PathVQA", "pmc_vqa": "PMC-VQA"}


def latest_rows():
    from merge_results import merge_results, latest_rows as _latest
    return _latest(merge_results(ROOT / "results"))
TEXT_TASKS = ["pubmedqa", "med_qa", "medmcqa"]
MODELS = {"gemma": "google/gemma-3-4b-it", "medgemma": "google/medgemma-4b-it",
          "jevify": "kushalpatil/jevify-gemma4-e4b", "laya": "convaiinnovations/laya", "ours": "laya-med"}


def load_rows():
    source = json.loads(RESULTS.read_text())
    latest = {}
    for row in sorted(source, key=lambda item: item.get("timestamp") or ""):
        latest[(row["model"], row["task"], row["mode"])] = row
    return latest


def get(rows, model, dataset, mode):
    return rows.get((model, dataset, mode))


def avg(rows, model, mode, metric, datasets=DATASETS):
    values = [get(rows, model, dataset, mode) for dataset in datasets]
    values = [row[metric] for row in values if row is not None and row.get(metric) is not None]
    return float(np.mean(values)) if values else None


def pct(value, digits=1):
    return "--" if value is None else f"{100 * value:.{digits}f}"


def num(value, digits=1):
    return "--" if value is None else f"{value:.{digits}f}"


def base(caption, label, spec, header):
    return [
        r"% Requires \usepackage{booktabs,graphicx,xcolor}",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4.2pt}", r"\renewcommand{\arraystretch}{1.13}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule", r"\rowcolor{gray!13}" + header,
        r"\midrule",
    ]


def end(lines):
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    return "\n".join(lines)


def main_accuracy(rows):
    methods = [
        ("Gemma-3", MODELS["gemma"], "conventional_llm", "AR VLM", "Autoregressive generation"),
        ("Gemma-3", MODELS["gemma"], "raw_jevify", "AR VLM", "Answer-label scoring"),
        ("MedGemma", MODELS["medgemma"], "conventional_llm", "AR VLM", "Autoregressive generation"),
        ("MedGemma", MODELS["medgemma"], "raw_jevify", "AR VLM", "Answer-label scoring"),
        ("Jevify-Gemma", MODELS["jevify"], "conventional_llm", "AR VLM", "Autoregressive generation"),
        ("Jevify-Gemma", MODELS["jevify"], "raw_jevify", "AR VLM", "Answer-label scoring"),
        ("Generic Laya", MODELS["laya"], "non_ar_decision", "NAR text", "Direct option scoring"),
        (r"\textbf{Laya-Med (ours)}", MODELS["ours"], "laya_med_raw", r"\textbf{NAR multimodal}", r"\textbf{Direct decision model}"),
    ]
    values_by_method = []
    for _, model, mode, _, _ in methods:
        tasks = TEXT_TASKS if model == MODELS["laya"] else DATASETS
        values = [get(rows, model, dataset, mode)["accuracy"]
                  if get(rows, model, dataset, mode) else None for dataset in DATASETS]
        values.append(float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None)
        values_by_method.append(values)
    best = [max((v[col] for v in values_by_method if v[col] is not None), default=None)
            for col in range(len(DATASETS) + 1)]
    lines = base(
            "Top-1 accuracy (\\%) across seven medical QA/VQA test sets (latest available 50-example run). Macro averages are unweighted across available datasets. Generic Laya is text-only; dashes denote unavailable image-task scores.",
            "tab:main-accuracy", "llcccccccc",
            r"Method & Paradigm & PubMedQA & VQA-RAD & MedQA & MedMCQA & SLAKE & PathVQA & PMC-VQA & Macro $\uparrow$ \\")
    for index, ((label, _, _, _, paradigm), values) in enumerate(zip(methods, values_by_method)):
        cells = []
        for column, value in enumerate(values):
            cell = pct(value)
            if value is not None and value == best[column]:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        shade = r"\rowcolor{teal!9}" if index == len(methods) - 1 else (r"\rowcolor{gray!5}" if index % 2 else "")
        lines.append(shade + " " + " & ".join([label, paradigm, *cells]) + r" \\")
    return end(lines)


def calibrated_ece(rows):
    methods = [("Gemma-3", MODELS["gemma"], "calibrated_jevify", DATASETS),
               ("MedGemma", MODELS["medgemma"], "calibrated_jevify", DATASETS),
               ("Jevify-Gemma", MODELS["jevify"], "calibrated_jevify", DATASETS),
               ("Generic Laya", MODELS["laya"], "calibrated_non_ar_decision", TEXT_TASKS),
               (r"\textbf{Laya-Med (ours)}", MODELS["ours"], "laya_med_calibrated", DATASETS)]
    lines = base(
            "Expected calibration error (ECE, \\%, lower is better) after held-out temperature scaling. Values are per dataset; macro average is unweighted across available datasets. Generic Laya is text-only.",
            "tab:calibrated-ece", "lcccccccc",
            r"Method & " + " & ".join(LABELS[d] for d in DATASETS) + r" & Macro $\downarrow$ \\")
    for i, (label, model, mode, tasks) in enumerate(methods):
        vals = [get(rows, model, d, mode)["ece"] if get(rows, model, d, mode) else None for d in DATASETS]
        vals.append(avg(rows, model, mode, "ece", tasks))
        shade = r"\rowcolor{teal!9}" if model == MODELS["ours"] else (r"\rowcolor{gray!5}" if i % 2 else "")
        lines.append(shade + " " + " & ".join([label, *[pct(v) for v in vals]]) + r" \\")
    return end(lines)


def calibration_gain(rows):
    methods = [("Gemma-3", MODELS["gemma"], "raw_jevify", "calibrated_jevify", DATASETS),
               ("MedGemma", MODELS["medgemma"], "raw_jevify", "calibrated_jevify", DATASETS),
               ("Jevify-Gemma", MODELS["jevify"], "raw_jevify", "calibrated_jevify", DATASETS),
               ("Generic Laya", MODELS["laya"], "non_ar_decision", "calibrated_non_ar_decision", TEXT_TASKS),
               (r"\textbf{Laya-Med (ours)}", MODELS["ours"], "laya_med_raw", "laya_med_calibrated", DATASETS)]
    lines = [r"% Requires \usepackage{booktabs,xcolor}", r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{Macro ECE before and after held-out temperature scaling. Reduction is $100\times(1-\mathrm{ECE}_{cal}/\mathrm{ECE}_{raw})$; positive values indicate improvement. Generic Laya uses text-only datasets.}",
                 r"\label{tab:calibration-gain}", r"\begin{tabular}{lcccc}", r"\toprule",
                 r"\rowcolor{gray!13} Method & Datasets & Raw ECE $\downarrow$ & Cal. ECE $\downarrow$ & Reduction (\%) $\uparrow$ \\", r"\midrule"]
    for i, (label, model, raw_mode, cal_mode, tasks) in enumerate(methods):
        raw, cal = avg(rows, model, raw_mode, "ece", tasks), avg(rows, model, cal_mode, "ece", tasks)
        reduction = (1 - cal / raw) * 100 if raw and cal is not None else None
        shade = r"\rowcolor{teal!9}" if model == MODELS["ours"] else (r"\rowcolor{gray!5}" if i % 2 else "")
        lines.append(shade + f" {label} & {len(tasks)} & {pct(raw)} & {pct(cal)} & {num(reduction)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def efficiency(rows):
    methods = [("Gemma-3-Generate", MODELS["gemma"], "conventional_llm", "AR generation"),
               ("Gemma-3-LabelScore", MODELS["gemma"], "raw_jevify", "AR label scoring"),
               ("MedGemma-Generate", MODELS["medgemma"], "conventional_llm", "AR generation"),
               ("MedGemma-LabelScore", MODELS["medgemma"], "raw_jevify", "AR label scoring"),
               ("Jevify-Gemma-Generate", MODELS["jevify"], "conventional_llm", "AR generation"),
               ("Jevify-Gemma-LabelScore", MODELS["jevify"], "raw_jevify", "AR label scoring"),
               ("Generic Laya (B1)", MODELS["laya"], "non_ar_decision", "NAR API decision"),
               (r"\textbf{Laya-Med (ours)}", MODELS["ours"], "laya_med_raw", "NAR local decision")]
    lines = base(
        "Inference efficiency on 50-example test subsets. Latency is the unweighted macro mean of per-dataset p50/p95, model loading excluded. Generic Laya uses remote API timing and is not directly hardware-comparable to local model timings.",
        "tab:inference-efficiency", "llcc",
        r"Method & Evaluation path & p50 (ms) $\downarrow$ & p95 (ms) $\downarrow$ \\")
    for i, (label, model, mode, path) in enumerate(methods):
        p50, p95 = avg(rows, model, mode, "p50_latency"), avg(rows, model, mode, "p95_latency")
        shade = r"\rowcolor{teal!9}" if model == MODELS["ours"] else (r"\rowcolor{gray!5}" if i % 2 else "")
        lines.append(shade + f" {label} & {path} & {num(p50)} & {num(p95)} " + r"\\")
    return end(lines)


def native_ablation(rows):
    methods = [("Gemma-3", MODELS["gemma"]), ("MedGemma", MODELS["medgemma"]),
               ("Jevify-Gemma", MODELS["jevify"])]
    lines = base(
        "Accuracy change when replacing conventional free-form generation with greedy one-token generation, $\\Delta$ accuracy in percentage points. Positive values indicate improvement from avoiding long-form answer parsing; this is a diagnostic decoding ablation, not the primary model comparison.",
        "tab:native-generation-ablation", "lcccccccc",
        r"Model & PubMedQA & VQA-RAD & MedQA & MedMCQA & SLAKE & PathVQA & PMC-VQA & Macro $\uparrow$ \\")
    for i, (label, model) in enumerate(methods):
        deltas=[]
        for d in DATASETS:
            gen, native = get(rows, model, d, "conventional_llm"), get(rows, model, d, "native_generate")
            deltas.append((native["accuracy"]-gen["accuracy"])*100 if gen and native else None)
        available=[v for v in deltas if v is not None]
        deltas.append(float(np.mean(available)) if available else None)
        lines.append((r"\rowcolor{gray!5}" if i % 2 else "") + label + " & " + " & ".join(num(v) for v in deltas) + r" \\")
    return end(lines)


def batching_speedup(rows):
    lines=[r"% Requires \usepackage{booktabs,xcolor}",r"\begin{table}[t]",r"\centering",r"\small",
        r"\caption{Generic Laya throughput by requested batch size on text-only tasks. Speedup is throughput at batch 16 divided by batch 1.}",
        r"\label{tab:laya-batching}",r"\begin{tabular}{lccc}",r"\toprule",
        r"\rowcolor{gray!13} Dataset & Throughput B1 $\uparrow$ & Throughput B16 $\uparrow$ & Speedup $\times$ \\",r"\midrule"]
    ratios=[]
    for i,d in enumerate(TEXT_TASKS):
        b1=get(rows,MODELS["laya"],d,"non_ar_decision")
        b16=get(rows,MODELS["laya"],d,"non_ar_decision_b16")
        a=b1["throughput"] if b1 else None; b=b16["throughput"] if b16 else None
        ratio=b/a if a and b else None
        if ratio is not None: ratios.append(ratio)
        lines.append((r"\rowcolor{gray!5}" if i%2 else "")+f"{LABELS[d]} & {num(a)} & {num(b)} & {num(ratio,2)} " + r"\\")
    macro_ratio=float(np.mean(ratios)) if ratios else None
    lines += [r"\midrule",f"\\textbf{{Macro mean}} & -- & -- & \\textbf{{{num(macro_ratio,2)}}} " + r"\\",
              r"\bottomrule",r"\end{tabular}",r"\end{table}",""]
    return "\n".join(lines)


def main():
    rows=load_rows()
    outputs={
        "main_accuracy.tex": main_accuracy(rows),
        "ece_calibrated.tex": calibrated_ece(rows),
        "calibration_gain.tex": calibration_gain(rows),
        "efficiency.tex": efficiency(rows),
        "ablation_native_generate.tex": native_ablation(rows),
        "batching_speedup.tex": batching_speedup(rows),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    for old in OUT.glob("*.tex"):
        old.unlink()
    for name,content in outputs.items():
        (OUT/name).write_text(content)
    (OUT/"all_tables.tex").write_text("\n".join(r"\input{tables/"+name[:-4]+"}" for name in outputs)+"\n")
    print(f"Restored {len(outputs)} previous tables to {OUT}")


if __name__ == "__main__":
    main()
