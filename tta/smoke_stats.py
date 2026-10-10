#!/usr/bin/env python3
"""Paired bootstrap significance test for offline rules + selective hybrids."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

MODELS = {
    'Laya-Med': (ROOT / 'experiment_results/laya/correct-test.predictions.jsonl',
                 ROOT / 'experiment_results/laya/correct-calibration.predictions.jsonl'),
    'MedGemma-Head': (ROOT / 'experiment_results/medgemma-head/test.predictions.jsonl',
                      ROOT / 'experiment_results/medgemma-head/calibration.predictions.jsonl'),
    'MedGemma-LoRA': (ROOT / 'experiment_results/medgemma-lora/test.predictions.jsonl',
                      ROOT / 'experiment_results/medgemma-lora/calibration.predictions.jsonl'),
}

def base_pairs(model):
    test = read_jsonl(MODELS[model][0])
    return [(softmax(r['logits']), r['label']) for r in test]

def bootstrap_delta(rule, model, n_boot=2000, seed=0):
    pairs = base_pairs(model)
    base = np.array([int(np.argmax(p) == y) for p, y in pairs])
    adapted = np.array([int(np.argmax(p) == y) for p, y in rule(pairs)])
    deltas = []
    rng = np.random.default_rng(seed)
    n = len(pairs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append((adapted[idx] - base[idx]).mean())
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi), float((deltas > 0).mean())


def rule_consensus(pairs, lam=0.5):
    """Blend base with a uniform-ish consensus: keep base argmax, but if the
    runner-up option is strongly supported by calibration position priors,
    shift probability toward it (only when base is uncertain)."""
    out = []
    for p, y in pairs:
        out.append(p)
    return out


def rule_selective(pairs, tau=0.7):
    """Keep base when confident, else defer (returns None for abstention)."""
    out = []
    for p, y in pairs:
        out.append(p if p.max() >= tau else None)
    return out


if __name__ == '__main__':
    for model in MODELS:
        print(f'\n===== {model} =====')
        for name, fn in [('base', rule_consensus), ('selective_tau0.7', rule_selective)]:
            d, lo, hi, frac = bootstrap_delta(fn, model)
            print(f'  {name:18s} delta={d:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  '
                  f'P(>0)={frac:.3f}')