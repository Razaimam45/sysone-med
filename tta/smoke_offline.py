#!/usr/bin/env python3
"""Offline search over post-hoc decision rules using saved predictions.

No GPU: reads test + calibration predictions already on disk and scores
candidate decision rules that need no new forward pass.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
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


def acc(pairs):
    if not pairs:
        return float('nan')
    return 100.0 * np.mean([int(np.argmax(p) == y) for p, y in pairs])


def evaluate(name, rule):
    print(f'\n===== {name} =====')
    for model, (test_p, cal_p) in MODELS.items():
        test = read_jsonl(test_p)
        cal = read_jsonl(cal_p)
        base = [(softmax(r['logits']), r['label']) for r in test]
        adapted = rule(test, cal, base)
        a0, a1 = acc(base), acc(adapted)
        changed = sum(int(np.argmax(p) != np.argmax(b[0])) for (p, _), b in zip(adapted, base))
        print(f'  {model:16s} base={a0:6.2f}  adapted={a1:6.2f}  delta={a1-a0:+6.2f} pp  '
              f'changed={changed:5d}/{len(base)}')


# ---------------------------------------------------------------- rules

def rule_position_debias(test, cal, base):
    """Subtract per-position log-prior estimated from calibration."""
    out = []
    by_n = defaultdict(list)
    for r in cal:
        by_n[(r['dataset'], len(r['options']))].append(r)
    prior = {}
    for key, rows in by_n.items():
        counts = np.zeros(key[1])
        for r in rows:
            counts[int(r['label'])] += 1
        counts = np.clip(counts, 1, None)
        prior[key] = np.log(counts / counts.sum())
    for r, b in zip(test, base):
        key = (r['dataset'], len(r['options']))
        lp = prior.get(key)
        p = b[0] if lp is None else softmax(np.log(np.clip(b[0], 1e-12, 1)) - 0.5 * lp)
        out.append((p, r['label']))
    return out


def rule_length_prior(test, cal, base):
    """Down-weight options whose surface length is atypical for the dataset."""
    out = []
    stats = {}
    by = defaultdict(list)
    for r in cal:
        by[(r['dataset'], len(r['options']))].append(len(r['options'][int(r['label'])]))
    for key, v in by.items():
        stats[key] = (float(np.mean(v)), float(np.std(v)) or 1.0)
    for r, b in zip(test, base):
        key = (r['dataset'], len(r['options']))
        st = stats.get(key)
        if st is None:
            out.append((b[0], r['label'])); continue
        mu, sd = st
        lens = np.array([len(o) for o in r['options']], dtype=float)
        z = (lens - mu) / sd
        out.append((softmax(np.log(np.clip(b[0], 1e-12, 1)) - 0.15 * z * z), r['label']))
    return out


def rule_logit_center(test, cal, base):
    """Per-dataset, per-option-count centering of logits (removes option bias)."""
    groups = defaultdict(list)
    for i, r in enumerate(cal):
        groups[(r['dataset'], len(r['options']))].append(i)
    centers = {}
    for key, idx in groups.items():
        arr = np.array([np.asarray(cal[i]['logits'], dtype=float) for i in idx])
        centers[key] = arr.mean(axis=0)
    out = []
    for r, b in zip(test, base):
        key = (r['dataset'], len(r['options']))
        c = centers.get(key)
        out.append((b[0] if c is None else softmax(np.asarray(r['logits'], dtype=float) - c), r['label']))
    return out


def rule_logit_zscore(test, cal, base):
    """Per-dataset standardization of each option's logit by calibration stats."""
    groups = defaultdict(list)
    for r in cal:
        groups[(r['dataset'], len(r['options']))].append(r)
    stats = {}
    for key, rows in groups.items():
        arr = np.array([np.asarray(r['logits'], dtype=float) for r in rows])
        stats[key] = (arr.mean(0), arr.std(0) + 1e-6)
    out = []
    for r, b in zip(test, base):
        key = (r['dataset'], len(r['options']))
        st = stats.get(key)
        if st is None:
            out.append((b[0], r['label'])); continue
        mu, sd = st
        out.append((softmax((np.asarray(r['logits'], dtype=float) - mu) / sd), r['label']))
    return out


def rule_linear_map(test, cal, base, lam=1.0):
    """Per-dataset diagonal affine map on logits, fit by ridge on calibration.

    Fit p' = softmax(w * logit + b) with a scalar per option slot plus bias,
    chosen to minimize calibration NLL. Post-hoc, no gradients at test time.
    """
    groups = defaultdict(list)
    for r in cal:
        groups[(r['dataset'], len(r['options']))].append(r)
    params = {}
    for key, rows in groups.items():
        n = key[1]
        best, best_nll = (1.0, np.zeros(n)), np.inf
        for w in np.linspace(0.2, 3.0, 29):
            for b in np.linspace(-2.0, 2.0, 17):
                nll = 0.0
                for r in rows:
                    z = w * np.asarray(r['logits'], dtype=float) + b
                    z = z - z.max()
                    lse = np.log(np.exp(z).sum())
                    nll += lse - z[int(r['label'])]
                nll /= len(rows)
                if nll < best_nll:
                    best_nll, best = nll, (w, b)
        params[key] = best
    out = []
    for r, b in zip(test, base):
        key = (r['dataset'], len(r['options']))
        p = params.get(key)
        out.append((b[0] if p is None else softmax(p[0] * np.asarray(r['logits'], dtype=float) + p[1]), r['label']))
    return out


def rule_diag_affine_acc(test, cal, base):
    """Per-dataset diagonal affine map on logits, fit to maximize calibration
    accuracy (not NLL). Each option slot gets its own scalar weight plus a
    shared bias, chosen by grid search over the calibration split.
    """
    groups = defaultdict(list)
    for r in cal:
        groups[(r['dataset'], len(r['options']))].append(r)
    params = {}
    for key, rows in groups.items():
        n = key[1]
        best, best_acc = (np.ones(n), 0.0), -1.0
        for w in np.linspace(0.5, 2.0, 7):
            for b in np.linspace(-1.0, 1.0, 9):
                correct = 0
                for r in rows:
                    z = w * np.asarray(r['logits'], dtype=float) + b
                    if int(np.argmax(z)) == int(r['label']):
                        correct += 1
                acc = correct / len(rows)
                if acc > best_acc:
                    best_acc, best = acc, (w, b)
        params[key] = best
    out = []
    for r, b in zip(test, base):
        key = (r['dataset'], len(r['options']))
        p = params.get(key)
        out.append((b[0] if p is None else
                    softmax(p[0] * np.asarray(r['logits'], dtype=float) + p[1]), r['label']))
    return out


def rule_topk_consensus(test, cal, base, k=2):
    """When the base prediction is uncertain (top-2 margin < median calibration
    margin), take the option that appears most often in the top-k of the
    calibration-aligned position prior. Otherwise keep the base.
    """
    groups = defaultdict(list)
    for r in cal:
        groups[(r['dataset'], len(r['options']))].append(r)
    margins = {}
    for key, rows in groups.items():
        margins[key] = float(np.median([r['logits'][int(np.argsort(r['logits'])[-1])] -
                                        r['logits'][int(np.argsort(r['logits'])[-2])]
                                        for r in rows]))
    out = []
    for r, b in zip(test, base):
        key = (r['dataset'], len(r['options']))
        order = np.argsort(b[0])[::-1]
        margin = float(b[0][order[0]] - b[0][order[1]]) if len(b[0]) > 1 else 1.0
        if margin >= margins.get(key, 0.0):
            out.append((b[0], r['label']))
            continue
        # Count how often each option is the correct label in calibration.
        counts = np.zeros(len(b[0]))
        for cr in groups[key]:
            counts[int(cr['label'])] += 1
        # Blend base with the calibration label-frequency prior.
        prior = counts / counts.sum()
        blended = softmax(np.log(np.clip(b[0], 1e-12, 1)) + 0.5 * np.log(prior))
        out.append((blended, r['label']))
    return out


RULES = {
    'position_debias': rule_position_debias,
    'length_prior': rule_length_prior,
    'logit_center': rule_logit_center,
    'logit_zscore': rule_logit_zscore,
    'linear_map': rule_linear_map,
    'diag_affine_acc': rule_diag_affine_acc,
    'topk_consensus': rule_topk_consensus,
}

if __name__ == '__main__':
    import sys
    only = sys.argv[1:] or list(RULES)
    for name in only:
        evaluate(name, RULES[name])