#!/usr/bin/env python3
"""Offline sweep of gated decision rules over a collected view set.

Reads /tmp/smoke_views.json (written by tta/smoke_variants.py) and reports
accuracy and base-change count for each candidate rule. No model inference.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def to_probs(logits, temperature):
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def base_acc(rows):
    return float(np.mean([int(np.argmax(r['base']) == r['label']) for r in rows]))


def report(name, probs_list, rows):
    correct = [int(np.argmax(p) == r['label']) for p, r in zip(probs_list, rows)]
    changed = [int(np.argmax(p) != np.argmax(r['base'])) for p, r in zip(probs_list, rows)]
    acc = float(np.mean(correct))
    return {'name': name, 'accuracy': acc, 'delta': acc - base_acc(rows),
            'changed_n': int(sum(changed)), 'n': len(rows)}


def main():
    payload = json.loads(Path('/tmp/smoke_views.json').read_text())
    T = payload['temperature']
    rows = payload['rows']
    base_acc_v = base_acc(rows)
    print(f"n={len(rows)} temperature={T} base_acc={base_acc_v:.4f}")

    # Per-example aligned raw logits + softmax probabilities.
    ex = []
    for r in rows:
        raw = []
        for log, order in zip(r['view_logits'], r['view_orders']):
            a = np.asarray(log, dtype=float)
            aligned = np.empty_like(a)
            aligned[list(order)] = a
            raw.append(aligned)
        ex.append({'base': np.asarray(r['base']), 'raw': raw,
                   'views': [to_probs(v, T) for v in raw], 'label': r['label']})

    results = []
    aggs = {
        'logit_mean': lambda raw: np.mean(raw, axis=0),
        'logit_median': lambda raw: np.median(raw, axis=0),
        'prob_mean': lambda raw, views: np.mean(views, axis=0),
    }
    gates = {
        'none': lambda p, base: True,
        'conf_lt_0.7': lambda p, base: float(p.max()) < 0.7,
        'conf_lt_0.8': lambda p, base: float(p.max()) < 0.8,
        'margin_lt_0.2': lambda p, base: (float(p.max() - np.sort(p)[-2]) if len(p) > 1 else float(p.max())) < 0.2,
        'margin_lt_0.4': lambda p, base: (float(p.max() - np.sort(p)[-2]) if len(p) > 1 else float(p.max())) < 0.4,
        'entropy_gt_0.5': lambda p, base: float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p))) > 0.5,
    }
    for aname, afn in aggs.items():
        for gname, gfn in gates.items():
            name = f'{aname}__{gname}'
            probs = []
            for e in ex:
                p = afn(e['raw'], e['views']) if aname.startswith('prob') else to_probs(afn(e['raw']), T)
                probs.append(p if gfn(p, e['base']) else e['base'])
            res = report(name, probs, rows)
            results.append(res)
            print(f"  {name:28s} acc={res['accuracy']:.4f} delta={res['delta']:+.4f} "
                  f"changed={res['changed_n']}/{res['n']}")
    best = max(results, key=lambda x: x['delta'])
    print(f"\nBEST: {best['name']} delta={best['delta']:+.4f} changed={best['changed_n']}/{best['n']}")


if __name__ == '__main__':
    main()