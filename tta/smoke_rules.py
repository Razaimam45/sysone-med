#!/usr/bin/env python3
"""Offline sweep of decision rules over a collected view set.

Reads /tmp/smoke_views.json (written by tta/smoke_variants.py) and reports
accuracy and base-change count for each candidate rule. No model inference.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def softmax_rows(views, temperature):
    out = []
    for v in views:
        z = np.asarray(v, dtype=float) / temperature
        z = z - z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return out


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
    print(f"n={len(rows)} temperature={T} base_acc={base_acc(rows):.4f}")

    rules = {}
    for r in rows:
        views = softmax_rows(r['views'], T)
        base = views[0]
        n = len(base)
        # Align raw view logits back to canonical option order.
        orders = r['view_orders']
        raw = []
        for log, order in zip(r['view_logits'], orders):
            a = np.asarray(log, dtype=float)
            aligned = np.empty_like(a)
            aligned[list(order)] = a
            raw.append(aligned)
        # logit-space aggregations
        logit_mean = np.mean(raw, axis=0)
        logit_median = np.median(raw, axis=0)
        # probability-space aggregations
        mean_all = np.mean(views, axis=0)
        sims = [float(np.dot(v, base) / (np.linalg.norm(v) * np.linalg.norm(base) or 1e-12)) for v in views]
        if len(views) > 2:
            keep = [i for i in range(len(views)) if i != int(np.argmin(sims))]
            trimmed = np.mean([views[i] for i in keep], axis=0)
        else:
            trimmed = mean_all
        median = np.median(views, axis=0)
        prod = np.exp(np.mean([np.log(np.clip(v, 1e-12, 1)) for v in views], axis=0))
        prod /= prod.sum()
        votes = np.zeros(n)
        for v in views:
            votes[int(np.argmax(v))] += 1
        w = np.array([v.max() for v in views])
        wmean = np.average(views, axis=0, weights=w)
        wmean /= wmean.sum()
        margins = np.array([v.max() - np.sort(v)[-2] if n > 1 else v.max() for v in views])
        if len(margins) > 1:
            keep = margins >= np.median(margins)
            conf = np.mean([views[i] for i in range(len(views)) if keep[i]], axis=0) if keep.sum() else mean_all
        else:
            conf = mean_all
        counts = np.bincount([int(np.argmax(v)) for v in views], minlength=n)
        cons = np.zeros(n); cons[int(counts.argmax())] = 1.0
        blend = 0.7 * base + 0.3 * cons
        blend /= blend.sum()
        wcounts = np.zeros(n)
        for v, wi in zip(views, w):
            wcounts[int(np.argmax(v))] += wi
        wcons = np.zeros(n); wcons[int(wcounts.argmax())] = 1.0
        # logit-space consensus: argmax of mean logits
        lcons = np.zeros(n); lcons[int(np.argmax(logit_mean))] = 1.0
        # weighted logit mean by confidence
        lwmean = np.average(raw, axis=0, weights=w)
        lprob = to_probs(lwmean, T)
        rules[r['id']] = {
            'mean_all': mean_all.tolist(), 'trimmed': trimmed.tolist(),
            'median': median.tolist(), 'prod': prod.tolist(),
            'votes': (votes / votes.sum()).tolist(), 'wmean': wmean.tolist(),
            'conf': conf.tolist(), 'consensus': cons.tolist(),
            'blend': blend.tolist(), 'wconsensus': wcons.tolist(),
            'logit_mean': to_probs(logit_mean, T).tolist(),
            'logit_median': to_probs(logit_median, T).tolist(),
            'lconsensus': lcons.tolist(),
            'lwmean': lprob.tolist(),
        }

    out = []
    for rule in ['mean_all', 'trimmed', 'median', 'prod', 'votes', 'wmean',
                 'conf', 'consensus', 'blend', 'wconsensus',
                 'logit_mean', 'logit_median', 'lconsensus', 'lwmean']:
        probs = [np.asarray(rules[r['id']][rule]) for r in rows]
        res = report(rule, probs, rows)
        out.append(res)
        print(f"  {rule:12s} acc={res['accuracy']:.4f} delta={res['delta']:+.4f} "
              f"changed={res['changed_n']}/{res['n']}")
    best = max(out, key=lambda x: x['delta'])
    print(f"\nBEST: {best['name']} delta={best['delta']:+.4f}")


def to_probs(logits, temperature):
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


if __name__ == '__main__':
    main()