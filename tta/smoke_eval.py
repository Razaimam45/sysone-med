#!/usr/bin/env python3
"""Offline evaluation of many decision rules over a collected view set.

Reads the JSON written by tta/smoke_variants.py and reports per-rule accuracy
and how often each rule changes the base prediction. No model inference here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


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
    return {'name': name, 'accuracy': acc, 'base': base_acc(rows),
            'delta': acc - base_acc(rows), 'changed_n': int(sum(changed)), 'n': len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--views', type=Path, required=True)
    p.add_argument('--permutations', type=int, default=6)
    args = p.parse_args()
    payload = load(args.views)
    T = payload['temperature']
    rows = payload['rows']
    print(f"n={len(rows)} temperature={T} base_acc={base_acc(rows):.4f}")

    results = []
    for r in rows:
        views = softmax_rows(r['views'], T)
        base = views[0]
        n = len(base)
        # --- candidate decision rules ---
        # 1. mean of all views (permutation consistency)
        mean_all = np.mean(views, axis=0)
        # 2. trimmed mean: drop the single most divergent view
        if len(views) > 2:
            sims = [float(np.dot(v, base) / (np.linalg.norm(v) * np.linalg.norm(base) or 1e-12)) for v in views]
            keep = [i for i in range(len(views)) if i != int(np.argmin(sims))]
            trimmed = np.mean([views[i] for i in keep], axis=0)
        else:
            trimmed = mean_all
        # 3. median of views
        median = np.median(views, axis=0)
        # 4. product of views (geometric mean in prob space)
        prod = np.exp(np.mean([np.log(np.clip(v, 1e-12, 1)) for v in views], axis=0))
        prod /= prod.sum()
        # 5. majority vote over argmax predictions
        votes = np.zeros(n)
        for v in views:
            votes[int(np.argmax(v))] += 1
        # 6. weighted mean by confidence (weight = max prob)
        weights = np.array([v.max() for v in views])
        wmean = np.average(views, axis=0, weights=weights)
        wmean /= wmean.sum()
        # 7. max-margin vote: keep views whose top-2 margin >= median margin
        margins = np.array([v.max() - np.sort(v)[-2] if n > 1 else v.max() for v in views])
        if len(margins) > 1:
            keep = margins >= np.median(margins)
            if keep.sum() >= 1:
                conf_views = np.mean([views[i] for i in range(len(views)) if keep[i]], axis=0)
            else:
                conf_views = mean_all
        else:
            conf_views = mean_all
        # 8. consensus-only: require >=2 views to agree on the argmax
        top = [int(np.argmax(v)) for v in views]
        counts = np.bincount(top, minlength=n)
        consensus = counts.argmax()
        cons = np.zeros(n); cons[consensus] = 1.0
        # 9. base + a small shift toward the most-agreed alternative
        alt = int(np.argsort(counts)[-2]) if n > 1 and counts.max() > 1 else top[0]
        blend = 0.5 * base + 0.5 * (np.eye(n)[alt] if alt != top[0] else base)
        blend /= blend.sum()
        r['rules'] = {
            'mean_all': mean_all.tolist(), 'trimmed': trimmed.tolist(),
            'median': median.tolist(), 'prod': prod.tolist(),
            'votes': (votes / votes.sum()).tolist(), 'wmean': wmean.tolist(),
            'conf_views': conf_views.tolist(), 'consensus': cons.tolist(),
            'blend': blend.tolist(),
        }
    rules = ['mean_all', 'trimmed', 'median', 'prod', 'votes', 'wmean',
             'conf_views', 'consensus', 'blend']
    for rule in rules:
        probs = [np.asarray(r['rules'][rule]) for r in rows]
        res = report(rule, probs, rows)
        results.append(res)
        print(f"  {rule:12s} acc={res['accuracy']:.4f} delta={res['delta']:+.4f} "
              f"changed={res['changed_n']}/{res['n']}")
    best = max(results, key=lambda x: x['delta'])
    print(f"\nBEST: {best['name']} delta={best['delta']:+.4f}")


if __name__ == '__main__':
    main()