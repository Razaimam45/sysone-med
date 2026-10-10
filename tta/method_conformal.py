"""Method 2: Conformal Prediction for Calibrated Decisions.

Converts raw option logits into valid prediction sets with guaranteed coverage,
adapting at test time using exchangeability of the calibration split.

Algorithm (split conformal, adaptive threshold):
  1. On the held-out CALIBRATION split, compute each example's nonconformity
     score = max probability minus the true-class probability (a "risk" score;
     higher means the model is confident in the wrong answer).
  2. At test time, for each example compute the same score, then form the
     prediction set of all classes whose score is <= the empirical quantile q
     of the calibration scores at level alpha (e.g. 0.1 -> 90% coverage).
  3. Report set size, coverage, accuracy of the "top-1 in set" rule, and the
     risk-vs-coverage tradeoff curve.

No optimization, no gradients, no model weight changes: pure test-time
adaptation of the decision interface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tta.base import (collect_logits, rows_for, save_json, torch_device,  # noqa: E402
                  to_probabilities)
from tta.build import build_adapter  # noqa: E402


def nonconformity(probabilities, label):
    """Risk-style score: confidence in the predicted class minus mass on truth."""
    return float(probabilities.max() - probabilities[label])


def conformal_threshold(scores, alpha):
    """Empirical (1-alpha) quantile, conservative (finite-sample valid)."""
    if len(scores) == 0:
        return float('inf')
    return float(np.quantile(scores, 1 - alpha, method='higher'))


def run(args, adapter=None, test=None):
    """Run conformal. `args` may be an argparse Namespace or a plain dict."""
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    adapter = adapter if adapter is not None else build_adapter(args.baseline, args.checkpoint, args.device, args.manifest,
                            args.batch_size, backbone=getattr(args, 'backbone', None))
    temperature = adapter.temperature()
    calib_rows = rows_for(adapter.splits, args.datasets, args.max_examples, split='calibration')
    test_rows = rows_for(adapter.splits, args.datasets, args.max_examples, split='test')
    calib = collect_logits(adapter, calib_rows, args.batch_size,
                           progress=args.progress, desc='calibration inference')
    test = test if test is not None else collect_logits(adapter, test_rows, args.batch_size,
                          progress=args.progress, desc='test inference')

    calib_scores = np.array([nonconformity(to_probabilities(r['logits'], temperature), r['label'])
                             for r in calib])
    result = {'method': 'conformal_prediction', 'baseline': args.baseline,
              'temperature': temperature, 'calibration_n': len(calib),
              'test_n': len(test), 'results': []}
    for alpha in args.alpha:
        threshold = conformal_threshold(calib_scores, alpha)
        rows = []
        for r in test:
            probs = to_probabilities(r['logits'], temperature)
            set_mask = (probs.max() - probs) <= threshold
            pred_set = [r['options'][i] for i, keep in enumerate(set_mask) if keep]
            rows.append({'id': r['id'], 'dataset': r['dataset'], 'label': r['label'],
                         'prediction': int(probs.argmax()), 'set': pred_set,
                         'set_size': len(pred_set),
                         # Index the mask: `label in set_mask` would compare the
                         # integer label against True/False and mark labels >= 2
                         # as uncovered whatever the set contains.
                         'covered': bool(set_mask[r['label']]),
                         'max_prob': float(probs.max())})
        sizes = np.array([row['set_size'] for row in rows])
        decided = [row for row in rows if row['set_size'] == 1]
        decided_acc = (float(np.mean([int(row['prediction'] == row['label']) for row in decided]))
                       if decided else None)
        result['results'].append({
            'alpha': alpha, 'target_coverage': 1 - alpha,
            'empirical_coverage': float(np.mean([row['covered'] for row in rows])),
            'mean_set_size': float(sizes.mean()), 'median_set_size': float(np.median(sizes)),
            'abstention_rate': float(np.mean(sizes != 1)), 'decided_accuracy': decided_acc,
            'rows': rows})
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
    for block in result['results']:
        print(f"alpha={block['alpha']:.2f} coverage={block['empirical_coverage']:.3f} "
              f"mean_set_size={block['mean_set_size']:.2f} decided_acc={block['decided_accuracy']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='laya_med',
                   choices=['laya_med', 'laya', 'medjev', 'medgemma_direct_headonly',
                            'medgemma_direct_lora', 'medgemma_ll'])
    p.add_argument('--checkpoint')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-examples', type=int)
    p.add_argument('--alpha', type=float, nargs='+', default=[0.3]) # options: default=[0.1, 0.2, 0.3]
    p.add_argument('--output', type=Path, default=Path('tta/conformal.json'))
    p.add_argument('--progress', action='store_true', help='show a tqdm progress bar over batches')
    args = p.parse_args()
    run(args)