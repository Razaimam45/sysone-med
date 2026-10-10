"""Method 1: Test-Time Routing / Selective Prediction.

Formalizes the Jev observation ("keeping the half of decisions with the largest
|p-0.5| raises median accuracy") as a test-time abstention policy. When the
calibrated confidence is low, the example is routed to an expensive fallback
(human clinician / larger model) instead of being guessed.

Policy family: accept a decision iff calibrated_confidence >= tau, else abstain.
Calibrated confidence = max probability after temperature scaling, optionally
combined with the |p-0.5| margin. We sweep tau and report accuracy, coverage
(rates of accepted decisions), and the risk-vs-coverage frontier (the curve the
TrustTrade / Adaptive AI Delegation literature optimizes).

No model weight changes; the "fallback" is recorded as an abstention outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tta.base import (collect_logits, rows_for, save_json, torch_device,  # noqa: E402
                  to_probabilities)
from tta.build import build_adapter  # noqa: E402


def confidence(probabilities):
    return float(probabilities.max())


def margin(probabilities):
    """|p_best - p_second|, the decision sharpness used by the Jev paper."""
    order = np.sort(probabilities)[::-1]
    return float(order[0] - order[1]) if len(order) > 1 else float(order[0])


def run(args, adapter=None, test=None):
    """Run routing. `args` may be an argparse Namespace or a plain dict."""
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    adapter = adapter if adapter is not None else build_adapter(args.baseline, args.checkpoint, args.device, args.manifest,
                            args.batch_size, backbone=getattr(args, 'backbone', None))
    temperature = adapter.temperature()
    test_rows = rows_for(adapter.splits, args.datasets, args.max_examples, split='test')
    test = test if test is not None else collect_logits(adapter, test_rows, args.batch_size, progress=args.progress,
                          desc='routing inference')

    scores = []
    for r in test:
        probs = to_probabilities(r['logits'], temperature)
        conf = confidence(probs)
        m = margin(probs)
        scores.append({'id': r['id'], 'dataset': r['dataset'], 'label': r['label'],
                       'prediction': int(probs.argmax()), 'correct': int(probs.argmax() == r['label']),
                       'confidence': conf, 'margin': m,
                       'score': conf if args.score == 'confidence' else m})
    result = {'method': 'test_time_routing', 'baseline': args.baseline,
              'score': args.score, 'temperature': temperature,
              'test_n': len(test), 'results': []}
    for tau in args.tau:
        accepted = [s for s in scores if s['score'] >= tau]
        coverage = len(accepted) / len(scores)
        accuracy = float(np.mean([s['correct'] for s in accepted])) if accepted else None
        result['results'].append({'tau': tau, 'coverage': coverage,
                                  'accepted_accuracy': accuracy,
                                  'accepted_n': len(accepted),
                                  'rows': scores})
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
    for block in result['results']:
        acc = 'n/a' if block['accepted_accuracy'] is None else f"{block['accepted_accuracy']:.3f}"
        print(f"tau={block['tau']:.2f} coverage={block['coverage']:.3f} accepted_acc={acc}")


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
    p.add_argument('--score', choices=['confidence', 'margin'], default='confidence')
    p.add_argument('--tau', type=float, nargs='+', default=[0.7]) # options: default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p.add_argument('--output', type=Path, default=Path('tta/routing.json'))
    p.add_argument('--progress', action='store_true', help='show a tqdm progress bar over batches')
    args = p.parse_args()
    run(args)