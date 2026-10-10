"""Method 4: Domain-Shifted Test-Time Calibration.

Instead of fitting one global temperature on the calibration split and
freezing it, refit the temperature online at test time using a sliding window
of recent predictions, and detect distribution shift on the probability
sequences. When drift is detected, the calibration parameters are re-fit or
decayed so the model stays calibrated as the input distribution moves.

Components:
  - Online temperature refit: every `window` examples, re-solve the NLL-minimizing
    temperature on the last `window` logits (cheap grid search, no gradients).
  - Drift detector: a Page-Hinkley or CUSUM test on the running mean of the
    top-1 confidence. A sustained change in confidence signals shift.
  - Response: on drift, shorten the refit window (decay) so the temperature
    tracks the new regime.

This is a post-hoc, parameter-free adaptation of the decision interface: no
backbone gradients, no weight changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from base import (collect_logits, rows_for, save_json, to_probabilities,  # noqa: E402
                  torch_device)
from build import build_adapter  # noqa: E402


def fit_temp(records):
    """NLL-minimizing temperature over a list of {logits,label} records (grid search).

    Returns (temperature, at_boundary): at_boundary flags that the optimum sat at
    the edge of the search grid, meaning the true optimum lies outside [0.05, 20].
    """
    if not records:
        return 1.0, False
    grid = np.exp(np.linspace(np.log(.05), np.log(20.), 121))
    losses = np.zeros(len(grid))
    for row in records:
        values = np.asarray(row['logits'], dtype=np.float64)
        scores = values[None, :] / grid[:, None]
        shifted = scores - scores.max(axis=1, keepdims=True)
        losses += -shifted[:, row['label']] + np.log(np.exp(shifted).sum(axis=1))
    index = int(np.argmin(losses / len(records)))
    return float(grid[index]), index in (0, len(grid) - 1)


def page_hinkley(confidences, delta=0.005, threshold=50.0):
    """Drift detector on the running mean of confidence. Returns alarm indices."""
    alarms, mean, sum_diff = [], 0.0, 0.0
    for i, c in enumerate(confidences):
        sum_diff += (c - mean - delta)
        mean = mean + (c - mean) / (i + 1)
        if sum_diff > threshold:
            alarms.append(i)
            sum_diff = 0.0
            mean = c
    return alarms


def run(args):
    """Run drift calibration. `args` may be an argparse Namespace or a plain dict."""
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    adapter = build_adapter(args.baseline, args.checkpoint, args.device, args.manifest,
                            args.batch_size, backbone=getattr(args, 'backbone', None))
    test = collect_logits(adapter, rows_for(adapter.splits, args.datasets, args.max_examples,
                                            split='test'),
                          args.batch_size, progress=args.progress, desc='test inference')
    window = args.window
    history = []
    temps = []
    confidences = []
    iterator = range(0, len(test), window)
    if args.progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc='drift refit',
                        total=(len(test) + window - 1) // window)
    for start in iterator:
        chunk = test[start:start + window]
        tail = test[max(0, start - window * args.decay):start + window]
        temp, at_boundary = fit_temp(tail)
        temps.append({'start': start, 'end': min(start + window, len(test)),
                      'temperature': temp, 'at_grid_boundary': at_boundary})
        for r in chunk:
            probs = to_probabilities(r['logits'], temp)
            confidences.append(float(probs.max()))
            history.append({'id': r['id'], 'dataset': r['dataset'], 'label': r['label'],
                            'prediction': int(probs.argmax()), 'correct': int(probs.argmax() == r['label']),
                            'temperature': temp, 'confidence': float(probs.max())})
    alarms = page_hinkley(confidences, args.delta, args.threshold)
    result = {'method': 'domain_shifted_calibration', 'baseline': args.baseline,
              'window': window, 'decay': args.decay, 'delta': args.delta,
              'threshold': args.threshold, 'test_n': len(test),
              'temperatures': temps, 'drift_alarms': alarms,
              'final_accuracy': float(np.mean([h['correct'] for h in history])),
              'mean_temperature': float(np.mean([t['temperature'] for t in temps])),
              'rows': history}
    save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    print(f"drift alarms at indices: {alarms}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='laya_med',
                   choices=['laya_med', 'laya', 'medjev', 'medgemma_direct_headonly',
                            'medgemma_direct_lora'])
    p.add_argument('--checkpoint')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-examples', type=int)
    p.add_argument('--window', type=int, default=200)
    p.add_argument('--decay', type=int, default=2, help='multiplicative window decay on drift')
    p.add_argument('--delta', type=float, default=0.005)
    p.add_argument('--threshold', type=float, default=50.0)
    p.add_argument('--output', type=Path, default=Path('tta/drift_cal.json'))
    p.add_argument('--progress', action='store_true', help='show tqdm progress bars over batches/windows')
    args = p.parse_args()
    run(args)