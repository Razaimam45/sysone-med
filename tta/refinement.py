"""Shared frozen-model inference and reporting for non-abstaining policies."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

from tta.base import collect_logits, rows_for, save_json, summarize, to_probabilities
from tta.build import build_adapter


def prepare(args, adapter=None, test=None):
    args = SimpleNamespace(**args) if isinstance(args, dict) else args
    if adapter is None:
        adapter = build_adapter(args.baseline, args.checkpoint, args.device,
                                args.manifest, args.batch_size,
                                backbone=getattr(args, 'backbone', None))
    rows = rows_for(adapter.splits, args.datasets, args.max_examples)
    if test is None:
        test = collect_logits(adapter, rows, args.batch_size)
    if len(test) != len(rows):
        raise ValueError('Cached predictions do not match test rows')
    return args, adapter, rows, test


def score(adapter, rows, batch_size):
    """No gradients or truth labels enter the scoring path, including collators."""
    if not rows:
        return []
    with torch.inference_mode():
        logits = adapter.score_logits([{**r, 'label': 0} for r in rows], batch_size)
    if len(logits) != len(rows):
        raise ValueError('Adapter returned the wrong number of predictions')
    temperature = adapter.temperature()
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    probabilities = []
    for row, values in zip(rows, logits):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(row['options']),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid candidate logits for {row['id']}")
        probabilities.append(to_probabilities(values, temperature))
    return probabilities


def finish(args, adapter, test, probabilities, details, method, config):
    records = []
    for original, probs, extra in zip(test, probabilities, details):
        prediction = int(np.argmax(probs))
        records.append({**original, 'logits': np.log(np.clip(probs, 1e-300, 1)).tolist(),
                        'probabilities': probs.tolist(), 'prediction': prediction,
                        'base_prediction': int(np.argmax(original['logits'])),
                        'correct': prediction == original['label'], **extra})
    result = {'method': method, 'baseline': args.baseline, 'test_n': len(test),
              'backprop': False, 'abstention_rate': 0.0, 'config': config,
              'base': summarize(test, adapter.temperature()),
              'adapted': summarize(records),
              'changed_n': sum(r['prediction'] != r['base_prediction'] for r in records),
              'rows': records}
    save_json(args.output, result)
    print(f"{method}: n={len(test)} accuracy={result['adapted']['accuracy']:.4f}")
    return result
