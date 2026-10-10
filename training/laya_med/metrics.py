"""Metrics and temperature calibration for variable-sized choice sets."""
from collections import defaultdict
import numpy as np


def summarize(records, temperature=1.0):
    if not records or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Nonempty predictions and positive finite temperature required')
    groups = defaultdict(list)
    for row in records:
        scores = np.asarray(row['logits'], dtype=np.float64) / temperature
        if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= row['label'] < len(scores):
            raise ValueError('Invalid logits/label')
        shifted = scores - scores.max()
        logp = shifted - np.log(np.exp(shifted).sum())
        prob = np.exp(logp)
        target = np.zeros_like(prob); target[row['label']] = 1
        groups[row['dataset']].append((int(prob.argmax() == row['label']), -logp[row['label']],
                                       float(((prob - target) ** 2).sum()), float(prob.max())))
    per_task = {}
    for name, rows in groups.items():
        values = np.asarray(rows)
        correct, losses, briers, confidence = values.T
        ece = 0.0
        for index in range(15):
            mask = (confidence >= index / 15) & (confidence < (index + 1) / 15 if index < 14 else confidence <= 1)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        per_task[name] = {'n': len(rows), 'accuracy': float(correct.mean()), 'nll': float(losses.mean()),
                          'brier': float(briers.mean()), 'ece': float(ece)}
    return {'per_dataset': per_task,
            'macro_accuracy': float(np.mean([v['accuracy'] for v in per_task.values()])),
            'macro_nll': float(np.mean([v['nll'] for v in per_task.values()]))}


def fit_temperature(records):
    """Minimize equal-task mean NLL on calibration cases only, with ragged logits."""
    if not records:
        raise ValueError('Empty calibration data')
    grouped = defaultdict(list)
    for row in records:
        grouped[row['dataset']].append(row)
    grid = np.exp(np.linspace(np.log(.05), np.log(20.), 121))
    # Vectorize across temperatures; no padding or fake choice classes.
    task_losses = []
    for rows in grouped.values():
        losses = np.zeros(len(grid))
        for row in rows:
            values = np.asarray(row['logits'], dtype=np.float64)
            if not np.isfinite(values).all() or not 0 <= row['label'] < len(values):
                raise ValueError('Invalid calibration logits/label')
            scores = values[None, :] / grid[:, None]
            shifted = scores - scores.max(axis=1, keepdims=True)
            losses += -shifted[:, row['label']] + np.log(np.exp(shifted).sum(axis=1))
        task_losses.append(losses / len(rows))
    index = int(np.argmin(np.mean(task_losses, axis=0)))
    return {'temperature': float(grid[index]), 'objective': 'macro_dataset_nll',
            'at_search_boundary': index in (0, len(grid) - 1), 'examples': len(records)}
