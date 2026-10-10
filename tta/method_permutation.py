"""Permutation Consistency: average probabilities after restoring option identity."""
from __future__ import annotations

import itertools
import math
import numpy as np

from tta.base import to_probabilities
from tta.refinement import prepare, score, finish


def permutations(n, count, rng):
    """Distinct views including identity; cap at n! for small answer spaces."""
    if n < 1 or count < 1:
        raise ValueError('Options and permutation count must be positive')
    target = min(count, math.factorial(n))
    identity = tuple(range(n))
    views = [identity]
    if target > 1:
        views.append(tuple(reversed(identity)))
    if n <= 7:
        remaining = [p for p in itertools.permutations(identity) if p not in views]
        rng.shuffle(remaining)
        views.extend(remaining[:target - len(views)])
    else:
        while len(views) < target:
            candidate = tuple(rng.permutation(n).tolist())
            if candidate not in views:
                views.append(candidate)
    return views


def score_views(adapter, row, views, batch_size):
    augmented = [{**row, 'options': [row['options'][i] for i in order]} for order in views]
    restored = []
    for order, probs in zip(views, score(adapter, augmented, batch_size)):
        aligned = np.empty_like(probs)
        aligned[list(order)] = probs
        restored.append(aligned)
    return restored


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    count, seed = getattr(args, 'permutations', 4), getattr(args, 'seed', 0)
    if count < 2:
        raise ValueError('permutations must be at least 2')
    rng = np.random.default_rng(seed)
    output, details = [], []
    for row, record in zip(rows, test):
        views = permutations(len(row['options']), count, rng)
        base = to_probabilities(record['logits'], adapter.temperature())
        probs = [base] + score_views(adapter, row, views[1:], args.batch_size)
        output.append(np.mean(probs, axis=0))
        details.append({'views': len(views), 'additional_scored_examples': len(views) - 1})
    return finish(args, adapter, test, output, details, 'permutation_consistency',
                  {'permutations': count, 'seed': seed, 'includes_original': True})
