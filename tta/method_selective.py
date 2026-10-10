"""CAS-inspired selective permutation refinement (not the original image-CAS TTA).

Probe an option-order view. If its aligned candidate distribution has low
cosine similarity to the original, average further views; otherwise retain
exactly the original distribution. No optimizer or parameter updates.
"""
from __future__ import annotations

import numpy as np
from tta.base import to_probabilities
from tta.refinement import prepare, finish
from tta.method_permutation import permutations, score_views


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    count = getattr(args, 'permutations', 4)
    threshold = getattr(args, 'cas_threshold', 0.95)
    seed = getattr(args, 'seed', 0)
    if count < 2 or not 0 <= threshold <= 1:
        raise ValueError('Require permutations >= 2 and CAS threshold in [0, 1]')
    rng = np.random.default_rng(seed)
    output, details = [], []
    for row, record in zip(rows, test):
        views = permutations(len(row['options']), count, rng)
        base = to_probabilities(record['logits'], adapter.temperature())
        probe = score_views(adapter, row, views[1:2], args.batch_size)
        similarity = (float(np.clip(np.dot(base, probe[0]) /
                      (np.linalg.norm(base) * np.linalg.norm(probe[0])), 0, 1))
                      if probe else 1.0)
        selected = similarity < threshold
        extra = score_views(adapter, row, views[2:], args.batch_size) if selected else []
        output.append(np.mean([base] + probe + extra, axis=0) if selected else base)
        details.append({'selected': selected, 'cross_view_similarity': similarity,
                        'additional_scored_examples': len(probe) + len(extra)})
    return finish(args, adapter, test, output, details, 'selective_permutation_refinement',
                  {'permutations': count, 'seed': seed, 'cas_threshold': threshold,
                   'gate': 'original_reverse_option_cosine',
                   'variant': 'gradient_free_CAS_inspired_not_original_CAS'})
