"""Permutation-Consensus Refinement: aggregate option-order views in logit space.

Each candidate option is scored under several option permutations (identity,
reverse, and random shuffles). The per-view logits are realigned to the
canonical option order and averaged in *logit* space, which preserves the
model's relative option preferences far better than averaging probabilities
(especially under high temperature). The refined distribution is the softmax
of the averaged logits. When all views agree on the top option the decision is
unchanged, so this is a low-abstention, gradient-free, test-time policy.

This is a research prototype: it is reported alongside the frozen baseline so
the effect of option-order aggregation is measurable. No labels are used for
gating or scoring; no parameters are updated.
"""
from __future__ import annotations

import numpy as np

from tta.base import to_probabilities
from tta.refinement import prepare, score, finish
from tta.method_permutation import permutations


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    count, seed = getattr(args, 'permutations', 4), getattr(args, 'seed', 0)
    if count < 2:
        raise ValueError('permutations must be at least 2')
    rng = np.random.default_rng(seed)
    T = adapter.temperature()
    output, details = [], []
    for row, record in zip(rows, test):
        views = permutations(len(row['options']), count, rng)
        # Raw per-view logits in the option order presented to the model.
        raw = [np.asarray(record['logits'], dtype=float)]
        for order in views[1:]:
            augmented = [{**row, 'options': [row['options'][i] for i in order]}]
            raw.append(np.asarray(adapter.score_logits(augmented, args.batch_size)[0],
                                  dtype=float))
        # Realign each view back to the canonical option order, then average.
        aligned = []
        for logits, order in zip(raw, views):
            a = np.empty_like(logits)
            a[list(order)] = logits
            aligned.append(a)
        mean_logits = np.mean(aligned, axis=0)
        probs = to_probabilities(mean_logits, T)
        output.append(probs)
        base = to_probabilities(record['logits'], T)
        details.append({'views': len(views), 'argmax_unchanged': int(np.argmax(probs) == np.argmax(base)),
                        'additional_scored_examples': len(views) - 1})
    return finish(args, adapter, test, output, details, 'permutation_consensus',
                  {'permutations': count, 'seed': seed, 'includes_original': True,
                   'aggregation': 'logit_mean_realigned',
                   'variant': 'gradient_free_option_order_consensus'})


def main():
    from tta.build import build_adapter
    from tta.base import rows_for
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', choices=['laya_med', 'medgemma_direct_lora', 'medgemma_ll'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+',
                   default=['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-samples', type=int, default=50)
    p.add_argument('--permutations', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    adapter = build_adapter(args.baseline, args.checkpoint, args.device, args.manifest, args.batch_size)
    common = {**vars(args), 'max_examples': args.max_samples, 'score': 'confidence',
              'tau': [0.3, 0.5, 0.7], 'alpha': [0.1, 0.2, 0.3]}
    result = run(common, adapter=adapter, test=None)
    print(f"BASE={result['base']['accuracy']:.4f} REFINED={result['adapted']['accuracy']:.4f} "
          f"N={result['test_n']} CHANGED={result['changed_n']}", flush=True)


if __name__ == '__main__':
    main()