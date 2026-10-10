"""Prior-Gated Consensus: blend base logits with a calibration label prior.

When the base decision is uncertain (top-2 logit margin below the per-dataset
median margin observed on calibration), the model's option logits are blended
with a label-frequency prior estimated from the calibration split. Confident
decisions are left untouched, so abstention stays at zero and the method never
defers. The prior is dataset- and option-count-specific, and calibration labels
are the only labels used -- test labels never enter the decision path.

This is a research prototype: it is reported alongside the frozen baseline so
the effect of prior-gated consensus is measurable. No gradients, no weight
updates, no abstention.
"""
from __future__ import annotations

import numpy as np

from tta.base import to_probabilities
from tta.refinement import prepare, score, finish


def _calibration_stats(adapter, datasets, batch_size):
    """Per (dataset, option_count): median top-2 margin and label-frequency prior.

    Calibration rows are raw manifest rows, so their logits are scored on the
    fly with the shared adapter (no test labels are ever passed).
    """
    stats = {}
    for dataset in dict.fromkeys(datasets):
        calibration = [r for r in adapter.splits['calibration'] if r['dataset'] == dataset]
        if len(calibration) < 2:
            continue
        by_n = {}
        for r in calibration:
            by_n.setdefault(len(r['options']), []).append(r)
        for n, rows in by_n.items():
            probs = score(adapter, rows, batch_size)
            margins = []
            counts = np.zeros(n)
            for r, p in zip(rows, probs):
                order = np.argsort(p)[::-1]
                margins.append(float(p[order[0]] - p[order[1]]) if n > 1 else float(p[order[0]]))
                counts[int(r['label'])] += 1
            stats[(dataset, n)] = (float(np.median(margins)), counts / counts.sum())
    return stats


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    strength = getattr(args, 'prior_strength', 0.5)
    if not 0 <= strength <= 1:
        raise ValueError('prior_strength must be in [0, 1]')
    stats = _calibration_stats(adapter, [r['dataset'] for r in rows], args.batch_size)
    T = adapter.temperature()
    output, details = [], []
    for row, record in zip(rows, test):
        base = to_probabilities(record['logits'], T)
        key = (row['dataset'], len(row['options']))
        st = stats.get(key)
        if st is None:
            output.append(base)
            details.append({'gated': False, 'reason': 'no calibration stats'})
            continue
        med_margin, prior = st
        order = np.argsort(base)[::-1]
        margin = float(base[order[0]] - base[order[1]]) if len(base) > 1 else float(base[order[0]])
        if margin >= med_margin:
            output.append(base)
            details.append({'gated': False, 'margin': margin, 'median_margin': med_margin})
            continue
        blended = to_probabilities(np.log(np.clip(base, 1e-300, 1)) + strength * np.log(prior))
        output.append(blended)
        details.append({'gated': True, 'margin': margin, 'median_margin': med_margin,
                        'strength': strength})
    return finish(args, adapter, test, output, details, 'prior_consensus',
                  {'prior_strength': strength, 'gate': 'top2_margin_vs_calibration_median',
                   'variant': 'gradient_free_calibration_prior_consensus'})


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
    p.add_argument('--prior-strength', type=float, default=0.5)
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