"""Experimental evidence-prior correction; no claim of guaranteed improvement.

For uncertain decisions, subtract a weak content-free answer prior in log
probability space. The null view preserves question/options, but removes case
context and images. This is meaningful only where evidence lives in state.
High-confidence predictions and rows with no removable evidence stay unchanged.
No labels are used for gating or scoring; no parameters are updated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from tta.base import to_probabilities
from tta.refinement import prepare, score, finish


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    strength = getattr(args, 'prior_strength', 0.25)
    confidence = getattr(args, 'prior_confidence', 0.6)
    min_margin = getattr(args, 'prior_margin', 0.05)
    if not 0 <= strength <= 1 or not 0 <= confidence <= 1 or not 0 <= min_margin <= 1:
        raise ValueError('Prior strength, confidence and margin must be in [0, 1]')
    output, details = [], []
    for row, record in zip(rows, test):
        base = to_probabilities(record['logits'], adapter.temperature())
        state = row.get('state', '')
        evidence = bool(state.get('text') or state.get('image')) if isinstance(state, dict) else bool(state)
        selected = evidence and len(base) > 1 and float(base.max()) < confidence and strength > 0
        probs = base
        accepted = False
        if selected:
            null = score(adapter, [{**row, 'state': ''}], args.batch_size)[0]
            corrected = to_probabilities(np.log(np.clip(base, 1e-300, 1)) -
                                         strength * np.log(np.clip(null, 1e-300, 1)))
            ordered = np.sort(corrected)
            accepted = float(ordered[-1] - ordered[-2]) >= min_margin
            if accepted:
                probs = corrected
        output.append(probs)
        details.append({'selected': bool(selected), 'correction_accepted': bool(accepted),
                        'additional_scored_examples': int(selected)})
    return finish(args, adapter, test, output, details, 'evidence_prior_correction',
                  {'strength': strength, 'confidence_gate': confidence,
                   'minimum_corrected_margin': min_margin,
                   'experimental': True, 'guaranteed_improvement': False,
                   'null_view': 'remove_state_text_and_image_keep_question_options'})


def main():
    from tta.build import build_adapter
    from tta.base import rows_for
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', choices=['laya_med', 'medgemma_direct_lora', 'medgemma_ll'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+', default=['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa'])
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-samples', type=int, default=10, help='Total examples, round-robin across datasets')
    p.add_argument('--prior-strength', type=float, default=0.25)
    p.add_argument('--prior-confidence', type=float, default=0.6)
    p.add_argument('--prior-margin', type=float, default=0.05)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.max_samples < 1:
        p.error('--max-samples must be positive')
    adapter = build_adapter(args.baseline, args.checkpoint, args.device, args.manifest, args.batch_size)
    groups = [iter(rows_for(adapter.splits, [name])) for name in args.datasets]
    selected = []
    while len(selected) < args.max_samples:
        added = 0
        for group in groups:
            row = next(group, None)
            if row is not None:
                selected.append(row)
                added += 1
            if len(selected) == args.max_samples:
                break
        if not added:
            break
    # prepare groups by dataset, consistently for both base and refined scoring.
    adapter.splits = {**adapter.splits, 'test': selected}
    args.datasets = [name for name in args.datasets if any(r['dataset'] == name for r in selected)]
    args.max_examples = None
    result = run(args, adapter)
    print(f"BASE={result['base']['accuracy']:.4f} REFINED={result['adapted']['accuracy']:.4f} "
          f"N={result['test_n']} CHANGED={result['changed_n']}", flush=True)
    print('PER_DATASET', result['base']['per_dataset'], result['adapted']['per_dataset'], flush=True)


if __name__ == '__main__':
    main()
