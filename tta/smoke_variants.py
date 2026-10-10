#!/usr/bin/env python3
"""Collect permutation views once, then evaluate many decision rules offline.

Usage:
    python tta/smoke_variants.py --baseline laya_med --checkpoint models/laya-med/best.pt \
        --samples 20 --permutations 6 --output /tmp/smoke_views.json

Writes per-example base logits + aligned permutation-view logits so that
decision rules can be compared without re-running the model.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

# NOTE: do not set CUDA_VISIBLE_DEVICES here -- the parent shell already
# restricts this process to one physical GPU, and re-mapping the visible
# ordinal inside the process breaks torch.device('cuda:2').
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tta.base import rows_for, to_probabilities
from tta.build import build_adapter
from tta.method_permutation import permutations, score_views


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+',
                   default=['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--permutations', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    adapter = build_adapter(args.baseline, args.checkpoint, args.device, args.manifest,
                            args.batch_size, backbone=None)
    groups = [iter(rows_for(adapter.splits, [name])) for name in args.datasets]
    selected = []
    while len(selected) < args.samples:
        added = 0
        for group in groups:
            row = next(group, None)
            if row is not None:
                selected.append(row)
                added += 1
            if len(selected) == args.samples:
                break
        if not added:
            break
    if not selected:
        raise SystemExit('no samples available')

    from tta.base import collect_logits
    test = collect_logits(adapter, selected, args.batch_size)
    T = adapter.temperature()
    rng = np.random.default_rng(args.seed)
    payload = {'temperature': T, 'rows': []}
    for row, record in zip(selected, test):
        views = permutations(len(row['options']), args.permutations, rng)
        base = to_probabilities(record['logits'], T)
        # Raw per-view logits in the option order presented to the model; the
        # evaluator aligns them back to the canonical option order.
        raw_views = [record['logits']]
        for order in views[1:]:
            augmented = [{**row, 'options': [row['options'][i] for i in order]}]
            raw_views.append(np.asarray(adapter.score_logits(augmented, args.batch_size)[0],
                                        dtype=float).tolist())
        payload['rows'].append({
            'id': row['id'], 'dataset': row['dataset'], 'label': int(row['label']),
            'options': list(row['options']), 'base_logits': record['logits'],
            'base': base.tolist(),
            'views': [base.tolist()] + [to_probabilities(v, T).tolist() for v in raw_views[1:]],
            'view_logits': raw_views,
            'view_orders': [list(o) for o in views],
        })
    import json
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(f'COLLECTED {len(payload["rows"])} examples with {args.permutations} views each', flush=True)


if __name__ == '__main__':
    main()