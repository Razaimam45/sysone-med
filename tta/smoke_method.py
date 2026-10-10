#!/usr/bin/env python3
"""Smoke-test a single TTA method on N round-robin samples (GPU 2).

Usage:
    python tta/smoke_method.py --baseline laya_med --checkpoint models/laya-med/best.pt \
        --method permutation --permutations 4 --samples 20 --output /tmp/smoke_perm.json

Clean up the output file afterwards; this script is a throwaway harness.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

# Pin to GPU 2 regardless of what the parent shell exported.
# NOTE: do not set CUDA_VISIBLE_DEVICES here -- the parent shell already
# restricts the process to a single physical GPU, and re-mapping the
# visible ordinal inside the process breaks torch.device('cuda:2').

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tta.base import rows_for
from tta.build import build_adapter
from tta.refinement import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+',
                   default=['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--method', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--permutations', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cas-threshold', type=float, default=0.95)
    p.add_argument('--memory-k', type=int, default=3)
    p.add_argument('--memory-capacity', type=int, default=256)
    p.add_argument('--memory-confidence', type=float, default=0.8)
    p.add_argument('--memory-chars', type=int, default=800)
    args = p.parse_args()

    from importlib import import_module
    module = import_module(f'tta.method_{args.method}')

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
    adapter.splits = {**adapter.splits, 'test': selected}
    common = {**vars(args), 'max_examples': None, 'score': 'confidence',
              'tau': [0.3, 0.5, 0.7], 'alpha': [0.1, 0.2, 0.3],
              'output': args.output}
    common.pop('samples', None)
    common.pop('method', None)
    result = module.run(common, adapter=adapter, test=None)
    print(f"METHOD={args.method} BASE={result['base']['accuracy']:.4f} "
          f"ADAPTED={result['adapted']['accuracy']:.4f} N={result['test_n']} "
          f"CHANGED={result['changed_n']}", flush=True)
    per = result['adapted']['per_dataset']
    print('PER_DATASET', {k: round(v['accuracy'], 4) for k, v in per.items()}, flush=True)


if __name__ == '__main__':
    main()