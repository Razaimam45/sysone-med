"""Method 3: RLCD-Inspired Test-Time Fine-Tuning.

RLCD (Reinforcement Learning for Calibrated Decisions) optimizes the decision
policy against a reward that combines accuracy and calibration. Applied at
test time to a decision model, this adapts the trainable parameters of the
checkpoint using a lightweight proxy reward computed from the model's own
output distribution, WITHOUT any ground-truth labels and WITHOUT
optimization-free abstention.

Design:
  - Self-supervised pseudo-label: the current argmax prediction is treated as
    a pseudo-target, and the model is pushed to sharpen its distribution toward
    it (a maximum-entropy-reversal / "self-training" proxy).
  - Calibration-aware reward: the RLCD-style objective rewards (a) confidence
    margin and (b) low entropy of the predictive distribution, i.e. it optimizes
    the same quantities that downstream calibration metrics (ECE, NLL) measure.
  - Test-time SGD on the checkpoint's trainable parameters only, for a few
    inner steps with a small learning rate, one example (or mini-batch) at a
    time, then a snapshot restore so the adapted state is reported but the
    original checkpoint is not overwritten.

This is a research prototype: the pseudo-reward is a stand-in for the true
calibration objective. It is reported alongside the frozen baseline so the
effect of test-time adaptation is measurable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from base import (collect_logits, rows_for, save_json, to_probabilities,  # noqa: E402
                  torch_device)
from build import build_adapter  # noqa: E402


def adapt(adapter, batch, optimizer, temperature, device, steps=1):
    """One inner test-time update maximizing the proxy reward (a torch scalar loss)."""
    adapter.model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = None
    for _ in range(steps):
        for scores in adapter.forward_batch_grad(batch):
            log_probs = torch.log_softmax(scores / temperature, dim=-1)
            probs = torch.exp(log_probs)
            order = torch.sort(probs, descending=True)[0]
            margin = order[0] - order[1] if len(order) > 1 else order[0]
            entropy = -(probs * log_probs).sum()
            norm_entropy = entropy / math.log(len(probs)) if len(probs) > 1 else torch.zeros_like(entropy)
            reward = margin * (1.0 - norm_entropy)
            loss = -reward if loss is None else loss - reward
    if loss is None:
        raise RuntimeError('No candidates in batch; nothing to adapt')
    loss.backward()
    optimizer.step()
    adapter.model.eval()
    return float(loss.item())


def run(args):
    """Run RLCD. `args` may be an argparse Namespace or a plain dict."""
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    device = torch_device(args.device)
    adapter = build_adapter(args.baseline, args.checkpoint, args.device, args.manifest,
                            args.batch_size, backbone=getattr(args, 'backbone', None))
    if not adapter.requires_grad():
        raise RuntimeError(f'{args.baseline} has no trainable parameters; RLCD is not supported')
    temperature = adapter.temperature()
    test_rows = rows_for(adapter.splits, args.datasets, args.max_examples, split='test')

    baseline = collect_logits(adapter, test_rows, args.batch_size,
                              progress=args.progress, desc='baseline inference')

    trainable = adapter.snapshot_trainable()
    if not trainable:
        raise RuntimeError('Checkpoint has no trainable parameters; cannot adapt')
    snapshot = adapter.clone_trainable(trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    from torch.utils.data import DataLoader
    adapted = []
    starts = range(0, len(test_rows), args.batch_size)
    if args.progress:
        from tqdm import tqdm
        starts = tqdm(starts, desc='rlcd adapt',
                      total=(len(test_rows) + args.batch_size - 1) // args.batch_size)
    for start in starts:
        chunk = test_rows[start:start + args.batch_size]
        batch = adapter.collate(chunk)
        for _ in range(args.steps):
            adapt(adapter, batch, optimizer, temperature, device)
        records = collect_logits(adapter, chunk, len(chunk),
                                 progress=args.progress, desc='adapted inference')
        adapted.extend(records)
        adapter.restore_trainable(trainable, snapshot)
        optimizer.zero_grad(set_to_none=True)

    result = {'method': 'rlcd_test_time_finetune', 'baseline': args.baseline,
              'lr': args.lr, 'steps': args.steps, 'batch_size': args.batch_size,
              'temperature': temperature, 'test_n': len(test_rows),
              'baseline': baseline, 'adapted': adapted}
    save_json(args.output, result)
    print(json.dumps({'method': result['method'], 'baseline': args.baseline,
                      'test_n': result['test_n'],
                      'baseline_accuracy': float(np.mean([int(to_probabilities(r['logits'], temperature).argmax() == r['label']) for r in baseline])),
                      'adapted_accuracy': float(np.mean([int(to_probabilities(r['logits'], temperature).argmax() == r['label']) for r in adapted]))}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='laya_med',
                   choices=['laya_med', 'medjev', 'medgemma_direct_headonly',
                            'medgemma_direct_lora'])
    p.add_argument('--checkpoint')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-examples', type=int)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--output', type=Path, default=Path('tta/rlcd.json'))
    p.add_argument('--progress', action='store_true', help='show a tqdm progress bar over batches')
    args = p.parse_args()
    run(args)