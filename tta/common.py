"""Shared checkpoint loading and inference for Test-Time Adaptation experiments.

Reuses the exact model-construction and inference paths from training/evaluate.py
so that every TTA method scores the same checkpoint on the same test rows.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training'), str(ROOT / 'scripts')])
from laya_med.data import MedicalCollator, MedicalDecisionDataset
from laya_med.model import LayaMedModel
from train import (asset_fingerprint, dim, forward, load_text, load_checkpoint_vision,
                   move, resolve_asset, verify_encoding)
from med_jevify.decision_data import file_hash, load_manifest


def load_bundle(checkpoint_path, device, manifest_path=None):
    """Load a trained Laya-Med checkpoint and return (model, tokenizer, saved, processor, manifest, splits)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 2:
        raise ValueError('Legacy checkpoint lacks audited split/config metadata')
    saved = checkpoint['args']
    manifest_path = manifest_path or saved['manifest']
    if file_hash(manifest_path) != checkpoint['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from training manifest')
    manifest, splits = load_manifest(manifest_path, verify_images=True)
    text_path = resolve_asset(saved['text_model'])
    if asset_fingerprint(text_path)['files'] != checkpoint['assets']['text']['files']:
        raise ValueError('Text assets differ from training')
    tokenizer, text = load_text(text_path, device, saved['lora_rank'])
    verify_encoding(tokenizer, manifest, saved['max_length'])
    processor = vision = None
    if not saved['no_images']:
        vision_path = resolve_asset(saved['vision_model'])
        if asset_fingerprint(vision_path)['files'] != checkpoint['assets']['vision']['files']:
            raise ValueError('Vision assets differ from training')
        processor, vision = load_checkpoint_vision(vision_path, device, saved)
    model = LayaMedModel(text, vision, dim(text), dim(vision) if vision is not None else 0,
                         saved['hidden_dim'], saved.get('fusion', 'mlp')).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    saved['calibration'] = checkpoint.get('calibration')  # temperature lives at checkpoint top-level
    return model, tokenizer, saved, processor, manifest, splits


def rows_for(splits, datasets=None, max_examples=None, split='test'):
    """Collect rows from a split, optionally restricted to named datasets and capped."""
    names = datasets or sorted({r['dataset'] for r in splits[split]})
    rows = []
    for name in names:
        selected = [r for r in splits[split] if r['dataset'] == name]
        if not selected:
            raise ValueError(f'No manifest {split} partition for requested dataset {name}')
        rows.extend(selected[:max_examples] if max_examples else selected)
    return rows


def collect_logits(model, rows, processor, tokenizer, max_length, device, batch_size,
                   progress=False, desc='inference'):
    """Run inference over `rows` and return per-example logits + row metadata."""
    loader = DataLoader(MedicalDecisionDataset(rows), batch_size=batch_size,
                        collate_fn=MedicalCollator(tokenizer, processor, max_length))
    records = []
    iterator = loader
    if progress:
        from tqdm import tqdm
        iterator = tqdm(loader, desc=desc, total=len(loader))
    with torch.inference_mode():
        for batch in iterator:
            moved = move(batch, device)
            logits = forward(model, moved)
            for scores, row in zip(logits, batch['rows']):
                records.append({'id': row.id, 'dataset': row.dataset, 'label': row.label,
                                'options': list(row.options),
                                'logits': scores.float().cpu().tolist()})
    return records


def collect_logits_timed(model, rows, processor, tokenizer, max_length, device, batch_size,
                         progress=False, desc='inference'):
    """Like collect_logits but also records per-example wall-clock latency (seconds)."""
    import time
    loader = DataLoader(MedicalDecisionDataset(rows), batch_size=batch_size,
                        collate_fn=MedicalCollator(tokenizer, processor, max_length))
    records = []
    iterator = loader
    if progress:
        from tqdm import tqdm
        iterator = tqdm(loader, desc=desc, total=len(loader))
    with torch.inference_mode():
        for batch in iterator:
            moved = move(batch, device)
            t0 = time.perf_counter()
            logits = forward(model, moved)
            dt = time.perf_counter() - t0
            per_example = dt / len(batch['rows'])
            for scores, row in zip(logits, batch['rows']):
                records.append({'id': row.id, 'dataset': row.dataset, 'label': row.label,
                                'options': list(row.options),
                                'logits': scores.float().cpu().tolist(),
                                'latency_s': per_example})
    return records


def to_probabilities(logits, temperature=1.0):
    values = np.asarray(logits, dtype=float) / temperature
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def accuracy(records, temperature=1.0):
    correct = sum(int(to_probabilities(r['logits'], temperature).argmax() == r['label'])
                  for r in records)
    return correct / len(records)


def per_dataset(records, temperature=1.0):
    groups = {}
    for r in records:
        groups.setdefault(r['dataset'], []).append(r)
    out = {}
    for name, rows in groups.items():
        probs = [to_probabilities(r['logits'], temperature) for r in rows]
        out[name] = {'n': len(rows),
                     'accuracy': float(np.mean([int(p.argmax() == r['label'])
                                                for p, r in zip(probs, rows)]))}
    return out


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def model_tokenizer(saved):
    """Reconstruct the text tokenizer for a saved run config (CPU, no rank needed)."""
    import torch
    from train import load_text
    return load_text(saved['text_model'], torch.device('cpu'), saved['lora_rank'])[0]


def torch_device(name):
    import torch
    return torch.device(name if name != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))