"""Shared TTA infrastructure: the Adapter protocol, logit collection, metrics, IO.

Every baseline that exposes per-option decision logits implements the Adapter
protocol here. The TTA methods are
written against the protocol, so they run on any baseline unchanged -- the only
per-baseline code is the adapter itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training')])


class Adapter(Protocol):
    """Per-baseline bridge from manifest rows to per-option logits.

    Implementations must provide :meth:`score_logits` (used by routing,
    conformal and drift_cal) and :attr:`temperature`. RLCD-capable adapters
    additionally expose :meth:`collate`, :meth:`forward_batch`,
    :meth:`snapshot_trainable` and :meth:`restore_trainable`.
    """

    def score_logits(self, examples: Sequence[dict], batch_size: int) -> list[np.ndarray]:
        """Return raw per-option scores (logits) for each example."""
        ...

    def temperature(self) -> float:
        """The checkpoint's held-out calibration temperature (1.0 if none)."""
        ...

    def requires_grad(self) -> bool:
        """Whether the adapter exposes trainable parameters (for RLCD)."""
        ...


def rows_for(splits: dict, datasets: list[str] | None = None,
             max_examples: int | None = None, split: str = 'test') -> list[dict]:
    """Collect rows from a split, optionally restricted to named datasets and capped."""
    names = datasets or sorted({r['dataset'] for r in splits[split]})
    rows = []
    for name in names:
        selected = [r for r in splits[split] if r['dataset'] == name]
        if not selected:
            raise ValueError(f'No manifest {split} partition for requested dataset {name}')
        rows.extend(selected[:max_examples] if max_examples else selected)
    return rows


def collect_logits(adapter: Adapter, rows: list[dict], batch_size: int,
                   progress: bool = False, desc: str = 'inference') -> list[dict]:
    """Run inference over `rows` and return per-example logits + row metadata.

    Batching is delegated to the adapter (some backends group by schema, some
    score one at a time); the method layer only needs the resulting logits.
    """
    import torch
    # Labels are needed by some collators, but never supply test truth to a model.
    with torch.inference_mode():
        logits = adapter.score_logits([{**r, 'label': 0} for r in rows], batch_size)
    if len(logits) != len(rows):
        raise RuntimeError(f'adapter returned {len(logits)} logits for {len(rows)} rows')
    records = []
    for row, scores in zip(rows, logits):
        arr = np.asarray(scores, dtype=float)
        records.append({'id': row['id'], 'dataset': row['dataset'], 'label': int(row['label']),
                        'options': list(row['options']), 'logits': arr.tolist()})
    return records


def to_probabilities(logits, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=float) / temperature
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def accuracy(records: list[dict], temperature: float = 1.0) -> float:
    correct = sum(int(to_probabilities(r['logits'], temperature).argmax() == r['label'])
                  for r in records)
    return correct / len(records) if records else 0.0


def per_dataset(records: list[dict], temperature: float = 1.0) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r['dataset'], []).append(r)
    out = {}
    for name, rows in groups.items():
        probs = [to_probabilities(r['logits'], temperature) for r in rows]
        out[name] = {'n': len(rows),
                     'accuracy': float(np.mean([int(p.argmax() == r['label'])
                                                for p, r in zip(probs, rows)]))}
    return out


def ece(confidences, correct, bins: int = 15) -> float:
    """Expected calibration error: |acc - conf| averaged over confidence bins."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(conf)
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        e += (n / total) * abs(corr[mask].mean() - conf[mask].mean())
    return float(e)


def summarize(records: list[dict], temperature: float = 1.0) -> dict:
    """Overall accuracy + ECE + per-dataset accuracy for a set of records."""
    probs = [to_probabilities(r['logits'], temperature) for r in records]
    conf = [float(p.max()) for p in probs]
    corr = [int(p.argmax() == r['label']) for p, r in zip(probs, records)]
    return {'n': len(records), 'accuracy': float(np.mean(corr)) if corr else 0.0,
            'ece': ece(conf, corr), 'per_dataset': per_dataset(records, temperature)}


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def torch_device(name: str):
    import torch
    return torch.device(name if name != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))