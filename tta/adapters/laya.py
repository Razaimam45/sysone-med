"""TTA adapter for the Laya text-only NAR baseline.

Laya is a closed package (``convaiinnovations/laya``) that returns choice
probabilities, not raw logits. We recover logits as ``log(probabilities)`` --
temperature scaling on these reproduces ``softmax(logits / T)`` exactly, since
the additive constant cancels in the softmax. RLCD is not supported: the
package exposes no gradient path.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / 'src')])
from med_jevify.decision_data import load_manifest
from med_jevify.non_ar import LayaAdapter  # noqa: E402


class LayaTtaAdapter:
    """Score manifest rows through a Laya checkpoint and expose log-probabilities."""

    def __init__(self, model: str = "convaiinnovations/laya", device: str = "auto",
                 fast: bool = False, manifest: str = "data/prepared/manifest.json"):
        self.device = device
        self.manifest = manifest
        self.splits = load_manifest(manifest)[1]
        self._inner = LayaAdapter(model=model, device=device, fast=fast)
        self._temp = 1.0

    def temperature(self) -> float:
        return self._temp

    def requires_grad(self) -> bool:
        return False

    def score_logits(self, examples: Sequence[dict], batch_size: int) -> list[np.ndarray]:
        probs, _ = self._inner.score(examples, batch_size)
        # log(probabilities) is a valid logit representation: softmax(log p / T)
        # == p^(1/T) / sum(p^(1/T)), so temperature scaling is preserved.
        return [np.log(np.clip(p, 1e-12, 1.0)) for p in probs]

    # RLCD is unsupported: Laya is a closed package with no gradient path.
    def collate(self, rows):  # pragma: no cover - protocol stub
        raise RuntimeError('Laya does not support RLCD (closed package, no gradients)')

    def forward_batch(self, batch):  # pragma: no cover - protocol stub
        raise RuntimeError('Laya does not support RLCD (closed package, no gradients)')

    def snapshot_trainable(self):  # pragma: no cover - protocol stub
        raise RuntimeError('Laya does not support RLCD (closed package, no gradients)')

    def clone_trainable(self, trainable):  # pragma: no cover - protocol stub
        raise RuntimeError('Laya does not support RLCD (closed package, no gradients)')

    def restore_trainable(self, trainable, snapshot):  # pragma: no cover - protocol stub
        raise RuntimeError('Laya does not support RLCD (closed package, no gradients)')


def build(model: str = "convaiinnovations/laya", device: str = "auto",
          fast: bool = False, manifest: str = "data/prepared/manifest.json") -> LayaTtaAdapter:
    return LayaTtaAdapter(model=model, device=device, fast=fast, manifest=manifest)