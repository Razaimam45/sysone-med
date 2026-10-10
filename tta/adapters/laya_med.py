"""TTA adapter for the Laya-Med checkpoint.

Wraps the existing ``tta/common.py`` loader so the four TTA methods can run on
Laya-Med without modification. RLCD is supported: the checkpoint's trainable
parameters (fusion head + LoRA) are snapshot/restored per batch.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training'), str(ROOT / 'tta')])
from tta.common import load_bundle, torch_device  # noqa: E402
from train import move  # noqa: E402
from laya_med.data import MedicalCollator, MedicalDecisionDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class LayaMedAdapter:
    """Score manifest rows through a Laya-Med checkpoint and expose raw logits."""

    def __init__(self, checkpoint: str, device: str = 'auto',
                 manifest: str | None = None, batch_size: int = 8):
        self.device = torch_device(device)
        self.model, self.tokenizer, self.saved, self.processor, self.manifest, self.splits = \
            load_bundle(checkpoint, self.device, manifest)
        self.max_length = self.saved['max_length']
        self.batch_size = batch_size
        self._temp = float((self.saved.get('calibration') or {}).get('temperature', 1.0) or 1.0)

    # --- Adapter protocol -------------------------------------------------

    def temperature(self) -> float:
        return self._temp

    def requires_grad(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())

    def score_logits(self, examples: Sequence[dict], batch_size: int) -> list[np.ndarray]:
        from tta.common import collect_logits
        # common.collect_logits runs the raw LayaMedModel through train.forward.
        records = collect_logits(self.model, examples, self.processor, self.tokenizer,
                                 self.max_length, self.device, batch_size)
        return [np.asarray(r['logits'], dtype=float) for r in records]

    # --- RLCD support -----------------------------------------------------

    def collate(self, rows: Sequence[dict]) -> dict:
        return next(iter(DataLoader(MedicalDecisionDataset(list(rows)),
                                    batch_size=len(rows),
                                    collate_fn=MedicalCollator(self.tokenizer, self.processor,
                                                               self.max_length))))

    def forward_batch(self, batch: dict) -> list[torch.Tensor]:
        from train import forward
        moved = move(batch, self.device)
        return list(forward(self.model, moved))

    def forward_batch_grad(self, batch: dict) -> list[torch.Tensor]:
        """Forward with gradients enabled (for RLCD fine-tuning)."""
        from train import forward
        return list(forward(self.model, move(batch, self.device)))

    def snapshot_trainable(self) -> list:
        return [p for p in self.model.parameters() if p.requires_grad]

    def clone_trainable(self, trainable: list) -> list:
        return {id(p): p.detach().clone() for p in trainable}

    def restore_trainable(self, trainable: list, snapshot: dict) -> None:
        for p in trainable:
            p.data.copy_(snapshot[id(p)])


def build(checkpoint: str, device: str = 'auto', manifest: str | None = None,
          batch_size: int = 8) -> LayaMedAdapter:
    return LayaMedAdapter(checkpoint, device=device, manifest=manifest, batch_size=batch_size)