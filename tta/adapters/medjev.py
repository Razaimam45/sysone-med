"""TTA adapter for the local MedJev checkpoint.

MedJev is text-only (it rejects multimodal rows), so this adapter restricts
scoring to the text datasets in the manifest. RLCD is supported: the checkpoint
exposes trainable LoRA parameters via ``DecisionModel.trainable_parameters()``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / 'src')])
from med_jevify.medjev_adapter import (MedJevAdapter, free_device, _resolve_dtype,  # noqa: E402
                                       state_text, is_multimodal, medjev_request)
from med_jevify.decision_data import load_manifest

TEXT_TASKS = ('pubmedqa', 'med_qa', 'medmcqa')


class MedJevTtaAdapter:
    """Score text manifest rows through a MedJev checkpoint and expose logits."""

    def __init__(self, run: str = "junma/MedJev-Qwen3.5-0.8B", device: str = "auto",
                 dtype: str = "bf16", manifest: str = "data/prepared/manifest.json"):
        self.device = torch.device(free_device(device))
        self.manifest = manifest
        self.splits = load_manifest(manifest)[1]
        self._inner = MedJevAdapter(run=run, device=str(self.device), dtype=dtype)
        self._temp = 1.0
        # MedJev scores one record at a time; tokenizer/max_length/device are
        # exposed for interface compatibility with the method layer.
        self.tokenizer = self._inner.tok
        self.max_length = self._inner.max_state

    def temperature(self) -> float:
        return self._temp

    def requires_grad(self) -> bool:
        return len(self._inner.model.trainable_parameters()) > 0

    def score_logits(self, examples: Sequence[dict], batch_size: int) -> list[np.ndarray]:
        if any(is_multimodal(ex) for ex in examples):
            raise ValueError('MedJev is text-only; multimodal rows are not supported.')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        # MedJev's pointer head is per-record, so score one at a time.
        logits = []
        for ex in examples:
            probs = self._inner._score_one(ex)
            logits.append(np.log(np.clip(probs, 1e-12, 1.0)))
        return logits

    # --- RLCD support -----------------------------------------------------

    def snapshot_trainable(self) -> list:
        return self._inner.model.trainable_parameters()

    def trainable(self) -> list:
        return self._inner.model.trainable_parameters()

    def clone_trainable(self, trainable: list) -> list:
        return {id(p): p.detach().clone() for p in trainable}

    def restore_trainable(self, trainable: list, snapshot: dict) -> None:
        for p in trainable:
            p.data.copy_(snapshot[id(p)])

    def collate(self, rows):
        # MedJev scores one record at a time; return rows as-is for per-record processing
        return {"rows": list(rows)}

    def forward_batch(self, batch):  # pragma: no cover - protocol stub
        raise RuntimeError('MedJev scores one record at a time; no batch forward')


def build(run: str = "junma/MedJev-Qwen3.5-0.8B", device: str = "auto",
          dtype: str = "bf16", manifest: str = "data/prepared/manifest.json") -> MedJevTtaAdapter:
    return MedJevTtaAdapter(run=run, device=device, dtype=dtype, manifest=manifest)