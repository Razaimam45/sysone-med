"""TTA adapter for the MedGemma-Direct family (head-only and LoRA).

Both checkpoints share the same architecture (frozen MedGemma backbone +
decision head, optionally with LoRA), so one adapter serves both. The adapter
auto-detects LoRA from the checkpoint. RLCD is supported: the decision head
(and LoRA weights, if present) are trainable.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training')])
from med_jevify.medgemma_direct import MedGemmaDirectAdapter  # noqa: E402
from med_jevify.decision_data import load_manifest
from laya_med.medgemma_direct import MedGemmaDirectCollator  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class MedGemmaDirectTtaAdapter:
    """Score manifest rows through a MedGemma-Direct checkpoint and expose logits."""

    def __init__(self, run: str = "models/medgemma-direct-headonly", device: str = "cuda:0",
                 dtype: str = "bf16", lora: bool | None = None,
                 backbone: str | None = None,
                 manifest: str = "data/prepared/manifest.json"):
        # Resolve "auto" to a concrete device before passing to torch.device()
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.manifest = manifest
        self.splits = load_manifest(manifest)[1]
        self._inner = MedGemmaDirectAdapter(run=str(run), device=str(self.device),
                                           dtype=dtype, lora=lora, backbone=backbone)
        self._temp = 1.0
        self.tokenizer = self._inner.processor.tokenizer
        self.max_length = self._inner.max_length

    def temperature(self) -> float:
        return self._temp

    def requires_grad(self) -> bool:
        return any(p.requires_grad for p in self._inner.model.parameters())

    def score_logits(self, examples: Sequence[dict], batch_size: int) -> list[np.ndarray]:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        # MedGemma-Direct's pointer head is per-record; score one at a time.
        return [self._inner._forward(ex) for ex in examples]

    # --- RLCD support -----------------------------------------------------

    def trainable(self) -> list:
        return [p for p in self._inner.model.parameters() if p.requires_grad]

    def clone_trainable(self, trainable: list) -> list:
        return {id(p): p.detach().clone() for p in trainable}

    def restore_trainable(self, trainable: list, snapshot: dict) -> None:
        for p in trainable:
            p.data.copy_(snapshot[id(p)])

    def collate(self, rows: Sequence[dict]) -> dict:
        wrapped = [self._inner._wrap_row(r) for r in rows]
        return self._inner.collator(wrapped)

    def forward_batch(self, batch: dict) -> list[np.ndarray]:
        moved = self._inner._move_batch(batch)
        amp = self.device.type == "cuda"
        with torch.inference_mode():
            if amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    scores = self._inner.model(batch["groups"], batch["option_counts"])
            else:
                scores = self._inner.model(batch["groups"], batch["option_counts"])
        return [s.float().cpu().numpy() for s in scores]


def build(run: str = "models/medgemma-direct-headonly", device: str = "cuda:0",
          dtype: str = "bf16", lora: bool | None = None,
          backbone: str | None = None, manifest: str = "data/prepared/manifest.json") -> MedGemmaDirectTtaAdapter:
    return MedGemmaDirectTtaAdapter(run=run, device=device, dtype=dtype, lora=lora,
                                    backbone=backbone, manifest=manifest)