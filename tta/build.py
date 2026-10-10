"""Adapter factory: build a baseline-specific adapter for TTA.

Each baseline exposes a different checkpoint format and inference path, so
we need a factory to construct the right adapter for each baseline. The
factory uses the Adapter protocol to expose a uniform interface to the TTA
methods.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Protocol

from tta.base import Adapter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training')])


class AdapterFactory(Protocol):
    """Build a baseline-specific adapter for TTA."""

    def __call__(self, checkpoint: str, device: str, manifest: str, batch_size: int) -> Adapter:
        ...


def build_adapter(baseline: str, checkpoint: str, device: str, manifest: str,
                   batch_size: int, backbone: str | None = None) -> Adapter:
    """Factory: build the right adapter for the given baseline.

    ``backbone`` selects the pretrained backbone for MedGemma-Direct-style
    baselines (medgemma, qwen, internvl3). It is ignored by baselines that do
    not use the MedGemma-Direct adapter.
    """
    if baseline == 'laya_med':
        from tta.adapters.laya_med import build as build_laya_med
        return build_laya_med(checkpoint, device, manifest, batch_size)
    elif baseline == 'laya':
        from tta.adapters.laya import build as build_laya
        return build_laya(model=checkpoint or "convaiinnovations/laya", device=device, manifest=manifest)
    elif baseline == 'medjev':
        from tta.adapters.medjev import build as build_medjev
        return build_medjev(checkpoint or "junma/MedJev-Qwen3.5-0.8B", device, manifest=manifest)
    elif baseline in ('medgemma_direct_headonly', 'medgemma_direct_lora'):
        from tta.adapters.medgemma_direct import build as build_medgemma_direct
        return build_medgemma_direct(checkpoint, device, lora=baseline == 'medgemma_direct_lora',
                                     backbone=backbone, manifest=manifest)
    elif baseline == 'medgemma_ll':
        from tta.adapters.medgemma_ll import build as build_medgemma_ll
        return build_medgemma_ll(model=checkpoint or "google/medgemma-4b-it", device=device, dtype='auto',
                                no_think=True, manifest=manifest)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")