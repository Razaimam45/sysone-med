#!/usr/bin/env python3
"""Smoke-test the fixed inference path for MedGemma-Direct.

Loads the model the same way train_medgemma_direct.py does (bf16 backbone +
float32 decision head) and runs a few batches through predict() to confirm the
autocast fix prevents the dtype-mismatch crash. No checkpoint is needed: the
weights come fresh from the hub, so this is a pure forward-path test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "src"), str(ROOT / "training")])
from laya_med.data import MedicalDecisionDataset
from laya_med.medgemma_direct import MedGemmaDirectCollator
from train_medgemma_direct import (  # noqa: E402
    load_direct_model, predict, select_dtype,
)
from med_jevify.decision_data import load_manifest  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = select_dtype("auto", device)
    print(f"device={device} dtype={dtype}", flush=True)

    manifest, splits = load_manifest("data/prepared/manifest.json", verify_images=True)
    processor, model = load_direct_model("google/medgemma-4b-it", device, 8, dtype, 512)
    collator = MedGemmaDirectCollator(processor, 512)

    rows = splits["validation"][:8]
    from torch.utils.data import DataLoader
    loader = DataLoader(MedicalDecisionDataset(rows), batch_size=4, collate_fn=collator)
    records = predict(model, loader, device, dtype)
    print(f"OK: {len(records)} validation records scored", flush=True)
    for r in records[:2]:
        print(r["id"], r["dataset"], r["label"], [round(x, 3) for x in r["logits"]], flush=True)


if __name__ == "__main__":
    main()