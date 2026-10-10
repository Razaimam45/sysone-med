#!/usr/bin/env python3
"""Train a MedGemma multimodal direct-candidate-scoring baseline."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "src"), str(ROOT / "training")])
from laya_med.data import MedicalDecisionDataset
from laya_med.medgemma_direct import MedGemmaDirectCollator, MedGemmaDirectModel
from laya_med.metrics import fit_temperature, summarize
from med_jevify.decision_data import file_hash, load_manifest
from train import asset_fingerprint, resolve_asset, restore_rng, rng_state, save_checkpoint, seed_all


def load_direct_model(model_name: str, device: torch.device, rank: int, dtype: torch.dtype,
                      head_size: int = 512, lora: bool = True,
                      backbone: str | None = None,
                      gradient_checkpointing: bool = True):
    """Build a MedGemma-Direct-style decision model on any registered backbone.

    ``lora=True`` (default): frozen backbone + rank-`rank` LoRA adapters on
    q/k/v/o projections of the language model. ``lora=False``: the backbone is
    fully frozen with no LoRA at all, and only ``decision_head`` is trainable.

    ``backbone`` selects the model class and processor via the
    :mod:`med_jevify.backbones` registry. It defaults to MedGemma when omitted
    so existing invocations are unchanged.
    """
    if lora and rank <= 0:
        raise ValueError("MedGemma-Direct LoRA requires a positive rank")
    from med_jevify.backbones import load_backbone, resolve_backbone

    spec = resolve_backbone(backbone) if backbone else resolve_backbone("medgemma")
    backbone_model, processor = load_backbone(spec, dtype, str(device))
    if not hasattr(backbone_model, "model") or not hasattr(backbone_model.model, "language_model"):
        raise ValueError(
            f"Backbone {spec.name!r} does not expose .model.language_model; "
            f"only medgemma and qwen2_5_vl are supported")
    backbone_model.requires_grad_(False)
    if lora:
        from peft import LoraConfig, TaskType, get_peft_model
        language_model = get_peft_model(
            backbone_model.model.language_model,
            LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       task_type=TaskType.FEATURE_EXTRACTION),
        )
        backbone_model.model.language_model = language_model
        # LoRA backprops through the full vision+language stack, so peak memory
        # is dominated by the backbone activations. Gradient checkpointing trades
        # a ~2x compute overhead for a large activation-memory reduction, which is
        # what lets a 3B Qwen2.5-VL fit on a single 48 GiB A6000. Head-only mode
        # caches features to disk and never backprops, so it is left untouched.
        if gradient_checkpointing:
            backbone_model.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    config = getattr(backbone_model.config, "text_config", backbone_model.config)
    hidden_size = int(config.hidden_size)
    model = MedGemmaDirectModel(backbone_model, hidden_size, head_size).to(device)
    trainable_backbone = [name for name, parameter in model.backbone.named_parameters()
                          if parameter.requires_grad]
    if lora and (not trainable_backbone or any("lora_" not in name for name in trainable_backbone)):
        raise RuntimeError(f"{spec.name} backbone must be frozen except for LoRA parameters")
    if not lora and trainable_backbone:
        raise RuntimeError(f"Head-only mode requires a fully frozen {spec.name} backbone")
    return processor, model


def head_only_scores(model, batch, device):
    """Score cached per-candidate features through the (float32) decision head.

    Used by the head-only mode, where the frozen MedGemma backbone is run once
    and its last-token hidden states are cached to disk. Training/eval then
    never re-enters the backbone.
    """
    scores = []
    for features in batch["features"]:
        scores.append(model.decision_head(features.to(device)).squeeze(-1))
    return scores


def cache_features(model, rows, processor, max_length, device, dtype, cache_dir, batch_size,
                   backbone: str | None = None, show_progress: bool = False):
    """Run the frozen MedGemma backbone once per split and cache per-candidate
    last-token hidden states (BEFORE the decision head) to ``cache_dir``.

    The cache is keyed by ``(row.id, candidate_index)`` so it survives across
    epochs and is shared by train/validation/calibration. Image-conditioned
    representations are preserved exactly: the same MedGemmaDirectCollator
    grouping (text-only vs image rows) is reused, so multimodal examples are
    encoded with their image attached.

    ``backbone`` selects the per-backbone collator via the registry; the default
    MedGemma collator is used when omitted. ``show_progress`` wraps the batch
    loop in a tqdm bar so long caches report progress.
    """
    from med_jevify.backbones import collator_for, resolve_backbone
    from torch.utils.data import DataLoader
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_backbone(backbone) if backbone else resolve_backbone("medgemma")
    collator = collator_for(spec, processor, max_length)
    model.eval()
    amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    features_by_row = {}

    iterator = range(0, len(rows), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="caching", total=(len(rows) + batch_size - 1) // batch_size,
                        unit="batch", dynamic_ncols=True)
    for start in iterator:
        chunk = rows[start:start + batch_size]
        loader = DataLoader(MedicalDecisionDataset(chunk), batch_size=len(chunk), collate_fn=collator)

        with torch.inference_mode():
            for batch in loader:
                moved = move_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    feats = model.extract_features(moved["groups"], moved["option_counts"])

                for row, features in zip(batch["rows"], feats):
                    # Handle both objects and dictionaries
                    row_id = row["id"] if isinstance(row, dict) else row.id
                    features_by_row[row_id] = [f.detach().cpu() for f in features]

        for row in chunk:
            # Handle both objects and dictionaries
            row_id = row["id"] if isinstance(row, dict) else row.id
            torch.save({"features": features_by_row[row_id]}, cache_dir / f"{row_id}.pt")
    return features_by_row


class CachedCollator:
    """Collate precomputed per-candidate features into a batch dict.

    Produces ``{"features": [tensor per row], "rows": [...]}`` where each
    features tensor has shape ``(n_options, hidden_size)``. This replaces the
    MedGemmaDirectCollator in head-only training/eval so the backbone is never
    re-run.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    def __call__(self, rows):
        features = []
        for row in rows:
            row_id = row["id"] if isinstance(row, dict) else row.id
            payload = torch.load(self.cache_dir / f"{row_id}.pt", weights_only=True)
            features.append(torch.stack(payload["features"]))
        return {"features": features, "rows": list(rows)}


def cached_loss(model, batch, device):
    """Cross-entropy over cached features for the head-only training objective."""
    rows = batch["rows"]
    losses = [nn.functional.cross_entropy(scores[None],
                 torch.tensor([row.label], dtype=torch.long, device=device))
              for scores, row in zip(head_only_scores(model, batch, device), rows)]
    return torch.stack(losses).mean()


def cached_predict(model, loader, device):
    """Score cached features and return per-example logits records."""
    model.eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            for scores, row in zip(head_only_scores(model, batch, device), batch["rows"]):
                records.append({"id": row.id, "dataset": row.dataset, "label": row.label,
                                "logits": scores.float().cpu().tolist()})
    return records


def select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device.type == "cuda" else torch.float32)
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def move_batch(batch, device):
    for group in batch["groups"]:
        group["inputs"] = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in group["inputs"].items()
        }
    return batch


def forward_batch(model, batch, device, dtype=None):
    """Run the model under the same autocast regime as training.

    The decision head is float32 while the MedGemma backbone is bfloat16, so a
    bare forward outside autocast raises
    ``mat1 and mat2 must have the same dtype``. Training wraps forward_batch in
    autocast; inference must do the same.
    """
    batch = move_batch(batch, device)
    if dtype is not None and device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type=device.type, dtype=dtype):
            return model(batch["groups"], batch["option_counts"])
    return model(batch["groups"], batch["option_counts"])


def batch_loss(logits, rows, device):
    if len(logits) != len(rows) or not rows:
        raise ValueError("Logits must match a nonempty row batch")
    losses = [nn.functional.cross_entropy(scores[None],
               torch.tensor([row.label], dtype=torch.long, device=device))
              for scores, row in zip(logits, rows)]
    return torch.stack(losses).mean()


def predict(model, loader, device, dtype=None):
    model.eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            logits = forward_batch(model, batch, device, dtype)
            for scores, row in zip(logits, batch["rows"]):
                records.append({"id": row.id, "dataset": row.dataset, "label": row.label,
                                "logits": scores.float().cpu().tolist()})
    return records


def trainable_state_dict(model):
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def load_trainable_state_dict(model, saved):
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if current.keys() != saved.keys():
        raise ValueError("MedGemma-Direct trainable parameter names differ from checkpoint")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))


def validate_resume(args, checkpoint, manifest_hash, assets):
    ignored = {"resume", "device", "output_dir", "medgemma_model", "revision", "dtype",
               "no_lora", "cache_dir", "epochs", "backbone",
               "no_gradient_checkpointing", "gradient_checkpointing", "max_samples"}
    saved_args = checkpoint["args"]
    for key, value in vars(args).items():
        if key in ignored:
            continue
        previous = saved_args.get(key)
        if previous != value:
            raise ValueError(f"Resume configuration differs at {key}")
    previous_asset = checkpoint["assets"].get("medgemma", {}).get("files")
    if checkpoint["manifest_sha256"] != manifest_hash or previous_asset != assets["medgemma"]["files"]:
        raise ValueError("Resume data/model fingerprints differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "data/prepared/manifest.json"))
    parser.add_argument("--medgemma-model", default="google/medgemma-4b-it")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", default=str(ROOT / "models/medgemma-direct"))
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--head-size", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing (faster, uses more memory). "
                             "Only relevant for LoRA training.")
    parser.add_argument("--no-lora", action="store_true",
                        help="Head-only mode: freeze all MedGemma params, drop LoRA, "
                             "train only the decision head on cached backbone features")
    parser.add_argument("--cache-dir",
                        help="Directory of cached per-candidate backbone features "
                             "(head-only mode). Created automatically if absent.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--sampling", choices=["balanced", "proportional"], default="balanced")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training/validation/calibration to this many examples total. "
                             "Applied per-split with balanced sampling across datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--backbone", default="medgemma",
                        help="Backbone model for the decision baseline. "
                             "medgemma (default), qwen (Qwen2.5-VL-3B-Instruct), "
                             "or internvl3 (text-only).")
    args = parser.parse_args()
    args.gradient_checkpointing = not args.no_gradient_checkpointing
    if min(args.epochs, args.batch_size, args.grad_accumulation, args.max_length) <= 0 or \
            args.head_size < 2 or args.learning_rate <= 0:
        parser.error("Epoch, batch, length, head, and learning-rate values must be positive")
    if not args.no_lora and args.lora_rank <= 0:
        parser.error("--lora-rank must be positive in LoRA mode")
    seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = select_dtype(args.dtype, device)
    if device.type != "cuda" and dtype != torch.float32:
        parser.error("Low-precision MedGemma-Direct training currently requires CUDA")
    output = Path(args.output_dir).resolve()
    if (output / "last.pt").exists() and not args.resume:
        parser.error("Output already contains a run; use --resume or a new output directory")

    print("Verifying shared split and image fingerprints...", flush=True)
    manifest, splits = load_manifest(args.manifest, verify_images=True)
    if manifest["encoding"]["max_length"] != args.max_length:
        raise ValueError("--max-length must match the prepared manifest token budget")
    for name in ("train", "validation", "calibration"):
        if not splits[name]:
            raise ValueError(f"Empty {name} partition")
    if args.max_samples is not None:
        import random
        rng = random.Random(args.seed)
        for name in ("train", "validation", "calibration"):
            if splits[name]:
                # Balanced subsample per dataset
                by_ds = {}
                for row in splits[name]:
                    by_ds.setdefault(row["dataset"], []).append(row)
                quota = max(1, args.max_samples // max(1, len(by_ds)))
                kept = []
                for ds, rows in by_ds.items():
                    rng.shuffle(rows)
                    kept.extend(rows[:min(quota, len(rows))])
                rng.shuffle(kept)
                if len(kept) > args.max_samples:
                    kept = kept[:args.max_samples]
                print(f"Subsampled {name}: {len(splits[name])} -> {len(kept)} (max-samples={args.max_samples})")
                splits[name] = kept
    manifest_hash = file_hash(args.manifest)
    args.medgemma_model = resolve_asset(args.medgemma_model, args.revision)
    assets = {"medgemma": asset_fingerprint(args.medgemma_model)}
    processor, model = load_direct_model(args.medgemma_model, device, args.lora_rank, dtype,
                                         args.head_size, lora=not args.no_lora,
                                         backbone=args.backbone,
                                         gradient_checkpointing=args.gradient_checkpointing)
    mode_label = "head_only" if args.no_lora else "lora"
    cache_dir = args.cache_dir or str(output / "cache")

    generator = torch.Generator().manual_seed(args.seed)
    counts = Counter(row["dataset"] for row in splits["train"])

    if args.no_lora:
        cache_dir = args.cache_dir or str(output / "cache")
        cache_dir = Path(cache_dir)

        # Check if cache exists for all splits
        cache_exists = True
        for name in ("train", "validation", "calibration"):
            if not splits[name]:
                continue

            # Check if any cached file is missing for the first 10 rows
            for row in splits[name][:10]:
                row_id = row["id"] if isinstance(row, dict) else row.id
                if not (cache_dir / f"{row_id}.pt").exists():
                    cache_exists = False
                    break
            if not cache_exists:
                break

        if not cache_exists:
            print(f"Cache missing. Generating cached features for all splits...")
            for name in ("train", "validation", "calibration"):
                print(f"cache {name}: {len(splits[name])} rows ...", flush=True)
                cache_features(model, splits[name], processor, args.max_length, device, dtype, cache_dir, args.batch_size, backbone=args.backbone, show_progress=True)
        else:
            print(f"Reusing existing cache from {cache_dir}")

        collator = CachedCollator(cache_dir)
        train_loader = DataLoader(MedicalDecisionDataset(splits["train"]), batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collator, generator=generator)
        loaders = {name: DataLoader(MedicalDecisionDataset(splits[name]), batch_size=args.batch_size,
                                    collate_fn=collator)
                   for name in ("validation", "calibration")}
    else:
        print("Not ca")
        from med_jevify.backbones import collator_for, resolve_backbone
        spec = resolve_backbone(args.backbone) if args.backbone else resolve_backbone("medgemma")
        collator = collator_for(spec, processor, args.max_length)
        sampler = (WeightedRandomSampler([1 / counts[row["dataset"]] for row in splits["train"]],
                                         len(splits["train"]), replacement=True, generator=generator)
                   if args.sampling == "balanced" else None)
        train_loader = DataLoader(MedicalDecisionDataset(splits["train"]), batch_size=args.batch_size,
                                  shuffle=sampler is None, sampler=sampler, collate_fn=collator,
                                  generator=generator)
        loaders = {name: DataLoader(MedicalDecisionDataset(splits[name]), batch_size=args.batch_size,
                                    collate_fn=collator)
                   for name in ("validation", "calibration")}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    counts_info = {"total": sum(parameter.numel() for parameter in model.parameters()),
                   "trainable": sum(parameter.numel() for parameter in trainable),
                   "lora": sum(parameter.numel() for name, parameter in model.named_parameters()
                               if parameter.requires_grad and "lora_" in name),
                   "decision_head": sum(parameter.numel() for name, parameter in model.named_parameters()
                                        if parameter.requires_grad and name.startswith("decision_head."))}
    print(json.dumps({"parameters": counts_info, "mode": mode_label,
                      "train_datasets": counts}), flush=True)

    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    updates = math.ceil(len(train_loader) / args.grad_accumulation) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, updates // 10), updates)
    amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    best, start_epoch, history = -1.0, 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_resume(args, checkpoint, manifest_hash, assets)
        if not (output / "best.pt").is_file():
            raise ValueError("Resume requires the original best.pt alongside last.pt")
        load_trainable_state_dict(model, checkpoint["trainable_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best, start_epoch, history = checkpoint["best_metric"], checkpoint["epoch"], checkpoint["history"]
        restore_rng(checkpoint["rng"], generator)

    output.mkdir(parents=True, exist_ok=True)
    model_family = "MedGemma-Direct-HeadOnly" if args.no_lora else "MedGemma-Direct"
    run = {"schema_version": 1, "model_family": model_family, "args": vars(args),
           "manifest_sha256": manifest_hash, "assets": assets, "parameters": counts_info,
           "selection_metric": "macro_dataset_accuracy", "precision": str(dtype),
           "versions": {name: version(name) for name in ("torch", "transformers", "peft", "numpy")},
           "hardware": str(device), "calibration": None}
    (output / "run.json").write_text(json.dumps(run, indent=2))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_examples, train_sum, train_count = 0, 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs} train",
                        unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(progress, 1):
            rows = batch["rows"]
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                if args.no_lora:
                    loss = cached_loss(model, batch, device)
                else:
                    loss = batch_loss(forward_batch(model, batch, device), rows, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, batch {step}")
            example_count = len(rows)
            scaler.scale(loss * example_count).backward()
            pending_examples += example_count
            train_sum += float(loss.detach()) * example_count
            train_count += example_count
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.div_(pending_examples)
                nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending_examples = 0
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")
        progress.close()

        validation_progress = tqdm(loaders["validation"], desc=f"epoch {epoch + 1}/{args.epochs} validation",
                                   unit="batch", dynamic_ncols=True)
        if args.no_lora:
            validation_records = cached_predict(model, validation_progress, device)
        else:
            validation_records = predict(model, validation_progress, device, dtype)
        validation_progress.close()
        metrics = summarize(validation_records)
        entry = {"epoch": epoch + 1, "train_loss": train_sum / train_count, "validation": metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        improved = metrics["macro_accuracy"] > best
        best = max(best, metrics["macro_accuracy"])
        checkpoint = dict(run, trainable_model=trainable_state_dict(model),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          scaler=scaler.state_dict(), epoch=epoch + 1, best_metric=best,
                          rng=rng_state(generator), history=history)
        if improved:
            save_checkpoint(checkpoint, output / "best.pt")
        save_checkpoint(checkpoint, output / "last.pt")

    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    load_trainable_state_dict(model, checkpoint["trainable_model"])
    if args.no_lora:
        calibration_records = cached_predict(model, loaders["calibration"], device)
    else:
        calibration_records = predict(model, loaders["calibration"], device, dtype)
    calibration = fit_temperature(calibration_records)
    checkpoint["calibration"] = calibration
    save_checkpoint(checkpoint, output / "best.pt")
    (output / "calibration.json").write_text(json.dumps({"fit": calibration,
                                                          "predictions": calibration_records}, indent=2))
    (output / "metadata.json").write_text(json.dumps(dict(run, calibration=calibration,
                                                            history=history), indent=2))
    print(f"Saved {output}; temperature={calibration['temperature']:.6f}", flush=True)


if __name__ == "__main__":
    main()
