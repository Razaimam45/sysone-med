#!/usr/bin/env python3
"""Train a Laya/Jev-inspired multimodal non-autoregressive decision scorer."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
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
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training')])
from laya_med.data import MedicalCollator, MedicalDecisionDataset
from laya_med.model import LayaMedModel
from laya_med.metrics import fit_temperature, summarize
from med_jevify.decision_data import file_hash, load_manifest, read_jsonl


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_asset(name, revision=None):
    if Path(name).is_dir():
        return str(Path(name).resolve())
    from huggingface_hub import snapshot_download
    return str(Path(snapshot_download(name, revision=revision)).resolve())


def asset_fingerprint(path):
    root = Path(path)
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.bin', '.txt', '.model')))
    return {'path': str(root), 'revision': root.name if root.parent.name == 'snapshots' else None,
            'files': {p.name: file_hash(p) for p in files}}


def load_text(name, device, rank):
    if rank < 0:
        raise ValueError('LoRA rank must be nonnegative')
    tokenizer = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name)
    encoder.requires_grad_(False)
    if rank:
        from peft import LoraConfig, TaskType, get_peft_model
        encoder = get_peft_model(encoder, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=.05,
                    target_modules=['query', 'key', 'value'], task_type=TaskType.FEATURE_EXTRACTION))
        if any(p.requires_grad and 'lora_' not in n for n, p in encoder.named_parameters()):
            raise RuntimeError('Unexpected trainable base encoder parameters')
    return tokenizer, encoder.to(device)


class OpenClipImageProcessor:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, return_tensors='pt'):
        return {'pixel_values': torch.stack([self.transform(image) for image in images])}


class OpenClipVisionEncoder(nn.Module):
    def __init__(self, visual, output_dim):
        super().__init__()
        self.visual = visual
        self.config = SimpleNamespace(hidden_size=output_dim)

    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=self.visual(pixel_values))


def configure_vision_training(encoder, train, rank=0):
    """Freeze, fully fine-tune, or attach attention LoRA without text-only forwarding."""
    if rank < 0:
        raise ValueError('LoRA rank must be nonnegative')
    if not train or rank == 0:
        return encoder.requires_grad_(train)
    from peft import LoraConfig, get_peft_model
    target = getattr(encoder, 'visual', encoder)
    # Use full module paths: a generic 'proj' also matches the patch embedding
    # and output head, rather than just the attention output projection.
    targets = []
    for name, module in target.named_modules():
        parts = name.split('.')
        attention_projection = parts[-1] in {'qkv', 'query', 'key', 'value',
                                               'q_proj', 'k_proj', 'v_proj', 'in_proj', 'out_proj'}
        attention_projection |= parts[-1] == 'proj' and any(p in {'attn', 'attention', 'self_attn'} for p in parts[:-1])
        if isinstance(module, nn.Linear) and attention_projection:
            targets.append(name)
    if not targets:
        raise ValueError('No supported visual attention projections found for LoRA')
    # No FEATURE_EXTRACTION task type: that wrapper injects input_ids and other
    # text arguments that timm vision towers do not accept.
    target = get_peft_model(target, LoraConfig(r=rank, lora_alpha=2 * rank,
                                              lora_dropout=.05, target_modules=targets))
    if any(p.requires_grad and 'lora_' not in name for name, p in target.named_parameters()):
        raise RuntimeError('Unexpected trainable base vision encoder parameters')
    if hasattr(encoder, 'visual'):
        encoder.visual = target
    else:
        encoder = target
    return encoder


def load_checkpoint_vision(name, device, saved_args):
    """Recreate the saved visual architecture, then freeze it for evaluation."""
    processor, encoder = load_vision(name, device, saved_args.get('train_vision', False),
                                     saved_args.get('lora_rank', 0))
    return processor, encoder.requires_grad_(False)


def load_vision(name, device, train, rank=0):
    root = Path(resolve_asset(name))
    if (root / 'open_clip_config.json').is_file():
        from open_clip.timm_model import TimmModel
        from open_clip.transform import PreprocessCfg, image_transform_v2
        config = json.loads((root / 'open_clip_config.json').read_text())
        cfg = config['model_cfg']; vision = cfg['vision_cfg']
        if 'timm_model_name' not in vision:
            raise ValueError('Only timm OpenCLIP visual towers are supported')
        visual = TimmModel(vision['timm_model_name'], embed_dim=cfg['embed_dim'],
                          image_size=vision.get('image_size', 224), pool=vision.get('timm_pool', 'avg'),
                          proj=vision.get('timm_proj', 'linear'),
                          proj_bias=vision.get('timm_proj_bias', False), pretrained=False)
        weights = torch.load(root / 'open_clip_pytorch_model.bin', map_location='cpu', weights_only=True)
        weights = weights.get('state_dict', weights)
        visual.load_state_dict({key.removeprefix('visual.'): value for key, value in weights.items()
                                if key.startswith('visual.')}, strict=True)
        # No unused CLIP text tower or implicit text-model download.
        processor = OpenClipImageProcessor(image_transform_v2(
            PreprocessCfg(size=vision.get('image_size', 224), **config.get('preprocess_cfg', {})), is_train=False))
        encoder = OpenClipVisionEncoder(visual, cfg['embed_dim'])
    else:
        processor = AutoImageProcessor.from_pretrained(root)
        encoder = AutoModel.from_pretrained(root)
        if hasattr(encoder, 'vision_model'):
            encoder = encoder.vision_model
    encoder = configure_vision_training(encoder, train, rank)
    return processor, encoder.to(device)


def dim(encoder):
    for key in ('hidden_size', 'projection_dim', 'embed_dim'):
        if getattr(encoder.config, key, None):
            return int(getattr(encoder.config, key))
    raise ValueError('Cannot infer encoder feature dimension')


def move(batch, device):
    for group in ('text', 'images'):
        if group in batch:
            batch[group] = {key: value.to(device) for key, value in batch[group].items()}
    batch['image_indices'] = batch['image_indices'].to(device)
    return batch


def forward(model, batch):
    return model(batch['text'], batch['option_counts'], batch.get('images'), batch['image_indices'])


def batch_loss(logits, rows, device):
    if len(logits) != len(rows) or not rows:
        raise ValueError('Logits must match nonempty row batch')
    return torch.stack([nn.functional.cross_entropy(scores[None], torch.tensor([row.label], device=device))
                        for scores, row in zip(logits, rows)]).mean()


def predict(model, loader, device):
    model.eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            logits = forward(model, move(batch, device))
            for scores, row in zip(logits, batch['rows']):
                records.append({'id': row.id, 'dataset': row.dataset, 'label': row.label,
                                'logits': scores.float().cpu().tolist()})
    return records


def rng_state(generator):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    generator.set_state(state['loader'].cpu())


def save_checkpoint(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume_config(args, checkpoint, manifest_hash, assets):
    ignored = {'resume', 'device', 'manifest', 'text_revision', 'vision_revision',
               'text_model', 'vision_model'}

    def saved_value(key):
        value = checkpoint['args'].get(key)
        if key == 'fusion' and value is None:
            return 'mlp'
        return value

    for key, value in vars(args).items():
        if key in ignored:
            continue
        if key == 'epochs':
            if value < saved_value(key):
                raise ValueError('Resume configuration differs from original run')
            continue
        if saved_value(key) != value:
            raise ValueError('Resume configuration differs from original run')

    if checkpoint['manifest_sha256'] != manifest_hash or checkpoint['assets'] != assets:
        raise ValueError('Resume data/model fingerprints differ')


def verify_encoding(tokenizer, manifest, max_length):
    fingerprint = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if manifest['encoding']['tokenizer_sha256'] != fingerprint or manifest['encoding']['max_length'] != max_length:
        raise ValueError('Tokenizer/length differs from prepared manifest; re-prepare data')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default=str(ROOT / 'data/prepared/manifest.json'))
    p.add_argument('--text-model', default=str(ROOT / 'data/models/text'))
    p.add_argument('--vision-model', default=str(ROOT / 'data/models/vision'))
    p.add_argument('--text-revision'); p.add_argument('--vision-revision')
    p.add_argument('--output-dir', default=str(ROOT / 'models/laya-med'))
    p.add_argument('--resume', help='Resume last.pt at an epoch boundary, with the same run configuration')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--grad-accumulation', type=int, default=4)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--hidden-dim', type=int, default=768)
    p.add_argument('--fusion', choices=['mlp', 'cross_attention'], default='mlp')
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--sampling', choices=['balanced', 'proportional'], default='balanced')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--no-images', action='store_true', help='Text-only ablation on the same cases/splits')
    p.add_argument('--train-vision', action='store_true', help='Train vision encoder (full FT if --lora-rank=0, LoRA if --lora-rank>0)')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accumulation, args.hidden_dim) <= 0 or args.hidden_dim < 2 or args.learning_rate <= 0:
        p.error('Epochs, batch/accumulation sizes, hidden dimension and learning rate must be positive')
    if args.no_images and args.train_vision:
        p.error('--no-images and --train-vision are incompatible')
    seed_all(args.seed)
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    output = Path(args.output_dir).resolve()
    if (output / 'last.pt').exists() and not args.resume:
        p.error('Output already contains a run; use --resume or a new output directory')
    print('Verifying split and image fingerprints...', flush=True)
    manifest, splits = load_manifest(args.manifest, verify_images=True)
    for split in ('train', 'validation', 'calibration'):
        if not splits[split]:
            raise ValueError(f'Empty {split} partition')
    manifest_hash = file_hash(args.manifest)
    args.text_model = resolve_asset(args.text_model, args.text_revision)
    if not args.no_images:
        args.vision_model = resolve_asset(args.vision_model, args.vision_revision)
    assets = {'text': asset_fingerprint(args.text_model)}
    if not args.no_images:
        assets['vision'] = asset_fingerprint(args.vision_model)
    tokenizer, text = load_text(args.text_model, device, args.lora_rank)
    verify_encoding(tokenizer, manifest, args.max_length)
    processor = vision = None
    if not args.no_images:
        processor, vision = load_vision(args.vision_model, device, args.train_vision, args.lora_rank)
    model = LayaMedModel(text, vision, dim(text), dim(vision) if vision is not None else 0, args.hidden_dim, args.fusion).to(device)
    collator = MedicalCollator(tokenizer, processor, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    counts = Counter(r['dataset'] for r in splits['train'])
    sampler = WeightedRandomSampler([1 / counts[r['dataset']] for r in splits['train']],
                                     len(splits['train']), replacement=True, generator=generator) if args.sampling == 'balanced' else None
    train_loader = DataLoader(MedicalDecisionDataset(splits['train']), batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler, collate_fn=collator, generator=generator)
    loaders = {s: DataLoader(MedicalDecisionDataset(splits[s]), batch_size=args.batch_size, collate_fn=collator)
               for s in ('validation', 'calibration')}
    trainable = [param for param in model.parameters() if param.requires_grad]
    parameter_counts = {'total': sum(p.numel() for p in model.parameters()),
                        'trainable': sum(p.numel() for p in trainable),
                        'text_trainable': sum(p.numel() for p in text.parameters() if p.requires_grad),
                        'vision_trainable': sum(p.numel() for p in vision.parameters() if p.requires_grad) if vision is not None else 0}
    print(json.dumps({'parameters': parameter_counts, 'train_datasets': counts}), flush=True)
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=.01)
    updates = math.ceil(len(train_loader) / args.grad_accumulation) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, updates // 10), updates)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, start_epoch, history = -1., 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        validate_resume_config(args, checkpoint, manifest_hash, assets)
        if not (output / 'best.pt').is_file():
            raise ValueError('Resume requires the original best.pt alongside last.pt')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        best, start_epoch, history = checkpoint['best_metric'], checkpoint['epoch'], checkpoint['history']
        restore_rng(checkpoint['rng'], generator)
    output.mkdir(parents=True, exist_ok=True)
    run = {'schema_version': 2, 'args': vars(args), 'manifest_sha256': manifest_hash, 'assets': assets,
           'parameters': parameter_counts, 'selection_metric': 'macro_dataset_accuracy',
           'versions': {name: version(name) for name in ('torch', 'transformers', 'peft', 'open_clip_torch', 'numpy')},
           'hardware': str(device), 'calibration': None}
    (output / 'run.json').write_text(json.dumps(run, indent=2))
    for epoch in range(start_epoch, args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        pending_examples, train_sum, train_count = 0, 0., 0
        progress = tqdm(train_loader, desc=f'epoch {epoch + 1}/{args.epochs} train', unit='batch', dynamic_ncols=True)
        for step, batch in enumerate(progress, 1):
            batch = move(batch, device)
            n = len(batch['rows'])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = batch_loss(forward(model, batch), batch['rows'], device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}, batch {step}')
            scaler.scale(loss * n).backward()
            pending_examples += n; train_sum += float(loss.detach()) * n; train_count += n
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                for param in trainable:
                    if param.grad is not None:
                        param.grad.div_(pending_examples)
                nn.utils.clip_grad_norm_(trainable, 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True); pending_examples = 0
            progress.set_postfix(loss=f'{float(loss.detach()):.4f}', lr=f'{scheduler.get_last_lr()[0]:.2e}')
        progress.close()
        validation_progress = tqdm(loaders['validation'], desc=f'epoch {epoch + 1}/{args.epochs} validation', unit='batch', dynamic_ncols=True)
        validation_records = predict(model, validation_progress, device)
        validation_progress.close()
        metrics = summarize(validation_records)
        entry = {'epoch': epoch + 1, 'train_loss': train_sum / train_count, 'validation': metrics}
        history.append(entry); print(json.dumps(entry), flush=True)
        improved = metrics['macro_accuracy'] > best
        best = max(best, metrics['macro_accuracy'])
        checkpoint = dict(run, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch + 1,
                          best_metric=best, rng=rng_state(generator), history=history)
        if improved:
            save_checkpoint(checkpoint, output / 'best.pt')
        save_checkpoint(checkpoint, output / 'last.pt')
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    records = predict(model, loaders['calibration'], device)
    calibration = fit_temperature(records)
    checkpoint['calibration'] = calibration
    save_checkpoint(checkpoint, output / 'best.pt')
    (output / 'calibration.json').write_text(json.dumps({'fit': calibration, 'predictions': records}, indent=2))
    (output / 'metadata.json').write_text(json.dumps(dict(run, calibration=calibration, history=history), indent=2))
    print(f'Saved {output}; temperature={calibration["temperature"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
