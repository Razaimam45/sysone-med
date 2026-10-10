#!/usr/bin/env python3
"""Evaluate manifest-bound decision checkpoints on untouched test groups."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training'), str(ROOT / 'utils')])
from laya_med.data import MedicalCollator, MedicalDecisionDataset
from laya_med.model import LayaMedModel
from laya_med.metrics import summarize
from med_jevify.metrics import brier, classwise_ece, ece, jsd, mean_confidence, nll, latency_summary
from med_jevify.decision_data import file_hash, load_manifest
from train import asset_fingerprint, dim, forward, load_text, load_checkpoint_vision, move, resolve_asset, verify_encoding


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def result_rows(report, records, latencies, model=None):
    """Only emit the calibrated row; the raw (temperature=1) row is not reported."""
    model = model or report.get('model', 'laya-med')
    rows = []
    for dataset in sorted({record['dataset'] for record in records}):
        selected = [record for record in records if record['dataset'] == dataset]
        durations = [latencies[index] for index, record in enumerate(records) if record['dataset'] == dataset]
        labels = [record['label'] for record in selected]
        scale = report['temperature']
        probabilities = []
        for record in selected:
            values = np.asarray(record['logits'], dtype=float) / scale
            values -= values.max()
            probabilities.append(np.exp(values) / np.exp(values).sum())
        p50, p95, throughput = latency_summary(durations)
        rows.append({'model': model, 'architecture': 'non-AR', 'task': dataset, 'mode': 'laya_med_calibrated',
                     'accuracy': float(np.mean([int(np.argmax(prob) == label) for prob, label in zip(probabilities, labels)])),
                     'mean_confidence': mean_confidence(probabilities), 'ece': ece(probabilities, labels),
                     'classwise_ece': classwise_ece(probabilities, labels), 'nll': nll(probabilities, labels),
                     'brier': brier(probabilities, labels), 'p50_latency': p50, 'p95_latency': p95,
                     'throughput': throughput, 'n': len(labels), 'temperature': scale,
                     'calibration_n': report['examples'],
                     'jsd': jsd(probabilities, labels), 'mean_label_mass': None, 'invalid_predictions': 0,
                     'probability_basis': 'direct non-autoregressive option logits; held-out calibration split',
                     'timing': report['timing']['definition']})
    return rows


def append_results(rows, results_dir, model='laya-med'):
    result_path = Path(results_dir) / 'results.json'
    existing = json.loads(result_path.read_text()) if result_path.exists() else []
    existing = [row for row in existing
                if not (row.get('model') == model and row.get('architecture') == 'non-AR')]
    existing.extend(rows)
    result_path.write_text(json.dumps(existing, indent=2))
    from make_report import write_report
    write_report(existing, results_dir)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(ROOT / 'models/laya-med/best.pt'))
    p.add_argument('--manifest', help='Must have the same fingerprint as the training manifest')
    p.add_argument('--split', choices=['test', 'calibration'], default='test')
    p.add_argument('--datasets', nargs='+', help='Default: every dataset with held-out test cases')
    p.add_argument('--text-model'); p.add_argument('--vision-model')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='auto')
    p.add_argument('--ablation', choices=['none', 'no-images', 'shuffle-images'], default='none')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-examples', type=int, help='Optional cap per dataset for an explicitly partial smoke evaluation')
    p.add_argument('--output', type=Path)
    p.add_argument('--append-results', type=Path, help='Append Laya-Med rows to an existing results directory')
    p.add_argument('--model', default='laya-med', help='model label written into appended result rows')
    args = p.parse_args(argv)
    if args.batch_size <= 0 or args.max_examples is not None and args.max_examples <= 0:
        p.error('Batch size and example cap must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 2:
        raise ValueError('Legacy checkpoint lacks audited split/config metadata; retrain with current pipeline')
    calibration = checkpoint.get('calibration')
    if calibration is None:
        raise ValueError('Checkpoint has no held-out calibration temperature')
    temperature = float(calibration['temperature'])
    saved = checkpoint['args']
    manifest_path = args.manifest or saved['manifest']
    if file_hash(manifest_path) != checkpoint['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from training manifest')
    manifest, splits = load_manifest(manifest_path, verify_images=True)
    names = args.datasets or sorted({r['dataset'] for r in splits[args.split]})
    rows = []
    for name in names:
        selected = [r for r in splits[args.split] if r['dataset'] == name]
        if not selected:
            raise ValueError(f'No manifest {args.split} partition for requested dataset {name}')
        rows.extend(selected[:args.max_examples] if args.max_examples else selected)
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    text_path = resolve_asset(args.text_model or saved['text_model'])
    if asset_fingerprint(text_path)['files'] != checkpoint['assets']['text']['files']:
        raise ValueError('Text assets differ from training')
    tokenizer, text = load_text(text_path, device, saved['lora_rank'])
    verify_encoding(tokenizer, manifest, saved['max_length'])
    processor = vision = None
    if not saved['no_images']:
        vision_path = resolve_asset(args.vision_model or saved['vision_model'])
        if asset_fingerprint(vision_path)['files'] != checkpoint['assets']['vision']['files']:
            raise ValueError('Vision assets differ from training')
        processor, vision = load_checkpoint_vision(vision_path, device, saved)
    elif args.ablation != 'none':
        raise ValueError('Image ablations require a multimodal checkpoint')
    model = LayaMedModel(text, vision, dim(text), dim(vision) if vision is not None else 0,
                         saved['hidden_dim'], saved.get('fusion', 'mlp')).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    dataset = MedicalDecisionDataset(rows)
    if args.ablation == 'no-images':
        processor = None
    if args.ablation == 'shuffle-images':
        rng = random.Random(args.seed)
        for name in names:
            indices = [i for i, row in enumerate(dataset.rows) if row.dataset == name and row.image]
            # Shuffle unique image identities, then rotate to guarantee a different image.
            images = list(dict.fromkeys(dataset.rows[i].image for i in indices)); rng.shuffle(images)
            if len(images) == 1:
                raise ValueError(f'{name}: need at least two distinct images for image shuffle')
            mapping = dict(zip(images, images[1:] + images[:1]))
            for i in indices:
                dataset.rows[i] = replace(dataset.rows[i], image=mapping[dataset.rows[i].image])
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=MedicalCollator(tokenizer, processor, saved['max_length']))
    records, latencies = [], []
    with torch.inference_mode():
        forward(model, move(next(iter(loader)), device))
        synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for batch in tqdm(loader, desc=f"{args.split}/{args.ablation}", unit="batch"):
            moved = move(batch, device); synchronize(device)
            before = time.perf_counter()
            logits = forward(model, moved); synchronize(device)
            elapsed = time.perf_counter() - before
            latencies.extend([elapsed / len(batch['rows'])] * len(batch['rows']))
            for scores, row in zip(logits, batch['rows']):
                records.append({'id': row.id, 'dataset': row.dataset, 'label': row.label,
                                'options': list(row.options),
                                'logits': scores.float().cpu().tolist(),
                                'latency_seconds': elapsed / len(batch['rows']),
                                'temperature': temperature, 'ablation': args.ablation})
        synchronize(device)
        duration = time.perf_counter() - start
    report = {'checkpoint': str(Path(args.checkpoint).resolve()), 'manifest_sha256': checkpoint['manifest_sha256'],
              'split': args.split,
              'ablation': args.ablation, 'seed': args.seed, 'batch_size': args.batch_size,
              'device': str(device), 'partial': args.max_examples is not None, 'examples': len(records),
              'temperature': temperature, 'raw': summarize(records), 'calibrated': summarize(records, temperature),
              'timing': {'definition': 'FP32 eval; one warmup; synchronized forward latency; pipeline includes preprocessing/transfers/output collection; loading excluded',
                         'forward_ms_per_example_p50': float(np.median(latencies) * 1000),
                         'forward_ms_per_example_p95': float(np.percentile(latencies, 95) * 1000),
                         'pipeline_examples_per_second': len(records) / duration,
                         'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}}
    output = args.output or Path(args.checkpoint).parent / f'evaluation-{args.split}-{args.ablation}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    output.with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    if args.append_results:
        append_results(result_rows(report, records, latencies, args.model), args.append_results)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
