#!/usr/bin/env python3
"""Run selected frozen-model inference-time policies with one shared model load."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tta.base import collect_logits, rows_for, save_json
from tta.build import build_adapter

METHODS = (
    # 'routing',
    # 'conformal',
    'permutation',
    'permutation_consensus',
    'prior_consensus',
    'memory',
    'selective',
    'calibrated',
    )


AVAILABLE_METHODS = ('routing', 'conformal', 'permutation', 'permutation_consensus',
                     'prior_consensus', 'memory', 'selective', 'calibrated')

def run_tta(args):
    from importlib import import_module
    from tta.report import write_tta_report

    if args.batch_size < 1 or args.max_samples < 1:
        raise ValueError('batch-size and max-samples must be positive')
    if args.permutations < 2 or not 0 <= args.cas_threshold <= 1:
        raise ValueError('Require permutations >= 2 and cas-threshold in [0, 1]')
    if (args.memory_k < 1 or args.memory_capacity < 1 or args.memory_chars < 100
            or not 0 <= args.memory_confidence <= 1):
        raise ValueError('Invalid memory configuration')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter(args.baseline, args.checkpoint, args.device, args.manifest,
                            args.batch_size, backbone=args.backbone)
    rows = rows_for(adapter.splits, args.datasets, args.max_samples)
    # Validate both partitions before expensive inference.
    rows_for(adapter.splits, args.datasets, args.max_samples, split='calibration')
    print(f'Collecting shared base predictions: {len(rows)} examples', flush=True)
    test = collect_logits(adapter, rows, args.batch_size, progress=args.progress)
    common = {**vars(args), 'max_examples': args.max_samples,
              'score': 'confidence', 'tau': [0.3, 0.5, 0.7], 'alpha': [0.1, 0.2, 0.3]}
    selected_methods = list(dict.fromkeys(args.methods))
    for name in selected_methods:
        module = import_module(f'tta.method_{name}')
        print(f'Running {name}', flush=True)
        module.run({**common, 'output': out_dir / f'{name}.json'}, adapter=adapter, test=test)
    summary = {key: value for key, value in vars(args).items() if key != 'output_dir'}
    summary['methods'] = selected_methods
    save_json(out_dir / 'summary.json', summary)
    # Keep Qwen/MedGemma and Laya-Med checkpoint variants in separate directories.
    report_name = out_dir.name.removesuffix('_tta')
    baseline_dir = out_dir.parent / report_name if report_name != out_dir.name else out_dir
    write_tta_report(out_dir, baseline_dir, args.baseline, args.datasets,
                     args.max_samples, args.checkpoint)
    print(f'Saved {len(selected_methods)} TTA results to {out_dir}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='laya_med',
                       choices=['laya_med', 'laya', 'medjev', 'medgemma_direct_headonly',
                                'medgemma_direct_lora', 'medgemma_ll'])
    parser.add_argument('--methods', nargs='+', choices=AVAILABLE_METHODS, default=list(METHODS),
                        help='Policies to run; unselected result files are preserved')
    parser.add_argument('--checkpoint')
    parser.add_argument('--manifest', default='data/prepared/manifest.json')
    parser.add_argument('--datasets', nargs='+',
                       choices=['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa'],
                       default=['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--output-dir')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--backbone', default=None)
    parser.add_argument('--permutations', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cas-threshold', type=float, default=0.95)
    parser.add_argument('--memory-k', type=int, default=3)
    parser.add_argument('--memory-capacity', type=int, default=256)
    parser.add_argument('--memory-confidence', type=float, default=0.8)
    parser.add_argument('--memory-chars', type=int, default=800)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = f'tta_results/{args.baseline}_tta'
    run_tta(args)
