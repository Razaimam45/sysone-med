#!/usr/bin/env python3
"""Prepare immutable, audited train/validation/calibration/test manifests."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from med_jevify.decision_data import (SPLITS, file_hash, image_path, leakage_keys,
                                     normalized, pixel_hash, read_jsonl, validate_row)
from med_jevify.tasks.pubmedqa import load_pubmedqa
from med_jevify.tasks.vqarad import load_vqarad


def prepare(rows, seed=42):
    """Union related cases globally, preserve held-out sources, quarantine conflicts."""
    accepted, rejected, image_cache, seen = [], [], {}, {}
    for original in rows:
        row = dict(original)
        try:
            validate_row(row)
            if image_path(row):
                path = str(Path(image_path(row)).resolve())
                row['state'] = dict(row['state'], image=path)
                if path not in image_cache:
                    image_cache[path] = pixel_hash(path)
                row['image_sha256'] = image_cache[path]
            state = row.get('state', '')
            context = state.get('text', '') if isinstance(state, dict) else state
            key = (row['dataset'], normalized(context), normalized(row['question']),
                   tuple(normalized(o) for o in row['options']), row.get('image_sha256'))
            row['_duplicate_key'] = key
            accepted.append(row)
        except (ValueError, OSError) as exc:
            rejected.append({'id': row.get('id'), 'reason': str(exc)})
    parent = list(range(len(accepted)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(accepted):
        for key in leakage_keys(row):
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(accepted):
        groups[find(i)].append(row)
    pools = defaultdict(list)
    priority = {'train': 0, 'validation': 1, 'calibration': 2, 'test': 3}
    # Keep the highest-priority source partition, never move training cases into a test set.
    for group in groups.values():
        highest = max(priority[r['source_split']] for r in group)
        keep = []
        duplicate_labels = defaultdict(set)
        for r in group:
            duplicate_labels[r['_duplicate_key']].add(r['label'])
        for row in group:
            key = row.pop('_duplicate_key')
            if priority[row['source_split']] != highest:
                rejected.append({'id': row['id'], 'reason': 'group overlaps protected source split'})
            elif len(duplicate_labels[key]) > 1:
                rejected.append({'id': row['id'], 'reason': 'conflicting labels for identical input'})
            elif key in seen or row['id'] in seen:
                rejected.append({'id': row['id'], 'reason': 'duplicate input or ID'})
            else:
                seen[key] = True
                seen[row['id']] = True
                keep.append(row)
        if keep:
            datasets = tuple(sorted({r['dataset'] for r in keep}))
            pools[(datasets, keep[0]['source_split'])].append(keep)
    output = {split: [] for split in SPLITS}
    supplied = defaultdict(set)
    for row in accepted:
        supplied[row['dataset']].add(row['source_split'])
    rng = random.Random(seed)
    for (datasets, source), chunks in sorted(pools.items()):
        rng.shuffle(chunks)
        if source in ('test', 'calibration'):
            allocation = [(source, 1.0)]
        elif source == 'validation':
            allocation = [('validation', .5), ('calibration', .5)]
        else:
            has_valid = all('validation' in supplied[d] for d in datasets)
            has_test = all('test' in supplied[d] for d in datasets)
            allocation = [('train', 1.0)]
            if not has_valid:
                allocation += [('validation', .1), ('calibration', .1)]
            if not has_test:
                allocation += [('test', .1)]
            allocation[0] = ('train', 1 - sum(f for _, f in allocation[1:]))
        # Partition whole groups, targeting row counts rather than question counts per group.
        if len(chunks) < len(allocation):
            if len(datasets) > 1:
                # Rare cross-dataset components stay together; ordinary source pools
                # supply each dataset's holdouts. Never split this component.
                allocation = [(source, 1.0)]
            else:
                raise ValueError(f'Not enough independent groups for {datasets}/{source}')
        total = sum(map(len, chunks))
        cursor = 0
        for index, (split, fraction) in enumerate(allocation):
            if index == len(allocation) - 1:
                selected = chunks[cursor:]
            else:
                selected, count = [], 0
                limit = len(chunks) - (len(allocation) - index - 1)
                while cursor < limit and (not selected or count < total * fraction):
                    selected.append(chunks[cursor]); count += len(chunks[cursor]); cursor += 1
            for group in selected:
                output[split].extend(group)
    counts = {s: dict(Counter(r['dataset'] for r in rs)) for s, rs in output.items()}
    return output, {'counts': counts, 'quarantined': rejected,
                    'quarantine_reasons': dict(Counter(r['reason'] for r in rejected)),
                    'grouping': 'IDs, exact text questions, decoded image SHA256, supplied patient/article/group IDs, PMC filename article IDs',
                    'limitations': 'No guarantee against near-duplicate images or unknown patient/article relationships; no pretraining-contamination audit.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--medical-data-dir', type=Path, default=ROOT / 'data/normalized')
    p.add_argument('--datasets', nargs='+', default=['med_qa', 'medmcqa', 'slake', 'path_vqa', 'pmc_vqa'])
    p.add_argument('--output-dir', type=Path, default=ROOT / 'data/prepared')
    p.add_argument('--extra-jsonl', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--text-model', default=str(ROOT / 'data/models/text'))
    p.add_argument('--max-length', type=int, default=512)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Output directory already exists; choose a new versioned directory')
    rows, inputs = [], {}
    for split in ('train', 'test'):
        for row in load_pubmedqa(split):
            rows.append(dict(row, id='pubmedqa:' + row['id'], dataset='pubmedqa', source_split=split))
    for source in (ROOT / 'data/sources/pubmedqa/ori_pqal.json', ROOT / 'data/sources/pubmedqa/test_ground_truth.json', ROOT / 'data/sources/vqarad/annotations.json'):
        inputs[str(source)] = file_hash(source)
    for row in load_vqarad(str(ROOT / 'data/sources/vqarad/annotations.json'), str(ROOT / 'data/sources/vqarad/images')):
        rows.append(dict(row, id='vqarad:' + row['id'], dataset='vqarad', source_split='train'))
    for name in args.datasets:
        directory = args.medical_data_dir / name
        if not (directory / 'train.jsonl').is_file():
            raise FileNotFoundError(f'Requested dataset missing: {directory}')
        for split in SPLITS:
            source = directory / f'{split}.jsonl'
            if source.is_file():
                inputs[str(source.resolve())] = file_hash(source)
                records = read_jsonl(source)
                for row in records:
                    if name == 'medmcqa' and row.get('input_schema') != 'question_options_only_v2':
                        raise ValueError('Regenerate MedMCQA using the updated normalizer before preparing splits')
                    rows.append(dict(row, dataset=name, source_split=split))
    if args.extra_jsonl:
        inputs[str(args.extra_jsonl.resolve())] = file_hash(args.extra_jsonl)
        rows.extend(dict(row, source_split='train') for row in read_jsonl(args.extra_jsonl))
    from transformers import AutoTokenizer
    from laya_med.data import MedicalDecisionDataset, protected_texts
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    if not 8 <= args.max_length <= tokenizer.model_max_length:
        raise ValueError('Invalid max_length for tokenizer')
    encoding = {'tokenizer_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                'text_model': args.text_model, 'max_length': args.max_length}
    text_rejected, usable = [], []
    print(f'Checking protected question/option lengths for {len(rows)} rows...', flush=True)
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        valid = []
        for row in chunk:
            try:
                validate_row(row)
                valid.append(row)
            except ValueError as exc:
                text_rejected.append({'id': row.get('id'), 'reason': str(exc)})
        examples = MedicalDecisionDataset(valid).rows
        strings = [text for row in examples for text in protected_texts(row)]
        if not strings:
            continue
        lengths = tokenizer(strings, add_special_tokens=False, return_length=True)['length']
        cursor = 0
        for original, row in zip(valid, examples):
            maximum = max(lengths[cursor:cursor + len(row.options)])
            cursor += len(row.options)
            if maximum + tokenizer.num_special_tokens_to_add(pair=True) > args.max_length:
                text_rejected.append({'id': row.id, 'reason': 'question/options exceed token budget'})
            else:
                usable.append(original)
    rows = usable
    print(f'Auditing {len(rows)} rows and image contents...', flush=True)
    output, audit = prepare(rows, args.seed)
    audit['quarantined'].extend(text_rejected)
    audit['quarantine_reasons'] = dict(Counter(r['reason'] for r in audit['quarantined']))
    expected = {'pubmedqa', 'vqarad', *args.datasets} | {r['dataset'] for r in rows}
    for split, records in output.items():
        missing = expected - {r['dataset'] for r in records}
        if missing:
            raise ValueError(f'{split} lacks datasets after audit: {sorted(missing)}')
    audit['input_sanitization'] = {'medmcqa': 'Excluded explanation, subject and topic annotations',
                                  'slake': 'Excluded ground-truth organ/location annotations'}
    args.output_dir.mkdir(parents=True)
    manifest = {'schema_version': 2, 'seed': args.seed, 'encoding': encoding, 'sources': inputs, 'splits': {}}
    for split, records in output.items():
        records.sort(key=lambda r: (r['dataset'], r['id']))
        target = args.output_dir / f'{split}.jsonl'
        target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        manifest['splits'][split] = {'path': target.name, 'count': len(records),
                                    'ids': [r['id'] for r in records], 'sha256': file_hash(target)}
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2))
    manifest['audit_sha256'] = file_hash(args.output_dir / 'audit.json')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'counts': audit['counts'], 'quarantine_reasons': audit['quarantine_reasons']}, indent=2))


if __name__ == '__main__':
    main()
