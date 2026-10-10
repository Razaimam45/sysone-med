"""Validated, checksummed split manifests shared by trainers and benchmarks."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SPLITS = ('train', 'validation', 'calibration', 'test')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized(value):
    return ' '.join(str(value).casefold().split())


def image_path(row):
    state = row.get('state')
    return state.get('image') if isinstance(state, dict) else None


def validate_row(row):
    for key in ('id', 'dataset', 'question', 'options', 'label'):
        if key not in row:
            raise ValueError(f'Missing {key}')
    options = row['options']
    if not str(row['id']).strip() or not str(row['dataset']).strip():
        raise ValueError('Empty ID or dataset')
    if not isinstance(row['question'], str) or not row['question'].strip():
        raise ValueError('Empty question')
    if not isinstance(options, (list, tuple)) or len(options) < 2:
        raise ValueError('At least two choices required')
    if any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError('Empty or non-string choice')
    if len({normalized(o) for o in options}) != len(options):
        raise ValueError('Duplicate choices')
    if type(row['label']) is not int or not 0 <= row['label'] < len(options):
        raise ValueError('Label outside choices')
    state = row.get('state', '')
    if not isinstance(state, (str, dict)):
        raise ValueError('State must be text or a mapping')
    if isinstance(state, dict) and not isinstance(state.get('text', ''), str):
        raise ValueError('State text must be a string')
    if row['dataset'] == 'medmcqa' and (row.get('input_schema') != 'question_options_only_v2' or
                                      (state.get('text', '') if isinstance(state, dict) else state)):
        raise ValueError('Regenerate MedMCQA: legacy inputs may contain answer explanations')
    if row['dataset'] in ('slake', 'path_vqa') and isinstance(state, dict) and state.get('text'):
        raise ValueError('Visual QA inputs must exclude ground-truth location metadata')
    if row['dataset'] in ('slake', 'path_vqa', 'pmc_vqa', 'vqarad') and not image_path(row):
        raise ValueError('Visual QA example missing image')
    if image_path(row) and not Path(image_path(row)).is_file():
        raise ValueError(f'Missing image: {image_path(row)}')


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pixel_hash(path):
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert('RGB')
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def leakage_keys(row):
    """Conservative grouping; generic visual questions are not duplicate cases."""
    keys = [('id', row['id'])]
    for field in ('patient_id', 'article_id', 'group_id'):
        if row.get(field):
            # Caller-supplied group IDs are scoped to their source dataset.
            keys.append((field, row['dataset'], str(row[field])))
    if row.get('image_sha256'):
        keys.append(('image', row['image_sha256']))
        name = Path(image_path(row)).name
        pmc = re.search(r'PMC\d+', name, re.I)
        if pmc:
            keys.append(('article', pmc.group().upper()))
    else:
        # Text question overlap remains grouped even when contexts/options differ.
        keys.append(('question', normalized(row['question'])))
    return keys


def load_manifest(path, verify_images=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 2:
        raise ValueError('Unsupported split manifest; run training/prepare_data.py')
    if manifest.get('audit_sha256') and file_hash(path.parent / 'audit.json') != manifest['audit_sha256']:
        raise ValueError('Audit fingerprint changed')
    result, ownership, ids, seen_images = {}, {}, set(), {}
    for split in SPLITS:
        spec = manifest['splits'][split]
        source = path.parent / spec['path']
        if file_hash(source) != spec['sha256']:
            raise ValueError(f'{split} data fingerprint changed')
        rows = read_jsonl(source)
        if len(rows) != spec['count'] or [r['id'] for r in rows] != spec['ids']:
            raise ValueError(f'{split} manifest IDs/count disagree')
        for row in rows:
            validate_row(row)
            if row['id'] in ids:
                raise ValueError(f'Duplicate ID: {row["id"]}')
            ids.add(row['id'])
            if image_path(row):
                if not row.get('image_sha256'):
                    raise ValueError('Image row missing fingerprint')
                if verify_images:
                    p = image_path(row)
                    if p not in seen_images:
                        seen_images[p] = pixel_hash(p)
                    if seen_images[p] != row['image_sha256']:
                        raise ValueError(f'Image content changed: {p}')
            for key in leakage_keys(row):
                if key in ownership and ownership[key] != split:
                    raise ValueError(f'Cross-split overlap: {key}')
                ownership[key] = split
        result[split] = rows
    return manifest, result
