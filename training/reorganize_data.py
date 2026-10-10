#!/usr/bin/env python3
"""One-time migration of the legacy data layout; preserves source contents and split IDs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from med_jevify.decision_data import file_hash

MOVES = {
    'data/pubmedqa': 'data/sources/pubmedqa',
    'data/vqarad': 'data/sources/vqarad',
    'data/medical_datasets': 'data/normalized',
    'data/downloads/raw': 'data/sources/huggingface',
    'data/downloads/hub': 'data/cache/hub',
    'data/downloads/datasets': 'data/cache/datasets',
    'data/downloads/manifest.json': 'data/manifests/downloads.json',
    'data/downloads/text-model': 'data/models/text',
    'data/downloads/vision-model': 'data/models/vision',
}


def remap(value):
    if isinstance(value, dict):
        return {remap(k): remap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v) for v in value]
    if isinstance(value, str):
        for before, after in MOVES.items():
            for prefix, replacement in ((str(ROOT / before), str(ROOT / after)), (before, after)):
                if value == prefix or value.startswith(prefix + '/'):
                    return replacement + value[len(prefix):]
    return value


def main():
    active = [(ROOT / before, ROOT / after) for before, after in MOVES.items()
              if (ROOT / before).exists() or (ROOT / before).is_symlink()]
    if not active:
        print('Data already uses the organized layout'); return
    for before, after in active:
        if after.exists() or after.is_symlink():
            raise FileExistsError(f'Migration destination already exists: {after}')
    links = {before: before.resolve() for before, _ in active if before.is_symlink()}
    for before, after in active:
        after.parent.mkdir(parents=True, exist_ok=True)
        if before in links:
            target = Path(remap(str(links[before])))
            after.symlink_to(os.path.relpath(target, after.parent), target_is_directory=True)
            before.unlink()
        else:
            before.rename(after)
    old = ROOT / 'data/downloads'
    if old.exists() and not any(old.iterdir()):
        old.rmdir()
    for directory in (ROOT / 'data/normalized', ROOT / 'data/prepared'):
        for path in directory.glob('**/*.jsonl'):
            temporary = path.with_suffix('.tmp')
            with path.open() as source, temporary.open('w') as target:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    # Only paths and metadata change; medical question/answer text remains verbatim.
                    if isinstance(row.get('state'), dict) and row['state'].get('image'):
                        row['state']['image'] = remap(row['state']['image'])
                    target.write(json.dumps(row, ensure_ascii=False) + '\n')
            temporary.replace(path)
    path = ROOT / 'data/manifests/downloads.json'
    path.write_text(json.dumps(remap(json.loads(path.read_text())), indent=2))
    for path in (ROOT / 'data/prepared').glob('**/manifest.json'):
        manifest = remap(json.loads(path.read_text()))
        manifest['sources'] = {source: file_hash(source) for source in manifest['sources']}
        for spec in manifest['splits'].values():
            spec['sha256'] = file_hash(path.parent / spec['path'])
        path.write_text(json.dumps(manifest, indent=2))
    (ROOT / 'data/manifests/layout-migration.json').write_text(json.dumps({
        'timestamp': datetime.now(timezone.utc).isoformat(), 'moves': MOVES,
        'note': 'Source contents and prepared split IDs retained; image paths and corresponding fingerprints updated.'}, indent=2))
    print('Moved source data, caches, normalized data and model aliases; updated JSONL paths and manifest fingerprints.')


if __name__ == '__main__':
    main()
