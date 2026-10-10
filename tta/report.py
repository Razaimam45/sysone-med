#!/usr/bin/env python3
"""Consolidated TTA report generation.

Writes a summary report for each baseline's TTA results into the main
results/ directory, so the paper can read per-baseline TTA performance
alongside the original baseline metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _load_json(path: Path):
    """Load a JSON file, returning None if missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_tta_report(tta_dir: Path, baseline_dir: Path, baseline: str,
                     datasets: Iterable[str], max_samples: int,
                     checkpoint: str | None = None) -> Path:
    """Write a consolidated TTA report for a baseline.

    The report is written to ``baseline_dir / tta_summary.json`` so it sits
    alongside the baseline's main results.
    """
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    methods = {}
    for name in ('routing', 'conformal', 'permutation', 'memory', 'selective', 'calibrated'):
        data = _load_json(tta_dir / f'{name}.json')
        if data is None:
            methods[name] = None
        elif name in ('routing', 'conformal'):
            methods[name] = {
                'method': data['method'], 'test_n': data['test_n'],
                'results': [{k: v for k, v in entry.items() if k != 'rows'}
                            for entry in data['results']]}
        else:
            methods[name] = {k: v for k, v in data.items() if k != 'rows'}
            details = data.get('rows', [])
            if all('additional_scored_examples' in row for row in details):
                methods[name]['additional_scored_examples'] = sum(
                    row['additional_scored_examples'] for row in details)
            if name == 'selective':
                methods[name]['selected_n'] = sum(row['selected'] for row in details)
    report = {
        'baseline': baseline, 'checkpoint': checkpoint,
        'datasets': list(datasets), 'max_samples': max_samples, 'methods': methods,
    }

    out_path = baseline_dir / "tta_summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path