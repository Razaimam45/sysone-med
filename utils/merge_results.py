#!/usr/bin/env python3
"""Merge per-baseline results/<baseline>/results.json into a single ordered list.

Each baseline keeps its own results.json with the same column schema; this helper
walks the directory and concatenates the rows, so paper-asset and table scripts
can read one merged list instead of a single monolithic file.
"""

from __future__ import annotations

import json
from pathlib import Path


def merge_results(results_dir: Path | str = "results") -> list[dict]:
    """Return all result rows from every results/<baseline>/results.json file."""
    results_dir = Path(results_dir)
    rows: list[dict] = []
    for sub in sorted(results_dir.iterdir()):
        path = sub / "results.json"
        if path.is_file():
            rows.extend(json.loads(path.read_text()))
    return rows


def latest_rows(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Keep the most recent row per (model, task, mode)."""
    latest: dict[tuple[str, str, str], dict] = {}
    for row in sorted(rows, key=lambda item: item.get("timestamp") or ""):
        latest[(row["model"], row["task"], row["mode"])] = row
    return latest


def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]