"""VQA-RAD loader and image-grouped splitting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def image_group_split(rows: Iterable[dict[str, Any]], calibration_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    cutoff = int(len(keys) * (1 - calibration_fraction))
    test = [row for key in keys[:cutoff] for row in groups[key]]
    calibration = [row for key in keys[cutoff:] for row in groups[key]]
    return calibration, test


def load_vqarad(annotation_file: str, image_dir: str) -> list[dict[str, Any]]:
    path = Path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(
            f"VQA-RAD annotations not found at {path}. Download the official files manually "
            "and pass --vqarad-annotations and --vqarad-images."
        )
    import json

    with path.open() as handle:
        data = json.load(handle)
    rows = data if isinstance(data, list) else data.get("data", data.get("questions", []))
    output = []
    for row in rows:
        answer = str(row.get("answer", "")).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        image_id = str(row.get("image_name") or row.get("image_id") or row.get("image"))
        output.append({
            "id": str(row.get("qid") or row.get("question_id") or len(output)),
            "image_id": image_id,
            "state": {"image": str(Path(image_dir) / image_id)},
            "question": str(row["question"]),
            "options": ("yes", "no"),
            "label": 0 if answer == "yes" else 1,
        })
    return output
