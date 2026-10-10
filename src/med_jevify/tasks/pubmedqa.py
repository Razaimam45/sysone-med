"""PubMedQA labelled-subset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OPTIONS = ("yes", "no", "maybe")


@dataclass(frozen=True)
class PubMedExample:
    id: str
    abstract: str
    question: str
    answer: str


def format_example(example: PubMedExample) -> dict[str, Any]:
    if example.answer not in OPTIONS:
        raise ValueError(f"Unexpected PubMedQA answer {example.answer!r}.")
    return {
        "id": example.id,
        "state": example.abstract,
        "question": example.question,
        "options": OPTIONS,
        "label": OPTIONS.index(example.answer),
    }


def load_pubmedqa(split: str = "train", cache_dir: str | None = None) -> list[dict[str, Any]]:
    """Load official labeled data using the published held-out test IDs.

    ``cache_dir`` may override the local dataset directory.
    """
    import json
    from pathlib import Path

    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported PubMedQA split: {split!r}")
    directory = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[3] / "data/sources/pubmedqa"
    with (directory / "ori_pqal.json").open() as handle:
        data = json.load(handle)
    with (directory / "test_ground_truth.json").open() as handle:
        test_labels = json.load(handle)
    result = []
    for pubid, row in data.items():
        if (pubid in test_labels) != (split == "test"):
            continue
        result.append(format_example(PubMedExample(
            pubid, " ".join(row["CONTEXTS"]), row["QUESTION"], row["final_decision"].lower()
        )))
    return result
