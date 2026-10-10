"""Pinned Jevify integration and compatibility helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Sequence

from . import JEVIFY_COMMIT


@dataclass(frozen=True)
class JevifyInfo:
    commit: str = JEVIFY_COMMIT
    package: str = "jevify"


def load_jevify() -> Any:
    """Import the pinned package and fail with an actionable message."""
    try:
        return importlib.import_module("jevify")
    except ImportError as exc:
        raise RuntimeError(
            "Jevify is not installed. Install this project with `pip install -e .` "
            "to install the pinned commit "
            f"{JEVIFY_COMMIT}."
        ) from exc


def verify_jevify(expected_commit: str = JEVIFY_COMMIT) -> JevifyInfo:
    """Verify the requested pin and the installed VCS commit via PEP 610 metadata."""
    if expected_commit != JEVIFY_COMMIT:
        raise ValueError(
            f"Only the repository pin {JEVIFY_COMMIT} is supported by this trial, "
            f"not {expected_commit}."
        )
    load_jevify()
    import json
    from importlib.metadata import distribution

    metadata = distribution("jevify").read_text("direct_url.json")
    actual = json.loads(metadata or "{}").get("vcs_info", {}).get("commit_id")
    if actual != expected_commit:
        raise RuntimeError(f"Cannot verify installed Jevify pin: found {actual!r}; "
                           f"install jevify from git at {expected_commit}.")
    return JevifyInfo()


def restrict_and_normalize(logits: Sequence[float]) -> list[float]:
    """Stable softmax for already-restricted answer logits."""
    import numpy as np

    values = np.asarray(logits, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Answer logits must be a finite, non-empty vector.")
    shifted = values - values.max()
    probs = np.exp(shifted)
    return (probs / probs.sum()).tolist()


def confidence_from_probs(probabilities: Sequence[float]) -> float:
    import numpy as np

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or not probs.size or (probs < 0).any():
        raise ValueError("Probabilities must be a non-empty non-negative vector.")
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability mass must be positive.")
    probs = probs / total
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum())
    return float(1 - entropy / np.log(len(probs))) if len(probs) > 1 else 1.0


def score_answer_tokens(tokenizer: Any, markers: Sequence[str]) -> dict[str, int]:
    """Ensure each marker is a single safe continuation token."""
    result: dict[str, int] = {}
    for marker in markers:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Unsafe answer marker {marker!r}: tokenizer produced {encoded}. "
                "Choose alternative markers before evaluation."
            )
        result[marker] = int(encoded[0])
    return result


def select_safe_markers(tokenizer: Any, count: int) -> tuple[str, ...]:
    """Choose single-token answer markers and make the choice reproducible/loggable."""
    candidates = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + tuple(str(i) for i in range(10))
    selected: list[str] = []
    for marker in candidates:
        try:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
        except (AttributeError, TypeError) as exc:
            raise TypeError("Tokenizer must provide encode(text, add_special_tokens=False).") from exc
        if len(encoded) == 1:
            selected.append(marker)
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(f"Could not find {count} single-token answer markers in this tokenizer.")


def make_choice_question(instructions: str, options: Sequence[str]) -> Any:
    """Construct Jevify's native choice primitive without duplicating its schema."""
    jevify = load_jevify()
    return jevify.Choice(instructions, {option: None for option in options})


def system_one(model: Any, state: Any, instructions: str, options: Sequence[str]) -> dict[str, Any]:
    """Call Jevify's public System One API for one bounded decision."""
    question = make_choice_question(instructions, options)
    return model.system_one(state, {"decision": question}, debug=True)
