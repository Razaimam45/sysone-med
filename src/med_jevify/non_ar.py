"""Adapters for non-autoregressive System One decision models."""

from __future__ import annotations

import time
import json
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


class LayaAdapter:
    """Adapt Laya's typed choice API to the trial's probability interface."""

    def __init__(self, model: str = "convaiinnovations/laya", device: str = "auto", fast: bool = False):
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError(
                "Laya is not installed. Install the optional dependency with "
                "`python -m pip install -e '.[non_ar]'`."
            ) from exc
        kwargs: dict[str, Any] = {"fast": fast}
        if device != "auto":
            kwargs["device"] = device
        self.model = model
        self.agent = laya.load(model, **kwargs)

    @staticmethod
    def _question(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": {
                "type": "choice",
                "instructions": example["question"],
                "criteria": {option: option for option in example["options"]},
            }
        }

    @staticmethod
    def _probabilities(result: dict[str, Any], options: Sequence[str]) -> np.ndarray:
        try:
            answer = result["answers"]["decision"]
            values = answer["probabilities"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Laya returned no choice probability distribution for the decision question."
            ) from exc
        if not isinstance(values, dict):
            raise RuntimeError("Laya choice probabilities must be a mapping keyed by option.")
        try:
            probabilities = np.asarray([float(values[option]) for option in options], dtype=float)
        except KeyError as exc:
            raise RuntimeError(f"Laya omitted probability for option {exc.args[0]!r}.") from exc
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise RuntimeError("Laya returned invalid choice probabilities.")
        total = probabilities.sum()
        if total <= 0:
            raise RuntimeError("Laya returned zero total choice probability.")
        return probabilities / total

    def score(self, examples: Sequence[dict[str, Any]], batch_size: int) -> tuple[list[np.ndarray], list[float]]:
        if any(isinstance(example.get("state"), dict) and example["state"].get("image") for example in examples):
            raise ValueError("The Laya non-autoregressive adapter is text-only; VQA-RAD needs a multimodal model.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        probabilities = [None] * len(examples)
        latencies = [0.0] * len(examples)
        # The upstream API accepts one shared question schema per predict_batch.
        # Only group examples with exactly the same question and choices.
        groups = defaultdict(list)
        for index, example in enumerate(examples):
            groups[json.dumps(self._question(example), sort_keys=True)].append(index)
        for schema, indices in groups.items():
            for start in range(0, len(indices), batch_size):
                positions = indices[start:start + batch_size]
                batch = [examples[i] for i in positions]
                begin = time.perf_counter()
                results = self.agent.predict_batch(
                    [example.get("state", "") for example in batch], json.loads(schema),
                    batch_size=len(batch), sort_by_length=len(batch) > 1)
                elapsed = time.perf_counter() - begin
                if len(results) != len(batch):
                    raise RuntimeError("Laya result count does not match submitted examples")
                for index, example, result in zip(positions, batch, results):
                    probabilities[index] = self._probabilities(result, example["options"])
                    latencies[index] = elapsed / len(batch)
        return probabilities, latencies
