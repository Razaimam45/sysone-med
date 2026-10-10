"""OpenRouter JEV API adapter: a remote non-autoregressive decision baseline.

JEV (``typesafe/jev-1.13``) is a hosted decision model exposed through the
OpenRouter ``/api/alpha/decisions`` endpoint. It is text-only, so this adapter
only supports text decision rows (state is a string or a text-only mapping).
Multimodal rows are rejected by the caller.

The adapter implements the same ``score(examples, batch_size) -> (probabilities,
latencies)`` interface as :class:`med_jevify.non_ar.LayaAdapter`, so it can be
plugged into :func:`med_jevify.evaluate.run_probability_backend` unchanged.
The endpoint accepts one ``state`` per request, so ``batch_size`` is advisory:
examples are issued concurrently up to ``max_workers`` and each example's
latency is its own HTTP request duration.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

import numpy as np

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"


def state_text(example: dict[str, Any]) -> str:
    """Extract the text context from a prepared decision row."""
    state = example.get("state", "")
    if isinstance(state, dict):
        return str(state.get("text", "") or "")
    return str(state or "")


def is_multimodal(example: dict[str, Any]) -> bool:
    state = example.get("state")
    return isinstance(state, dict) and bool(state.get("image"))


def choice_question(example: dict[str, Any]) -> dict[str, Any]:
    """Build JEV's typed ``choice`` question from a prepared row."""
    return {
        "decision": {
            "type": "choice",
            "instructions": example["question"],
            "criteria": {option: option for option in example["options"]},
        }
    }


def normalize_choice_probabilities(result: dict[str, Any], options: Sequence[str]) -> np.ndarray:
    """Map JEV's option-keyed probability dict onto the row's option order."""
    answer = result["answers"]["decision"]
    values = answer["probabilities"]
    if not isinstance(values, dict):
        raise RuntimeError("JEV returned no choice probability distribution.")
    try:
        probabilities = np.asarray([float(values[option]) for option in options], dtype=float)
    except KeyError as exc:
        raise RuntimeError(f"JEV omitted probability for option {exc.args[0]!r}.") from exc
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("JEV returned invalid choice probabilities.")
    total = float(probabilities.sum())
    if total <= 0:
        raise RuntimeError("JEV returned zero total choice probability.")
    return probabilities / total


class JevApiAdapter:
    """Score text decision rows through the hosted JEV decision endpoint."""

    def __init__(self, model: str = JEV_MODEL, api_key: str | None = None,
                 timeout: float = 60.0, max_workers: int = 8):
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment check
            raise RuntimeError(
                "The `requests` package is required for the JEV API adapter."
            ) from exc
        self.model = model
        self.timeout = timeout
        self.max_workers = max(1, int(max_workers))
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before using the JEV API adapter."
            )

    def _score_one(self, example: dict[str, Any]) -> tuple[np.ndarray, float]:
        import requests

        begin = time.perf_counter()
        response = requests.post(
            JEV_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "state": state_text(example),
                "questions": choice_question(example),
            },
            timeout=self.timeout,
        )
        if response.status_code == 429:
            raise RuntimeError("JEV API rate limited (HTTP 429); retry later.")
        response.raise_for_status()
        probabilities = normalize_choice_probabilities(response.json(), example["options"])
        return probabilities, time.perf_counter() - begin

    def score(self, examples: Sequence[dict[str, Any]], batch_size: int
              ) -> tuple[list[np.ndarray], list[float]]:
        """Score text rows. ``batch_size`` is advisory; the API is one request per example."""
        if any(is_multimodal(example) for example in examples):
            raise ValueError("The JEV API adapter is text-only; multimodal rows are not supported.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        probabilities: list[np.ndarray | None] = [None] * len(examples)
        latencies = [0.0] * len(examples)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._score_one, example): index
                       for index, example in enumerate(examples)}
            for future in as_completed(futures):
                index = futures[future]
                probabilities[index], latencies[index] = future.result()
        return probabilities, latencies  # type: ignore[arg-type]