"""MedJev local checkpoint adapter: a trained non-autoregressive decision baseline.

MedJev (JunMa11/MedJev, Apache-2.0) is a fine-tuned Qwen3.5-0.8B with a LoRA
adapter and a pointer head that scores each option against a `<decide>` position.
It is text-only, so this adapter only supports text decision rows (state is a
string or a text-only mapping). Multimodal rows are rejected by the caller.

The adapter implements the same ``score(examples, batch_size) -> (probabilities,
latencies)`` interface as :class:`med_jevify.non_ar.LayaAdapter`, so it can be
plugged into :func:`med_jevify.evaluate.run_probability_backend` unchanged.

Request construction
--------------------
Each prepared row becomes one MedJev record with a single ``choice`` question.
MedJev's ``materialize`` requires every question to carry a ``label``, so a dummy
label is supplied and discarded: the returned ``meta`` records the option keys in
the order they were rendered, and since ``option_text(name, "") == name`` the
probability index lines up with the row's option order exactly.

State text comes from ``state_text`` (string rows, or the ``text`` field of a
text-only mapping). MedJev truncates the state from the head at ``MAX_STATE``
tokens; long PubMedQA abstracts and MedQA vignettes are handled transparently.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

MEDJEV_RUN = "junma/MedJev-Qwen3.5-0.8B"
MEDJEV_DTYPE = "bfloat16"

_DTYPE_MAP = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def _resolve_dtype(dtype: str):
    """Map a CLI dtype string to a torch dtype, accepting both the short and full names."""
    import torch
    name = _DTYPE_MAP.get(dtype, dtype)
    return getattr(torch, name)


def free_device(device: str = "auto") -> str:
    """Pick a CUDA device with the most free memory, or return ``device`` unchanged.

    ``device`` is ``"auto"`` (default) or an explicit ``"cuda:N"``. Useful on shared
    machines where a default GPU may already be occupied by another training job.
    """
    import torch

    if not device.startswith("cuda") and device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    if device.startswith("cuda"):
        return device
    best, best_free = 0, -1
    for index in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(index)[0]
        if free > best_free:
            best, best_free = index, free
    return f"cuda:{best}"


def state_text(example: dict[str, Any]) -> str:
    """Extract the text context from a prepared decision row."""
    state = example.get("state", "")
    if isinstance(state, dict):
        return str(state.get("text", "") or "")
    return str(state or "")


def is_multimodal(example: dict[str, Any]) -> bool:
    state = example.get("state")
    return isinstance(state, dict) and bool(state.get("image"))


def medjev_request(example: dict[str, Any]) -> dict[str, Any]:
    """Build a MedJev request for one prepared row (dummy label, discarded)."""
    return {
        "state": state_text(example),
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": example["question"],
                "criteria": {option: "" for option in example["options"]},
                "label": example["options"][0],
                "src": "medjevify",
            }
        },
    }


class MedJevAdapter:
    """Score text decision rows through a local MedJev checkpoint."""

    def __init__(self, run: str = MEDJEV_RUN, device: str = "cuda:0",
                 dtype: str = MEDJEV_DTYPE, merge: bool = True):
        import torch
        from medjev.checkpoint import load
        from medjev.model import MAX_STATE, MAX_BRANCH

        self._torch = torch
        self.run = run
        self.device = device
        self.tok, self.model = load(run, device, dtype=_resolve_dtype(dtype))
        self.model.lm.config.use_cache = True
        self.max_state = MAX_STATE
        self.max_branch = MAX_BRANCH

    def _score_one(self, example: dict[str, Any]) -> np.ndarray:
        from medjev.records import materialize

        begin = time.perf_counter()
        rec = materialize(medjev_request(example))
        enc = self.model.encode(self.tok, rec, max_state=self.max_state,
                                max_branch=self.max_branch)
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize(self.device)
        with self._torch.no_grad():
            probs, _ = self.model.probs_and_prefix(enc)
        if str(self.device).startswith("cuda"):
            self._torch.cuda.synchronize(self.device)
        self._last_latency = time.perf_counter() - begin
        return np.asarray(probs[0], dtype=float)

    def score(self, examples: Sequence[dict[str, Any]], batch_size: int
              ) -> tuple[list[np.ndarray], list[float]]:
        """Score text rows one at a time (the pointer head is per-record)."""
        if any(is_multimodal(example) for example in examples):
            raise ValueError("The MedJev adapter is text-only; multimodal rows are not supported.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        probabilities, latencies = [], []
        for example in examples:
            probabilities.append(self._score_one(example))
            latencies.append(self._last_latency)
        return probabilities, latencies