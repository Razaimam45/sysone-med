"""Tests for the local MedGemma-Direct adapter."""

import numpy as np
import pytest

from med_jevify.medgemma_direct import MedGemmaDirectAdapter, state_text


def test_state_text_handles_string_and_mapping():
    assert state_text({"state": "abstract text"}) == "abstract text"
    assert state_text({"state": {"text": "step1", "image": None}}) == "step1"
    assert state_text({"state": ""}) == ""


def test_adapter_score_interface():
    """The adapter exposes score(examples, batch_size) -> (probabilities, latencies)."""
    adapter = MedGemmaDirectAdapter.__new__(MedGemmaDirectAdapter)
    adapter._score_one = lambda example: np.array([0.7, 0.3])
    adapter._last_latency = 0.0
    probs, latencies = adapter.score(
        [{"id": "x", "question": "Q?", "options": ["yes", "no"]}], 1)
    assert len(probs) == 1 and len(latencies) == 1
    assert np.isclose(probs[0].sum(), 1.0)


def test_adapter_rejects_invalid_batch_size():
    adapter = MedGemmaDirectAdapter.__new__(MedGemmaDirectAdapter)
    with pytest.raises(ValueError):
        adapter.score([{"id": "x", "question": "Q?", "options": ["yes", "no"]}], 0)