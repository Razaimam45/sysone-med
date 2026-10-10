"""Tests for the local MedJev adapter (checkpoint loading is mocked)."""

import numpy as np
import pytest

from med_jevify.medjev_adapter import MedJevAdapter, free_device, is_multimodal, medjev_request, state_text


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_free_device_returns_explicit_cuda_unchanged():
    assert free_device("cuda:2") == "cuda:2"


def test_free_device_returns_explicit_non_cuda_unchanged():
    assert free_device("cpu") == "cpu"


def test_state_text_handles_string_and_mapping():
    assert state_text({"state": "abstract text"}) == "abstract text"
    assert state_text({"state": {"text": "step1", "image": None}}) == "step1"
    assert state_text({"state": ""}) == ""


def test_is_multimodal_rejects_image_rows():
    assert is_multimodal({"state": {"text": "x", "image": "img.png"}}) is True
    assert is_multimodal({"state": {"text": "x", "image": None}}) is False
    assert is_multimodal({"state": "plain"}) is False


def test_medjev_request_is_a_single_choice_question():
    request = medjev_request({"question": "Q?", "options": ["yes", "no", "maybe"]})
    assert request["state"] == ""
    q = request["questions"]["decision"]
    assert q["type"] == "choice"
    assert q["instructions"] == "Q?"
    assert list(q["criteria"]) == ["yes", "no", "maybe"]
    assert q["label"] == "yes"          # dummy label, discarded by materialize


def test_adapter_score_returns_option_ordered_probabilities(monkeypatch):
    """Materialize is mocked so no real checkpoint is needed for the unit test."""
    import med_jevify.medjev_adapter as adapter_module

    class FakeModel:
        def encode(self, tok, rec, **kw):
            return {"ids": [0], "seg": [0], "pos": [0], "opt": [-2], "decide_idx": [0],
                    "opt_idx": [[0]], "labels": [0], "state_truncated": False}

        def probs_and_prefix(self, enc):
            return [np.array([0.2, 0.7, 0.1])], (1, None, None)

    monkeypatch.setattr(adapter_module, "MedJevAdapter",
                        lambda **kw: MedJevAdapter.__new__(MedJevAdapter))
    adapter = MedJevAdapter.__new__(MedJevAdapter)
    adapter._torch = type("T", (), {
        "cuda": type("C", (), {"synchronize": staticmethod(lambda d: None)})(),
        "no_grad": staticmethod(lambda: _NullCtx()),
    })()
    adapter.model = FakeModel()
    adapter.tok = object()
    adapter.max_state = 2048
    adapter.max_branch = 2560
    adapter.device = "cpu"
    adapter._last_latency = 0.0

    example = {"state": "ctx", "question": "Q?", "options": ["yes", "no", "maybe"]}
    probs, latencies = adapter.score([example], 1)
    np.testing.assert_allclose(probs[0], [0.2, 0.7, 0.1])
    assert latencies[0] > 0


def test_adapter_rejects_multimodal_rows(monkeypatch):
    import med_jevify.medjev_adapter as adapter_module
    monkeypatch.setattr(adapter_module, "MedJevAdapter",
                        lambda **kw: MedJevAdapter.__new__(MedJevAdapter))
    adapter = MedJevAdapter.__new__(MedJevAdapter)
    adapter._torch = type("T", (), {"cuda": type("C", (), {"synchronize": staticmethod(lambda d: None)})()})
    adapter.model = type("M", (), {"encode": lambda self, *a, **k: None,
                                   "probs_and_prefix": lambda self, e: ([np.array([1.0])], None)})()
    adapter.tok = object()
    adapter.max_state = 2048
    adapter.max_branch = 2560
    adapter.device = "cpu"
    adapter._last_latency = 0.0
    with pytest.raises(ValueError, match="text-only"):
        adapter.score([{"state": {"text": "x", "image": "img.png"},
                        "question": "Q?", "options": ["yes", "no"]}], 1)