"""Tests for the hosted JEV API adapter (mocked HTTP)."""

from types import SimpleNamespace

import numpy as np
import pytest

from med_jevify.jev_api import JevApiAdapter, choice_question, is_multimodal, state_text


def _adapter(monkeypatch, responses):
    """Build an adapter whose HTTP calls return the canned responses in order."""
    import requests

    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: responses.pop(0),
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    return JevApiAdapter(max_workers=1), calls


def test_state_text_handles_string_and_mapping():
    assert state_text({"state": "abstract text"}) == "abstract text"
    assert state_text({"state": {"text": "step1", "image": None}}) == "step1"
    assert state_text({"state": ""}) == ""


def test_is_multimodal_rejects_image_rows():
    assert is_multimodal({"state": {"text": "x", "image": "img.png"}}) is True
    assert is_multimodal({"state": {"text": "x", "image": None}}) is False
    assert is_multimodal({"state": "plain"}) is False


def test_choice_question_uses_row_options():
    question = choice_question({"question": "Q?", "options": ["yes", "no"]})
    assert question["decision"]["type"] == "choice"
    assert question["decision"]["instructions"] == "Q?"
    assert set(question["decision"]["criteria"]) == {"yes", "no"}


def test_score_maps_probabilities_to_option_order(monkeypatch):
    responses = [
        {"answers": {"decision": {"probabilities": {"no": 0.2, "yes": 0.8}}}},
    ]
    adapter, calls = _adapter(monkeypatch, responses)
    probs, latencies = adapter.score(
        [{"state": "ctx", "question": "Q?", "options": ["yes", "no"]}], 1)
    np.testing.assert_allclose(probs[0], [0.8, 0.2])
    assert latencies[0] > 0
    assert calls[0]["model"] == "typesafe/jev-1.13"
    assert calls[0]["state"] == "ctx"


def test_score_rejects_multimodal_rows(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    adapter = JevApiAdapter(max_workers=1)
    with pytest.raises(ValueError, match="text-only"):
        adapter.score([{"state": {"text": "x", "image": "img.png"},
                        "question": "Q?", "options": ["yes", "no"]}], 1)


def test_score_concurrent_requests_preserve_order(monkeypatch):
    import requests
    import time

    def fake_post(url, headers, json, timeout):
        time.sleep(0.01 * (ord(json["state"][0]) % 5))
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"answers": {"decision": {"probabilities": {"a": 1.0, "b": 0.0}}}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    adapter = JevApiAdapter(max_workers=4)
    examples = [{"state": f"state{i}", "question": "Q?", "options": ["a", "b"]}
                for i in range(6)]
    probs, latencies = adapter.score(examples, 6)
    assert [float(p[0]) for p in probs] == [1.0] * 6
    assert len(latencies) == 6


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        JevApiAdapter()


def test_rate_limit_raises(monkeypatch):
    import requests

    def fake_post(url, headers, json, timeout):
        return SimpleNamespace(status_code=429, raise_for_status=lambda: None, json=lambda: {})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    adapter = JevApiAdapter(max_workers=1)
    with pytest.raises(RuntimeError, match="429"):
        adapter.score([{"state": "ctx", "question": "Q?", "options": ["yes", "no"]}], 1)