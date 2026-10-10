from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.run_experiments import (  # noqa: E402
    candidate_audit,
    conformal_mask,
    metric,
    paired_table,
    probs,
    require_same,
)


def record(row_id: str, label: int, logits: list[float], dataset: str = "task") -> dict:
    return {"id": row_id, "dataset": dataset, "label": label, "logits": logits,
            "temperature": 1.0, "options": ["a", "b"]}


def test_logits_use_temperature_saved_with_prediction():
    row = record("one", 0, [0.0, 2.0])
    assert np.allclose(probs(row), [1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2))])
    assert np.allclose(probs(row, 2.0), [1 / (1 + np.exp(1)), 1 / (1 + np.exp(-1))])


def test_candidate_audit_compares_ordered_candidates_and_labels():
    manifest = [{"id": "task:one", "dataset": "task", "label": 1,
                 "options": ["a", "b"]}]
    prediction = {"id": "task:one", "dataset": "task", "label": 1,
                  "options": ["a", "b"], "probabilities": [0.2, 0.8]}
    audit = candidate_audit([prediction], manifest, "model")
    assert audit[0]["unknown_ids"] == 0
    assert audit[0]["label_mismatches"] == 0
    assert audit[0]["option_mismatches"] == 0
    assert audit[0]["invalid_vectors"] == 0


def test_paired_comparison_rejects_different_example_sets():
    a = [record("one", 0, [2.0, 0.0])]
    b = [record("other", 0, [2.0, 0.0])]
    with pytest.raises(ValueError, match="different IDs"):
        require_same(a, b, "a", "b")


def test_paired_table_reports_bootstrap_interval_and_paired_delta():
    reference = [record("one", 0, [2.0, 0.0]), record("two", 1, [2.0, 0.0])]
    candidate = [record("one", 0, [2.0, 0.0]), record("two", 1, [0.0, 2.0])]
    rows = paired_table(reference, candidate, "ref", "candidate", None, None, 200, 1)
    pooled = next(row for row in rows if row["dataset"] == "__pooled__")
    assert pooled["accuracy_delta"] == 0.5
    assert pooled["accuracy_ci_low"] <= pooled["accuracy_delta"] <= pooled["accuracy_ci_high"]


def test_generated_invalid_answer_counts_wrong_even_if_allowed_mass_argmax_is_correct():
    generated = {"id": "one", "dataset": "task", "label": 0, "prediction": -1,
                 "options": ["a", "b", "__invalid__"],
                 "probabilities": [0.6, 0.3, 0.1], "confidence": 0.1}
    assert metric(generated)["accuracy"] == 0.0
    assert metric(generated)["confidence"] == 0.1


def test_brier_aligns_explicit_invalid_outcome_as_zero_for_direct_models():
    reference = [{"id": "one", "dataset": "task", "label": 0,
                  "options": ["a", "b", "__invalid__"],
                  "probabilities": [0.7, 0.2, 0.1]}]
    candidate = [{"id": "one", "dataset": "task", "label": 0,
                  "options": ["a", "b"], "probabilities": [0.7, 0.3]}]
    row = paired_table(reference, candidate, "ref", "candidate", None, None, 100, 1)[0]
    assert row["probability_dimensions_match"] is False
    assert row["brier_comparable"] is True
    assert np.isfinite(row["brier_delta"])


def test_conformal_full_set_preserves_baseline_predictions():
    probabilities = np.asarray([0.7, 0.2, 0.1])
    mask = conformal_mask(probabilities, float("inf"))
    assert mask.all()
    assert int(probabilities.argmax()) == 0
    assert metric({"id": "one", "dataset": "task", "label": 0,
                   "probabilities": probabilities.tolist()})["accuracy"] == 1.0
