"""Classification and timing metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _inputs(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if probs.ndim != 2 or len(probs) != len(y) or len(probs) == 0:
        raise ValueError("Probabilities must be a non-empty 2-D array matching labels.")
    probs = np.clip(probs, 1e-9, 1)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, y


def accuracy(probabilities, labels) -> float:
    probs, y = _inputs(probabilities, labels)
    return float((probs.argmax(axis=1) == y).mean())


def mean_confidence(probabilities) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1)
    probs /= probs.sum(axis=1, keepdims=True)
    return float(probs.max(axis=1).mean())


def ece(probabilities, labels, bins: int = 15) -> float:
    probs, y = _inputs(probabilities, labels)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def classwise_ece(probabilities, labels, bins: int = 15) -> float:
    probs, y = _inputs(probabilities, labels)
    edges = np.linspace(0, 1, bins + 1)
    per_class = []
    for cls in range(probs.shape[1]):
        class_value = 0.0
        for index in range(bins):
            mask = (probs[:, cls] > edges[index]) & (probs[:, cls] <= edges[index + 1])
            if mask.any():
                class_value += mask.mean() * abs(probs[mask, cls].mean() - (y[mask] == cls).mean())
        per_class.append(class_value)
    return float(np.mean(per_class))


def nll(probabilities, labels) -> float:
    probs, y = _inputs(probabilities, labels)
    return float(-np.log(probs[np.arange(len(y)), y]).mean())


def brier(probabilities, labels) -> float:
    probs, y = _inputs(probabilities, labels)
    one_hot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def latency_summary(seconds: Sequence[float]) -> tuple[float, float, float]:
    values = np.asarray(seconds, dtype=float)
    if not len(values):
        return (float("nan"), float("nan"), 0.0)
    return (float(np.percentile(values, 50) * 1000), float(np.percentile(values, 95) * 1000), float(len(values) / values.sum()))


def jsd(probabilities, labels) -> float:
    """Jensen-Shannon divergence to one-hot targets, as in upstream eval.py."""
    probs, y = _inputs(probabilities, labels)
    target = np.eye(probs.shape[1])[y]
    midpoint = 0.5 * (probs + target)
    return float(np.mean(0.5 * np.sum(target * np.log(np.clip(target, 1e-9, 1) / midpoint), axis=1)
                         + 0.5 * np.sum(probs * np.log(probs / midpoint), axis=1)))
