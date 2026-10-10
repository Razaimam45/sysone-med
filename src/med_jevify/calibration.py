"""Post-hoc scalar temperature calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float = 1.0

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        values = np.asarray(logits, dtype=float)
        shifted = values / self.temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> TemperatureScaler:
    values = np.asarray(logits, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if values.ndim != 2 or len(values) != len(targets) or len(values) == 0:
        raise ValueError("Logits must be a non-empty 2-D array matching labels.")
    if (targets < 0).any() or (targets >= values.shape[1]).any():
        raise ValueError("Labels must be valid class indices.")
    # Same grid and objective as the pinned upstream train/eval.py (one-hot targets).
    grid = np.exp(np.linspace(np.log(0.3), np.log(30.0), 120))
    losses = []
    for temperature in grid:
        probabilities = TemperatureScaler(float(temperature)).transform(values)
        losses.append(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-9, 1)).mean())
    return TemperatureScaler(float(grid[int(np.argmin(losses))]))

