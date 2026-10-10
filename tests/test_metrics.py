import numpy as np

from med_jevify.calibration import TemperatureScaler, fit_temperature
from med_jevify.metrics import brier, ece, nll
from med_jevify.upstream import confidence_from_probs, restrict_and_normalize, score_answer_tokens


def test_restrict_and_normalize_and_confidence():
    probabilities = restrict_and_normalize([2.0, 0.0])
    assert np.isclose(sum(probabilities), 1)
    assert confidence_from_probs([1.0, 0.0]) == 1


def test_metrics_known_values():
    probabilities = [[0.8, 0.2], [0.4, 0.6]]
    labels = [0, 1]
    assert np.isclose(nll(probabilities, labels), (-np.log(0.8) - np.log(0.6)) / 2)
    assert np.isclose(brier(probabilities, labels), 0.2)
    assert 0 <= ece(probabilities, labels) <= 1


def test_temperature_scaling_fit_is_positive_and_normalized():
    scaler = fit_temperature([[4, 0], [0, 4]], [0, 1])
    assert scaler.temperature > 0
    assert np.allclose(TemperatureScaler(2).transform(np.array([[2, 0]])), [[0.7310586, 0.2689414]])


def test_single_token_markers_are_checked():
    class Tokenizer:
        def encode(self, value, add_special_tokens=False):
            return [ord(value)]

    assert score_answer_tokens(Tokenizer(), ["A"]) == {"A": ord("A")}
