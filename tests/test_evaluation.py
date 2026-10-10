import math
from types import SimpleNamespace

import numpy as np
import pytest

from med_jevify.calibration import TemperatureScaler, fit_temperature
from med_jevify.evaluate import (conventional_generate, label_logits, label_logits_batch,
                                 native_generate, summarize)
from med_jevify.non_ar import LayaAdapter


def test_laya_adapter_preserves_choice_probability_order():
    result = {"answers": {"decision": {"probabilities": {"no": 0.2, "yes": 0.8}}}}
    probabilities = LayaAdapter._probabilities(result, ("yes", "no"))
    np.testing.assert_allclose(probabilities, [0.8, 0.2])


def test_label_logits_use_upstream_matching_without_api_rounding():
    jevify = pytest.importorskip('jevify')
    from jevify.prompting import render, SYSTEM_PROMPT
    from jevify.scoring import match_label_probs

    table = {'A': math.log(0.6), ' a': math.log(0.1), 'B': math.log(0.00001234)}
    seen = []

    def next_token(messages):
        seen.append(messages)
        return SimpleNamespace(logprobs=table)

    jev = SimpleNamespace(backend=SimpleNamespace(next_token=next_token), system_prompt=SYSTEM_PROMPT)
    example = {'state': 'context', 'question': 'Question?', 'options': ('yes', 'no', 'maybe')}
    logits, mass = label_logits(jev, example)
    expected = render('context', jevify.Choice('Question?', {x: None for x in example['options']}))
    assert seen == [expected.messages]
    raw, expected_mass = match_label_probs(table, expected.labels)
    np.testing.assert_allclose(logits, [math.log(raw[label] + 1e-9) for label in expected.labels])
    assert mass == expected_mass
    assert np.exp(logits[1]) > 0  # would disappear in the four-decimal API response
    assert np.isclose(np.exp(logits[2]), 1e-9)  # upstream floor for absent top-k label


def test_label_logits_batch_uses_backend_batch_api(monkeypatch):
    from med_jevify import evaluate

    rendered = [SimpleNamespace(messages=[{"role": "user", "content": "q"}],
                                labels=['A', 'B'], keys=['yes', 'no'])] * 2
    monkeypatch.setattr(evaluate, 'render_example', lambda jev, example: rendered[example['index']])
    calls = []

    class Backend:
        def next_token_batch(self, messages):
            calls.append(messages)
            return [SimpleNamespace(logprobs={'A': np.log(.7), 'B': np.log(.3)}) for _ in messages]

    result = label_logits_batch(SimpleNamespace(backend=Backend(), system_prompt='prompt'),
                                [{'index': 0, 'options': ('yes', 'no')},
                                 {'index': 1, 'options': ('yes', 'no')}])
    assert len(calls) == 1
    assert len(calls[0]) == 2
    np.testing.assert_allclose(np.exp(result[0][0]), [.7, .3])


def test_temperature_changes_confidence_not_predictions():
    logits = np.array([[8., 0.], [0., 8.], [8., 0.], [0., 8.]])
    labels = [0, 1, 1, 0]
    scaler = fit_temperature(logits, labels)
    raw = TemperatureScaler().transform(logits)
    calibrated = scaler.transform(logits)
    np.testing.assert_array_equal(raw.argmax(1), calibrated.argmax(1))
    assert not np.allclose(raw, calibrated)
    before = summarize('model', 'task', 'raw_jevify', raw, labels, [0.1]*4)
    after = summarize('model', 'task', 'calibrated_jevify', calibrated, labels, [0.1]*4)
    assert before.accuracy == after.accuracy
    assert after.nll < before.nll
    assert after.mean_confidence < before.mean_confidence


@pytest.mark.parametrize('token, expected', [('A', 0), (' b', 1), ('unrelated', -1)])
def test_native_uses_generate_not_jevify_probabilities(token, expected):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('jevify')
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(sequences=torch.tensor([[7, 8, 2]]), logits=(torch.log(torch.tensor([[0.6, 0.3, 0.1]])),))

    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, bos_token_id=2,
                                decode=lambda ids: token, batch_decode=lambda ids, **kw: ["A", " b", "unrelated"])
    backend = SimpleNamespace(lm=SimpleNamespace(generate=generate), device='cpu',
                              tokenizer=tokenizer, encode=lambda messages: [7, 8])
    jev = SimpleNamespace(backend=backend, system_prompt='classifier')
    result = native_generate(jev, {'state': 'context', 'question': '?', 'options': ('yes', 'no')})
    assert (result["prediction"], result["generated_token"]) == (expected, token)
    np.testing.assert_allclose(result["probabilities"], [0.6, 0.3, 0.1], atol=1e-7)
    assert np.isclose(result["confidence"], [0.6, 0.3, 0.1][expected if expected >= 0 else 2])
    assert len(calls) == 1
    assert calls[0]['generation_config'].max_new_tokens == 1
    assert calls[0]['generation_config'].do_sample is False


def test_native_metrics_use_emitted_answer_and_keep_invalid_mass(monkeypatch):
    from med_jevify import evaluate

    outputs = iter([
        ({'prediction': 0, 'generated_token': 'A', 'probabilities': [.6, .3, .1],
          'confidence': .6, 'label_mass': .9}, .1),
        ({'prediction': -1, 'generated_token': '?', 'probabilities': [.2, .3, .5],
          'confidence': .4, 'label_mass': .5}, .2),
    ])
    monkeypatch.setattr(evaluate, 'timed_call', lambda *args: next(outputs))
    record, details = evaluate.run_native([
        {'id':'1','label':0,'options':['yes','no']},
        {'id':'2','label':1,'options':['yes','no']},
    ], object(), 'model', 'task')
    assert record.accuracy == .5
    assert record.mean_confidence == .5
    assert np.isclose(record.ece, .4)
    assert np.isclose(record.nll, (-np.log(.6)-np.log(.3))/2)
    assert record.invalid_predictions == 1
    assert details[1]['options'][-1] == '__invalid__'
    assert 'jevify_commit' not in record.as_dict()


def test_conventional_parses_label_prefix_and_generates_multiple_tokens():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('jevify')

    def generate(**kwargs):
        return SimpleNamespace(sequences=torch.tensor([[7, 8, 2, 3]]),
                               logits=(torch.log(torch.tensor([[.6, .3, .1]])),))

    tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
        decode=lambda ids: 'A because the answer is yes',
        batch_decode=lambda ids, **kw: ['A', ' b', 'unrelated'])
    backend = SimpleNamespace(lm=SimpleNamespace(generate=generate), device='cpu',
                              tokenizer=tokenizer, encode=lambda messages: [7, 8])
    jev = SimpleNamespace(backend=backend, system_prompt='classifier')
    result = conventional_generate(jev, {'state': 'context', 'question': '?',
                                         'options': ('yes', 'no')})
    assert result['prediction'] == 0
    assert result['generated_text'].startswith('A because')
    np.testing.assert_allclose(result['probabilities'], [.6, .3, .1], atol=1e-7)


def test_run_probability_backend_uses_explicit_architecture():
    from med_jevify import evaluate

    class Backend:
        def score(self, examples, batch_size):
            return [np.array([0.7, 0.3]) for _ in examples], [0.1] * len(examples)

    record, _ = evaluate.run_probability_backend(
        [{"id": "1", "label": 0, "options": ["yes", "no"]}], Backend(), "m", "t", "mode", 1)
    assert record.architecture == "non-AR"

    record, _ = evaluate.run_probability_backend(
        [{"id": "1", "label": 0, "options": ["yes", "no"]}], Backend(), "m", "t", "mode", 1,
        architecture="AR")
    assert record.architecture == "AR"


def test_laya_batches_only_matching_questions_and_preserves_order():
    class Agent:
        def __init__(self):
            self.calls = []
        def predict_batch(self, states, questions, **kwargs):
            self.calls.append((states, questions))
            value = .9 if questions['decision']['instructions'] == 'first' else .1
            return [{'answers': {'decision': {'probabilities': {'yes': value, 'no': 1 - value}}}} for _ in states]
    adapter = LayaAdapter.__new__(LayaAdapter)
    adapter.agent = Agent()
    examples = [{'state': str(i), 'question': question, 'options': ['yes', 'no']}
                for i, question in enumerate(['first', 'second', 'first'])]
    probabilities, _ = adapter.score(examples, 8)
    assert [round(float(p[0]), 1) for p in probabilities] == [.9, .1, .9]
    assert len(adapter.agent.calls) == 2
    assert adapter.agent.calls[0][0] == ['0', '2']
