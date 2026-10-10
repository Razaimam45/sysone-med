import copy
import numpy as np
import torch
from tta.method_prior import run


class FakeAdapter:
    def __init__(self, label=0):
        self.splits = {'test': [dict(id='a', dataset='demo', label=label,
                                    question='Diagnosis?', options=['a', 'b'],
                                    state={'text': 'Clinical evidence', 'image': 'image.png'})]}
        self.calls = []

    def temperature(self):
        return 1.

    def score_logits(self, rows, batch_size):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert all(r['label'] == 0 for r in rows)
        self.calls.extend(copy.deepcopy(rows))
        return [np.log([.55, .45] if r['state'] else [.95, .05]) for r in rows]


def settings(tmp_path, **extra):
    return dict(baseline='fake', datasets=['demo'], batch_size=1,
                max_examples=None, output=tmp_path / 'prior.json', **extra)


def test_prior_changes_answer_without_truth_or_gradients(tmp_path):
    adapter = FakeAdapter()
    original = copy.deepcopy(adapter.splits)
    result = run(settings(tmp_path), adapter)
    assert result['rows'][0]['prediction'] == 1
    assert result['rows'][0]['base_prediction'] == 0
    assert adapter.calls[-1]['state'] == ''
    assert adapter.splits == original
    repeated = run(settings(tmp_path), FakeAdapter(label=1))
    assert result['rows'][0]['probabilities'] == repeated['rows'][0]['probabilities']
    assert result['config']['guaranteed_improvement'] is False


def test_confidence_gate_preserves_original(tmp_path):
    adapter = FakeAdapter()
    result = run(settings(tmp_path, prior_confidence=.5), adapter)
    assert len(adapter.calls) == 1
    assert result['changed_n'] == 0


def test_no_evidence_skips_null_view(tmp_path):
    adapter = FakeAdapter()
    adapter.splits['test'][0]['state'] = ''
    result = run(settings(tmp_path), adapter)
    assert len(adapter.calls) == 1
    assert result['changed_n'] == 0
