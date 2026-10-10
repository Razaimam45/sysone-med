import copy
import numpy as np
import pytest
import torch
from tta.method_calibrated import POLICIES, choose_policy, candidates, reference_view, run


def test_selection_requires_improvement_in_both_blocks():
    labels = np.ones(4, dtype=int)
    predictions = {p: [0, 0, 0, 0] for p in POLICIES}
    predictions['reference'] = [1, 1, 0, 0]
    assert choose_policy(predictions, labels, 2) == 'base'
    predictions['prior_reinforce'] = [1, 0, 1, 0]
    assert choose_policy(predictions, labels, 2) == 'prior_reinforce'


class Adapter:
    def __init__(self):
        self.seen = []
    def temperature(self):
        return 1.
    def score_logits(self, rows, batch_size):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert all(r['label'] == 0 for r in rows)
        self.seen.extend(copy.deepcopy(rows))
        return [np.array([.2, .4]) for r in rows]


def row(id='x', label=0):
    return dict(id=id, dataset='demo', question='Which diagnosis?',
                options=['yes', 'no'], label=label,
                state={'text': 'lung mass', 'image': 'image.png'})


def test_test_labels_do_not_affect_candidates_and_images_preserved():
    adapter = Adapter()
    bank = [row('cal', 1)]
    a = candidates(adapter, row(label=0), bank)
    b = candidates(adapter, row(label=1), bank)
    for policy in POLICIES:
        np.testing.assert_array_equal(a[policy], b[policy])
    view = reference_view(row(), bank)
    assert view['state']['image'] == 'image.png'
    assert 'reference only: no' in view['state']['text']


def test_short_calibration_uses_base(tmp_path):
    adapter = Adapter()
    adapter.splits = {'calibration': [row('cal')], 'test': [row()]}
    result = run(dict(baseline='fake', datasets=['demo'], max_examples=None,
                      batch_size=1, output=tmp_path/'result.json'), adapter)
    assert result['changed_n'] == 0
    assert result['config']['policies'] == {'demo': 'base'}


def test_overlap_rejected(tmp_path):
    adapter = Adapter()
    adapter.splits = {'calibration': [row()], 'test': [row()]}
    with pytest.raises(ValueError, match='overlap'):
        run(dict(baseline='fake', datasets=['demo'], max_examples=None,
                 batch_size=1, output=tmp_path/'result.json'), adapter)
