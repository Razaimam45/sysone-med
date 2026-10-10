"""Policy and runner regression tests without checkpoint downloads or a GPU."""
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from tta import method_memory, method_permutation, method_selective
from tta.run_tta import build_parser, run_tta, METHODS, AVAILABLE_METHODS


def row(id='a', dataset='demo', label=1):
    return dict(id=id, dataset=dataset, label=label, question='Which diagnosis?',
                options=['benign', 'malignant', 'unknown'], state={'text': 'lung mass', 'image': 'test.png'})


class Adapter:
    def __init__(self, rows, scorer=None):
        self.splits = {'test': rows, 'calibration': [row('calibration')]}
        self.calls = []
        self.scorer = scorer or (lambda r: np.array([3., 0., 0.]))
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def temperature(self):
        return 1.

    def score_logits(self, rows, batch_size):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        assert all(r['label'] == 0 for r in rows)
        self.calls.extend(copy.deepcopy(rows))
        return [self.scorer(r) for r in rows]


def args(tmp_path, **kw):
    return dict(baseline='laya_med', datasets=['demo'], max_examples=None,
                batch_size=2, output=tmp_path / 'result.json', **kw)


def test_permutation_unmaps_probabilities_and_caps_views(tmp_path):
    original = row()
    snapshot = copy.deepcopy(original)
    def scorer(r):
        # Candidate identity plus an order bias: average must remove positional advantage.
        return np.array([{'benign': 0., 'malignant': 2., 'unknown': -1.}[o]
                         + (4. if i == 0 else 0.) for i, o in enumerate(r['options'])])
    adapter = Adapter([original], scorer)
    result = method_permutation.run(args(tmp_path, permutations=10), adapter)
    assert result['rows'][0]['views'] == 6
    assert result['rows'][0]['base_prediction'] == 0
    assert result['rows'][0]['prediction'] == 1
    assert len(adapter.calls) == 6
    assert original == snapshot
    assert all(r['state'] == original['state'] for r in adapter.calls)
    assert adapter.weight.item() == 1 and adapter.weight.grad is None


def test_memory_uses_only_previous_predictions_and_preserves_images(tmp_path):
    rows = [row('a'), row('b'), row('c', 'second')]
    def scorer(r):
        memory = 'Previous model guesses' in r['state']['text']
        return np.array([0., 4., 0.]) if memory else np.array([4., 0., 0.])
    adapter = Adapter(rows, scorer)
    settings = args(tmp_path, memory_k=1)
    settings['datasets'] = ['demo', 'second']
    result = method_memory.run(settings, adapter)
    assert [r['retrieved_ids'] for r in result['rows']] == [[], ['a'], []]
    assert result['rows'][1]['prediction'] == 1
    contextual = adapter.calls[-1]
    assert contextual['state']['image'] == 'test.png'
    assert 'Guessed answer: benign' in contextual['state']['text']
    changed = copy.deepcopy(rows)
    for r in changed:
        r['label'] = 2
    repeated = method_memory.run(settings, Adapter(changed, scorer))
    assert [r['probabilities'] for r in repeated['rows']] == [r['probabilities'] for r in result['rows']]
    fresh = method_memory.run(settings, adapter)
    assert fresh['rows'][0]['memory_size_before'] == 0


def test_memory_capacity_and_no_self_retrieval(tmp_path):
    rows = [row('a'), row('b'), row('b'), row('c')]
    result = method_memory.run(args(tmp_path, memory_capacity=1), Adapter(rows))
    assert max(r['memory_size_before'] for r in result['rows']) == 1
    assert result['rows'][2]['retrieved_ids'] == []


@pytest.mark.parametrize('stable', [True, False])
def test_selective_gates_additional_views(tmp_path, stable):
    scorer = ((lambda r: np.array([3. if o == 'benign' else 0. for o in r['options']]))
              if stable else None)
    adapter = Adapter([row()], scorer)
    result = method_selective.run(args(tmp_path), adapter)
    record = result['rows'][0]
    assert record['selected'] is (not stable)
    assert len(adapter.calls) == (2 if stable else 4)
    assert record['additional_scored_examples'] == (1 if stable else 3)
    assert np.isclose(sum(record['probabilities']), 1.)
    if stable:
        assert record['prediction'] == record['base_prediction']


@pytest.mark.parametrize('module', [method_permutation, method_memory, method_selective])
def test_single_candidate_and_empty_input(tmp_path, module):
    single = {**row(), 'options': ['only'], 'label': 0}
    result = module.run(args(tmp_path), Adapter([single], lambda r: np.array([0.])))
    assert result['rows'][0]['prediction'] == 0
    assert result['abstention_rate'] == 0
    settings = args(tmp_path)
    settings['datasets'] = None
    empty = module.run(settings, Adapter([]))
    assert empty['test_n'] == 0


def test_runner_writes_six_reports_one_load_and_no_stale_rlcd(tmp_path, monkeypatch):
    import tta.run_tta as runner
    adapter = Adapter([row()])
    builds = []
    def build(*args, **kwargs):
        builds.append((args, kwargs))
        return adapter
    monkeypatch.setattr(runner, 'build_adapter', build)
    out = tmp_path / 'qwen_direct_lora_tta'
    out.mkdir()
    (out / 'rlcd.json').write_text('{"stale": true}')
    parsed = build_parser().parse_args(['--output-dir', str(out), '--methods', *AVAILABLE_METHODS])
    parsed.datasets = ['demo']
    run_tta(parsed)
    assert len(builds) == 1
    assert all((out / f'{method}.json').is_file() for method in AVAILABLE_METHODS)
    report = json.loads((tmp_path / 'qwen_direct_lora' / 'tta_summary.json').read_text())
    assert set(report['methods']) == set(AVAILABLE_METHODS)
    assert all(report['methods'].values())
    assert report['methods']['routing']['results'][0]['accepted_accuracy'] is not None
    assert report['methods']['conformal']['results'][0]['mean_set_size'] >= 1
    assert adapter.weight.grad is None


def test_shell_runs_enabled_configurations_with_safe_argument_forwarding(tmp_path):
    log = tmp_path / 'calls.jsonl'
    fake = tmp_path / 'fake python'
    fake.write_text('#!/usr/bin/env python3\nimport sys, os, json\n'
                    'with open(os.environ["CALL_LOG"], "a") as f:\n'
                    '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
    fake.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / 'tta/run_all_tta.sh'
    subprocess.run(['bash', str(script), '--methods', 'calibrated'], cwd=tmp_path,
                   env={**os.environ, 'TTW_PYTHON': str(fake), 'CALL_LOG': str(log),
                        'OUTPUT_ROOT': str(tmp_path / 'results with spaces')}, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 9  # User has disabled configuration #5.
    assert len({c[c.index('--output-dir') + 1] for c in calls}) == 9
    assert all(c[-2:] == ['--methods', 'calibrated'] for c in calls)
    assert calls[6][calls[6].index('--backbone') + 1] == 'qwen'
    assert calls[8][calls[8].index('--checkpoint') + 1] == 'Qwen/Qwen2.5-VL-3B-Instruct'


@pytest.mark.parametrize('baseline,module,checkpoint,backbone', [
    ('laya_med', 'laya_med', 'models/laya-med/best.pt', None),
    ('laya_med', 'laya_med', 'models/laya-med-vision-lora/best.pt', None),
    ('laya', 'laya', 'convaiinnovations/laya', None),
    ('medjev', 'medjev', 'junma/MedJev-Qwen3.5-0.8B', None),
    ('medgemma_direct_headonly', 'medgemma_direct', 'models/medgemma-direct-headonly', None),
    ('medgemma_direct_lora', 'medgemma_direct', 'models/medgemma-direct', None),
    ('medgemma_ll', 'medgemma_ll', 'google/medgemma-4b-it', None),
    ('medgemma_direct_headonly', 'medgemma_direct', 'models/qwen_direct_headonly', 'qwen'),
    ('medgemma_direct_lora', 'medgemma_direct', 'models/qwen_direct_lora', 'qwen'),
    ('medgemma_ll', 'medgemma_ll', 'Qwen/Qwen2.5-VL-3B-Instruct', None),
])
def test_factory_passes_checkpoint_manifest_and_backbone(monkeypatch, baseline, module, checkpoint, backbone):
    import importlib
    import inspect
    from tta.build import build_adapter
    adapter_module = importlib.import_module('tta.adapters.' + module)
    signature = inspect.signature(adapter_module.build)
    calls = []
    sentinel = object()
    def build(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)
        return sentinel
    monkeypatch.setattr(adapter_module, 'build', build)
    assert build_adapter(baseline, checkpoint, 'cpu', '/custom/manifest.json', 2,
                         backbone=backbone) is sentinel
    assert calls[0]['manifest'] == '/custom/manifest.json'
    assert checkpoint in calls[0].values()
    if backbone:
        assert calls[0]['backbone'] == backbone


def test_only_calibrated_preserves_existing_outputs(tmp_path, monkeypatch):
    import tta.run_tta as runner
    adapter = Adapter([row()])
    monkeypatch.setattr(runner, 'build_adapter', lambda *a, **kw: adapter)
    original = {}
    for method in ('routing', 'conformal', 'permutation', 'memory', 'selective'):
        # Valid empty JSON reports allow consolidated reporting without reruns.
        payload = {'method': method, 'test_n': 0, 'results': [], 'rows': []}
        original[method] = json.dumps(payload)
        (tmp_path / f'{method}.json').write_text(original[method])
    parsed = build_parser().parse_args(['--output-dir', str(tmp_path), '--methods', 'calibrated'])
    parsed.datasets = ['demo']
    run_tta(parsed)
    assert (tmp_path / 'calibrated.json').is_file()
    assert json.loads((tmp_path / 'summary.json').read_text())['methods'] == ['calibrated']
    for method, content in original.items():
        assert (tmp_path / f'{method}.json').read_text() == content
    assert len(adapter.calls) == 1  # Shared base only; short calibration falls back.
