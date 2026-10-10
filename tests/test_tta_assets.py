"""Tests for the TTA asset generator (analysis/make_tta_assets.py).

These guard the aggregation contract the paper table depends on: per-dataset
aggregation followed by a macro-average, tie-aware risk--coverage curves, a
no-TTA reference recomputed from the identical TTA test rows, conformal coverage
recomputed from the stored prediction sets, and a roster that carries only the
pipelines with adaptation reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))

import make_tta_assets as tta  # noqa: E402


def rows(spec, dataset='pubmedqa'):
    """Build routing-shaped rows from (confidence, correct) pairs."""
    return [dict(id=f'{dataset}:{i}', dataset=dataset, confidence=c,
                 correct=int(ok), label=0, prediction=0, score=c, margin=c)
            for i, (c, ok) in enumerate(spec)]


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def test_risk_coverage_is_monotone_and_pooled_at_full_retention():
    data = rows([(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.5, False)])
    coverage, risk = tta.risk_coverage(data)
    assert np.allclose(coverage, np.arange(1, 6) / 5)
    assert np.all(np.diff(coverage) > 0)
    assert risk[0] == 0.0                      # the most confident example is correct
    assert risk[-1] == pytest.approx(2 / 5)    # retain-all risk is the pooled error rate
    assert 1 - risk[-1] == tta.mean_accuracy(data)


def test_risk_coverage_averages_within_exact_ties():
    data = rows([(0.9, True), (0.9, False), (0.1, True)])
    coverage, risk = tta.risk_coverage(data)
    # Inside the 0.9 tie, one retained example is correct half of the time.
    assert risk[0] == pytest.approx(0.5)
    assert risk[1] == pytest.approx(0.5)
    assert risk[2] == pytest.approx(1 / 3)      # retain-all: one error in three


def test_accuracy_at_reads_the_matching_retention_level():
    data = rows([(0.9, True), (0.8, False), (0.7, True), (0.6, True)])
    coverage, risk = tta.risk_coverage(data)
    assert tta.accuracy_at(coverage, risk, 1.0) == pytest.approx(0.75)
    assert tta.accuracy_at(coverage, risk, 0.25) == pytest.approx(1.0)
    assert tta.accuracy_at(coverage, risk, 0.5) == pytest.approx(0.5)
    assert tta.accuracy_at(coverage, risk, 0.75) == pytest.approx(2 / 3)


def test_reported_accuracy_stays_inside_the_attainable_bracket():
    data = rows([(0.9, True), (0.9, True), (0.9, False), (0.9, False), (0.1, True)])
    coverage, risk = tta.risk_coverage(data)
    low, high = tta.bracket_at(data, target=0.4)
    value = tta.accuracy_at(coverage, risk, 0.4)
    assert low <= value <= high
    assert low <= tta.reference_at(data, target=0.4) <= high
    assert np.isnan(tta.accuracy_at(np.array([]), np.array([]), 0.8))


def test_conformal_stats_recompute_coverage_from_options():
    # `covered` is the report's own flag; the runner computes it as
    # `label in boolean_mask`, so it must be ignored whenever options are known.
    data = [dict(id='a', set_size=1, covered=False, prediction=0, label=0,
                 set=['right']),
            dict(id='b', set_size=1, covered=True, prediction=1, label=0,
                 set=['wrong']),
            dict(id='c', set_size=2, covered=True, prediction=0, label=0,
                 set=['right', 'other']),
            dict(id='d', set_size=3, covered=True, prediction=0, label=1,
                 set=['x', 'y', 'z'])]
    options = {'a': ['right'], 'b': ['truth', 'wrong'],
               'c': ['right', 'other'], 'd': ['x', 'y', 'z']}
    got = tta.conformal_stats(data, options)
    assert got['coverage'] == pytest.approx(0.75)     # a and c are covered
    assert got['mean_set_size'] == pytest.approx(1.75)
    assert got['singleton_rate'] == pytest.approx(0.5)
    assert got['decided_accuracy'] == pytest.approx(0.5)   # one singleton is wrong
    assert got['missing_options'] == 0


def test_conformal_stats_falls_back_to_the_report_flag_without_options():
    data = [dict(id='a', set_size=1, covered=True, prediction=0, label=0, set=['r']),
            dict(id='b', set_size=1, covered=False, prediction=0, label=0, set=['w'])]
    got = tta.conformal_stats(data, {})
    assert got['coverage'] == pytest.approx(0.5)
    assert got['missing_options'] == 2


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    perfect = [dict(confidence=1.0, correct=1) for _ in range(10)]
    assert tta.ece(perfect) == pytest.approx(0.0)
    assert np.isnan(tta.ece([]))


def test_macro_average_skips_absent_and_invalid_datasets():
    assert tta.macro([0.5, 1.0, None, float('nan')]) == pytest.approx(0.75)
    assert np.isnan(tta.macro([None, float('nan')]))


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def test_heat_and_num_render_latex_cells():
    assert tta.heat(0.0, 0.0, 1.0).startswith(r'\cellcolor[RGB]')
    assert tta.heat(0.0, 0.0, 1.0) != tta.heat(1.0, 0.0, 1.0)
    assert tta.heat(float('nan'), 0.0, 1.0) == ''
    assert tta.num(73.4) == '73.4'
    assert tta.num(None) == '---'
    assert tta.num(float('nan')) == '---'


def test_reported_rows_carry_the_four_mechanisms():
    assert [label for label, _ in tta.REPORTED_ROWS] == [
        'no TTA', '+~Routing', '+~Conformal', '+~Online recal.']


# --------------------------------------------------------------------------- #
# Integration against the checked-in reports
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def records():
    if not (ROOT / 'tta_results').is_dir():
        pytest.skip('no tta_results directory')
    return tta.collect([tta.load_report(spec) for spec in tta.BASELINES],
                       tta.load_options())


def test_validate_reports_no_invariant_violations(records):
    assert tta.validate(records) == []


def test_roster_excludes_pipelines_without_adaptation_reports(records):
    assert [r['display'] for r in records] == [
        'MedJev', 'Direct-Head', 'Direct-LoRA', 'Laya', 'LayaMed']
    assert {r['display'] for r in records if r['ours']} == {
        'Direct-Head', 'Direct-LoRA', 'LayaMed'}
    assert 'MedGemma-LL' not in {r['display'] for r in records}
    for rec in records:
        if rec['modality'] == 'text':
            assert set(rec['datasets']) <= set(tta.TEXT_DS)
        else:
            assert set(rec['datasets']) == set(tta.DS)


def test_reference_and_adapted_cells_share_the_same_test_rows(records):
    for rec in records:
        for dataset in rec['datasets']:
            routed = rec['routing'][dataset]['rows']
            refit = rec['drift'][dataset]['rows']
            assert {r['id'] for r in routed} == {r['id'] for r in refit}
            assert tta.mean_accuracy(routed) == tta.base_cells(rec)[dataset]


def test_all_reported_cells_are_valid_accuracies(records):
    for rec in records:
        cells = (list(tta.base_cells(rec).values()) + list(tta.routing_cells(rec).values())
                 + list(tta.recalibration_cells(rec).values())
                 + list(tta.conformal_cells(rec).values()))
        assert cells and all(0.0 <= value <= 1.0 for value in cells)
        assert len(tta.routing_cells(rec)) == len(rec['datasets'])
        # The conformal row may dash a dataset whose sets are never singletons.
        assert tta.conformal_cells(rec)
        for stats in rec['conformal'][tta.ALPHA].values():
            assert 0.0 <= stats['coverage'] <= 1.0
            assert stats['rows'] == 50


def test_macro_average_dashes_text_only_modality_gaps(records):
    text = next(r for r in records if r['display'] == 'MedJev')
    cells = dict(tta.base_cells(text))
    cells['avg'] = tta.macro(list(cells.values()))
    assert set(cells) == set(tta.TEXT_DS) | {'avg'}
    assert cells['avg'] == pytest.approx(np.mean([cells[d] for d in tta.TEXT_DS]))


def test_rendered_table_is_balanced_and_has_one_block_per_pipeline(records, tmp_path):
    text = tta.render_table6(records)
    assert r'\label{tab:tta_main}' in text
    assert text.count(r'\begin{table*}') == text.count(r'\end{table*}') == 1
    assert text.count(r'\begin{tabular*}') == text.count(r'\end{tabular*}') == 1
    assert text.rstrip().endswith(r'\end{table*}')
    assert text.count(r'\hspace{1.1em}') == 3 * len(records)   # three adapted rows per block
    for name in tta.DN:
        assert name in text
    assert 'MedGemma' not in text
    written = tta.write_table6(records, tmp_path)
    assert written == tmp_path / 'table6_tta_main.tex'
    assert written.read_text() == text


def test_artifact_records_operating_points_and_panel_data(records, tmp_path):
    tta.write_table6(records, tmp_path)
    artifact = tta.build_artifact(records, {}, tmp_path / 'table6_tta_main.tex',
                                  tmp_path / 'figure7_tta_methods.png')
    payload = json.loads(json.dumps(artifact, default=float))
    assert payload['operating_points'] == dict(
        retention=tta.RETENTION, alpha=tta.ALPHA, target_coverage=tta.TARGET,
        examples_per_dataset=50)
    assert payload['table6']['label'] == 'tab:tta_main'
    assert len(payload['table6']['rows']) == len(records)
    assert 'MedGemma (LL)' in payload['roster']['excluded']
    for pipeline in payload['pipelines']:
        assert 0.0 <= pipeline['ranking']['aurc'] <= 1.0
        retained = pipeline['ranking']['retained_accuracy']['0.8']
        assert retained >= pipeline['macro_accuracy']['no TTA'] - 1e-9


def test_figure7_is_written(records, tmp_path):
    pytest.importorskip('matplotlib')
    path, panels = tta.figure7(records, tmp_path)
    assert path.stat().st_size > 10_000
    assert set(panels) == {'selective', 'conformal', 'recalibration'}
    assert {p['pipeline'] for p in panels['selective']} == {r['display'] for r in records}
    for point in panels['conformal'][0]['points']:
        assert 0.0 <= point['realised'] <= 1.0
    assert json.loads(json.dumps(panels, default=float))
