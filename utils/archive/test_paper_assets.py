"""Scientific reporting regressions: ties, supports, cohort matching and export parity."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_assets',ROOT/'utils/make_paper_assets.py')
paper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)

def test_selective_ties_are_order_independent():
    rows=[{'probabilities':[.7,.3],'prediction':0,'label':label} for label in [0,1,1,0]]
    x,risk=paper.risk_curve(rows)
    np.testing.assert_allclose(risk,.5)
    np.testing.assert_allclose(paper.risk_curve(rows[::-1])[1],risk)
    np.testing.assert_allclose(x,[.25,.5,.75,1])

def test_nll_does_not_clip_confident_wrong_logits():
    row=paper.softmax_row({'logits':[0.,-100.],'label':1},1.)
    assert np.isclose(paper.stats([row])['nll'],100.)

def test_ece_includes_confidence_one():
    rows=[{'probabilities':[1.,0.],'prediction':0,'label':0}]
    assert paper.stats(rows)['ece']==0

def test_matched_cohorts_and_probability_exports():
    groups,pilot,full,test,fullpred=paper.load_data()
    for d in paper.DS:
        measured=paper.stats(paper.items(groups,paper.MODELS[3],d))
        for metric,value in measured.items():
            assert np.isclose(value,pilot['calibrated']['per_dataset'][d][metric],rtol=1e-8,atol=1e-10)
        rows=[paper.softmax_row(r,full['temperature']) for r in fullpred if r['dataset']==d]
        for metric,value in paper.stats(rows).items():
            assert np.isclose(value,full['calibrated']['per_dataset'][d][metric],rtol=1e-8,atol=1e-10)
    assert len(test)==19586

def test_generation_selection_uses_emitted_answer_confidence():
    row={'prediction':1,'label':0,'probabilities':[.9,.1],'confidence':.1}
    np.testing.assert_allclose(paper.confidence([row]),[.1])

def test_public_method_taxonomy_preserves_source_modes():
    assert paper.METHODS[0][2]=='conventional_llm'
    assert paper.METHODS[1][2]=='conventional_llm'
    assert paper.METHODS[2][1]==paper.METHODS[1][1]
    assert all('jevify' not in method[0].lower() for method in paper.METHODS)
