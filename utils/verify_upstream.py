"""Compare local scoring/metrics with a checkout of the pinned upstream commit."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream-dir', required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(root/'src'))
from med_jevify.evaluate import label_logits, summarize
from med_jevify.calibration import fit_temperature
from med_jevify.upstream import verify_jevify
import jevify
upstream = args.upstream_dir
spec = importlib.util.spec_from_file_location('upstream_eval', upstream/'train/eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
verify_jevify()
installed = Path(jevify.__file__).parent
files = list((upstream/'src/jevify').rglob('*.py'))
assert all(p.read_bytes() == (installed/p.relative_to(upstream/'src/jevify')).read_bytes() for p in files)
for probs, labels in [
    (np.array([[.8,.2],[.4,.6]]), [0,1]),
    (np.array([[1.,0.,0.],[0.,.5,.5],[.2,.3,.5]]), [1,2,0]),
]:
    expected = module.metrics(probs, np.eye(probs.shape[1])[labels])
    result = summarize('model','task','raw_jevify',probs,labels,[.1]*len(labels))
    for own, source in [('accuracy','acc'),('mean_confidence','mean_conf'),('ece','ece'),('nll','nll'),('brier','brier'),('jsd','jsd')]:
        np.testing.assert_allclose(getattr(result,own),expected[source],rtol=1e-12,atol=1e-12)
    logits=np.log(probs+1e-9)
    np.testing.assert_allclose(fit_temperature(logits,labels).temperature,
                               module.fit_temperature(logits,np.eye(probs.shape[1])[labels]),rtol=1e-12)
be=SimpleNamespace(next_token=lambda messages: SimpleNamespace(logprobs={'A':np.log(.7),' a':np.log(.1),'B':np.log(.00001234)}))
example={'state':'context','question':'Question?', 'options':('yes','no','maybe')}
jev=jevify.Jevify(be)
actual, mass=label_logits(jev,example)
keys, expected, expected_mass=module.label_logits(be,example['state'],{'type':'choice','instructions':example['question'],'criteria':{x:None for x in example['options']}})
np.testing.assert_array_equal(actual,expected)
assert mass==expected_mass
report={'commit': '66f49bff53b3276fecc6b4bd16a31de277bebfaf',
        'installed_source_files_identical':len(files),
        'label_logits_parity':'exact on fixture including token variants, tiny probability and absent label',
        'metric_parity':'accuracy, confidence, ECE, NLL, Brier, JSD within 1e-12 on binary/multiclass fixtures',
        'temperature_parity':'same optimum within 1e-12 on fixtures',
        'scope':'algorithm parity, not reproduction of upstream datasets/checkpoints/training'}
(root/'results/details').mkdir(parents=True, exist_ok=True)
(root/'results/details/upstream_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
