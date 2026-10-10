import math
import unittest
from server import normalize, payload, select_samples

class DemoTests(unittest.TestCase):
    def test_image_interleaving_preserves_cases(self):
        samples=select_samples('slake',1000)
        self.assertEqual(len(samples),355)
        self.assertEqual(len({s['id'] for s in samples}),355)
        for a,b in zip(samples[:30],samples[1:31]):
            self.assertNotEqual(a['state']['image'],b['state']['image'])
        self.assertEqual(samples[0]['id'],'slake:test:11938')
    def test_medjev_replaces_hosted_model(self):
        p=payload(1000,'med_qa')
        self.assertIn('medjev',[m['id'] for m in p['models']])
        self.assertNotIn('jev',[m['id'] for m in p['models']])
        self.assertNotIn('medjev',[m['id'] for m in payload(1000,'slake')['models']])
    def test_all_models_have_complete_saved_results(self):
        for dataset in ('slake','med_qa'):
            data=payload(1000,dataset)
            for model in data['models']:
                ledger=data['recorded'][model['id']]
                for sample in data['samples']:
                    self.assertIn(sample['id'],ledger)
                    self.assertIsNotNone(ledger[sample['id']]['latency_ms'])
            for i in range(data['total']):
                sample=data['samples'][i % data['unique']]
                self.assertTrue(all(sample['id'] in data['recorded'][m['id']] for m in data['models']))
    def test_temperature(self):
        sample={'id':'a','options':['a','b'],'label':1}
        r=normalize({'logits':[0,2],'temperature':2,'latency_seconds':.01},sample)
        self.assertAlmostEqual(r['confidence'],1/(1+math.exp(-1)))
        self.assertEqual(r['latency_ms'],10)
        self.assertTrue(r['correct'])
    def test_generated_answer_not_argmax(self):
        sample={'id':'a','options':['a','b'],'label':1}
        r=normalize({'probabilities':[.8,.1,.1],'prediction':1,'options':['a','b','__invalid__']},sample)
        self.assertTrue(r['correct']);self.assertEqual(r['confidence'],.1)
        self.assertIsNone(r['latency_ms'])
    def test_alignment_and_nan_rejected(self):
        s={'id':'a','options':['a','b'],'label':1}
        for r in [{'probabilities':[float('nan'),0]}, {'options':['b','a'],'probabilities':[.5,.5]}]:
            with self.assertRaises(ValueError):normalize(r,s)
    def test_real_datasets(self):
        for name,count,models in [('slake',355,4),('med_qa',1000,5)]:
            p=payload(1000,name)
            self.assertEqual(p['unique'],count);self.assertEqual(len(p['models']),models)
            self.assertEqual(len({r['id'] for r in p['samples']}),count)
            self.assertTrue(p['recorded']['gen'])
            self.assertTrue(p['recorded']['head'])
            self.assertTrue(p['recorded']['ll'])
            self.assertTrue(p['recorded']['nar'])
if __name__=='__main__':unittest.main()
