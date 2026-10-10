#!/usr/bin/env python3
"""Dependency-free recorded inference viewer; optional fresh inference worker."""
import argparse
import json
import math
import threading
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {'medjev': 'results/medjev/details/predictions.jsonl', 'gen': 'results/medgemma_ar/details/predictions.jsonl',
           'head': 'experiment_results/medgemma-head/test.predictions.jsonl',
           'nar': 'experiment_results/laya/correct-test.predictions.jsonl',
           'll': 'results/medgemma_ll/details/predictions.jsonl'}
MODELS = [dict(id='medjev', name='MedJev', subtitle='Local Qwen3.5 · 0.8B decision model'),
          dict(id='gen', name='AR Gen', subtitle='MedGemma · 32-token generation'),
          dict(id='ll', name='AR LL', subtitle='MedGemma · native one-token generation'),
          dict(id='head', name='AR Head', subtitle='MedGemma · frozen backbone + head'),
          dict(id='nar', name='NAR', subtitle='Laya-Med · calibrated decision head')]

def rows(path):
    if not path.exists():
        return []
    # A live writer may have an incomplete final line. Never expose partial JSON.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n') and line.strip()]

def normalize(record, sample):
    options = record.get('options', sample['options'])
    if options[:len(sample['options'])] != sample['options'] or record.get('label', sample['label']) != sample['label']:
        raise ValueError(f"Prediction alignment mismatch: {sample['id']}")
    p = record.get('probabilities')
    if p is None:
        temp = float(record.get('temperature', 1))
        if temp <= 0: raise ValueError('Invalid temperature')
        z = [float(x)/temp for x in record['logits']]
        p = [math.exp(x-max(z)) for x in z]
        p = [x/sum(p) for x in p]
    p = [float(x) for x in p]
    if len(p) not in (len(sample['options']), len(sample['options'])+1) or any(not math.isfinite(x) or x < 0 for x in p) or abs(sum(p)-1)>1e-4:
        raise ValueError('Invalid probability vector')
    pred = int(record.get('prediction', max(range(len(p)), key=p.__getitem__)))
    if pred < -1 or pred >= len(p): raise ValueError('Invalid prediction')
    confidence = p[pred] if pred >= 0 else p[-1] if len(p)>len(sample['options']) else None
    latency = record.get('latency_seconds')
    if latency is not None and (not math.isfinite(latency) or latency < 0): raise ValueError('Invalid latency')
    return dict(prediction=pred, probabilities=p, confidence=confidence, correct=pred==sample['label'],
                latency_ms=latency*1000 if latency is not None else None,
                brier=sum((v-(i==sample['label']))**2 for i,v in enumerate(p)),
                nll=-math.log(max(p[sample['label']], 1e-12)))

def select_samples(dataset, limit):
    samples = [r for r in rows(ROOT/'data/prepared/test.jsonl') if r['dataset']==dataset][:limit]
    if dataset == 'slake':
        # Round-robin image groups without changing any question/image association.
        from collections import defaultdict, deque
        groups = defaultdict(deque)
        for sample in samples:
            groups[sample['state']['image']].append(sample)
        active = deque(groups.values())
        samples = []
        while active:
            group = active.popleft()
            samples.append(group.popleft())
            if group: active.append(group)
    return samples

def payload(limit=1000, dataset="slake"):
    samples = select_samples(dataset, limit)
    if not samples: raise ValueError('No cases in prepared test split')
    index = {r['id']: r for r in samples}
    ledgers = {key: {r['id']: normalize(r,index[r['id']]) for r in rows(ROOT/path) if r.get('id') in index} for key,path in SOURCES.items()}
    # Merge by stable sample ID, never by a presentation step or source filename.
    # Collected MedJev rows belong in the same ledger as the other saved models.
    for r in rows(ROOT/'demo/runs/live.jsonl'):
        key = r.get('model_key')
        if key in ledgers and r.get('id') in index and r.get('dataset') == dataset:
            ledgers[key][r['id']] = normalize(r,index[r['id']])
    models = [m for m in MODELS if dataset=='med_qa' or m['id']!='medjev']
    missing = {m['id']: [ident for ident in index if ident not in ledgers[m['id']]] for m in models}
    missing = {key: ids for key, ids in missing.items() if ids}
    if missing:
        raise ValueError('Incomplete saved comparison: ' + ', '.join(f'{key}: {len(ids)} missing' for key,ids in missing.items()))
    for m in models:
        for prediction in ledgers[m['id']].values():
            if prediction['latency_ms'] is None:
                raise ValueError(f"Missing measured latency for {m['id']}")
    image_ids = {}
    for r in samples:
        image_path = r.get('state', {}).get('image')
        if image_path:
            image_ids.setdefault(image_path, r['id'])
        r['image_url'] = '/image?id='+image_ids[image_path] if image_path else None
        r['image_id'] = Path(r.get('state', {}).get('image') or '').parent.name if r.get('state', {}).get('image') else None
        r['context'] = r.get('state', {}).get('text', '')
        r.pop('state', None)
    result = dict(samples=samples, total=limit, unique=len(samples), models=models,
                  recorded=ledgers, sources={**SOURCES, 'collected': 'demo/runs/live.jsonl'},
                  coverage={m['id']: len(ledgers[m['id']]) for m in models},
                  status='Complete saved comparison · all models ready')
    return result


def prepare_bundle(limit=1000):
    """Persist a complete, validated comparison for both UI playback modes."""
    bundle = {dataset: payload(limit,dataset) for dataset in ('slake','med_qa')}
    directory = ROOT/'demo/data'
    directory.mkdir(exist_ok=True)
    target = directory/'comparison.json'
    temporary = directory/'comparison.tmp'
    temporary.write_text(json.dumps(bundle,allow_nan=False))
    temporary.replace(target)
    return bundle


def _device():
    """Pick a CUDA device with the most free memory, or fall back to CPU."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = 0, -1
    for index in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(index)[0]
        if free > best_free:
            best, best_free = index, free
    return f"cuda:{best}"


def _prepared_sample(dataset, sample_id):
    """Reconstruct a full prepared row (with `state`) for a bundle sample id."""
    for r in rows(ROOT/'data/prepared/test.jsonl'):
        if r.get('dataset') == dataset and r.get('id') == sample_id:
            return r
    return None


# Global cache for loaded model adapters (loaded once, reused across requests).
_MODEL_CACHE = {}
_LOAD_LOCK = threading.RLock()
_INFER_LOCK = threading.Lock()


def _get_adapter(model_key):
    """Return a cached adapter for the given model key, loading it on first use."""
    with _LOAD_LOCK:
        return _load_adapter(model_key)


def _load_adapter(model_key):
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    import sys
    sys.path[:0] = [str(ROOT), str(ROOT/'src')]
    import torch

    device = _device()
    dtype = 'bf16' if device.startswith('cuda') else 'fp32'

    if model_key == 'medjev':
        from med_jevify.medjev_adapter import MedJevAdapter
        adapter = MedJevAdapter(run='junma/MedJev-Qwen3.5-0.8B', device=device, dtype=dtype)
    elif model_key == 'gen':
        from med_jevify.upstream import load_jevify
        jev = load_jevify().Jevify.from_transformers('google/medgemma-4b-it', device=device, dtype='auto', no_think=False)
        adapter = jev
    elif model_key == 'll':
        # Gen and LL share one MedGemma instance to avoid loading it twice.
        adapter = _load_adapter('gen')
    elif model_key == 'head':
        from med_jevify.medgemma_direct import MedGemmaDirectAdapter
        adapter = MedGemmaDirectAdapter(run=str(ROOT/'models/medgemma-direct-headonly'), device=device, dtype=dtype)
    elif model_key == 'nar':
        from tta.adapters.laya_med import LayaMedAdapter
        adapter = LayaMedAdapter(str(ROOT/'models/laya-med-vision-lora/best.pt'), device=device,
                                 manifest=str(ROOT/'data/prepared/manifest.json'), batch_size=1)
    else:
        raise ValueError(f"Unsupported model: {model_key}")

    _MODEL_CACHE[model_key] = adapter
    return adapter


def run_live_inference(sample, model_key):
    """Run live inference for a sample and model, returning a normalized record."""
    import sys
    sys.path[:0] = [str(ROOT), str(ROOT/'src')]
    import torch

    # Reconstruct the prepared row (adapters need `state`, which the bundle strips).
    prepared = _prepared_sample(sample['dataset'], sample['id']) or sample
    adapter = _get_adapter(model_key)

    if model_key == 'medjev':
        probs, lat = adapter.score([prepared], 1)
        probabilities = probs[0].tolist()
        latency = lat[0]
    elif model_key == 'gen':
        from med_jevify.evaluate import conventional_generate, timed_call
        result, latency = timed_call(adapter, conventional_generate, prepared)
        probabilities = result['probabilities']
    elif model_key == 'll':
        from med_jevify.evaluate import native_generate, timed_call
        result, latency = timed_call(adapter, native_generate, prepared)
        probabilities = result['probabilities']
    elif model_key == 'head':
        probs, lat = adapter.score([prepared], 1)
        probabilities = probs[0].tolist()
        latency = lat[0]
    elif model_key == 'nar':
        from tta.common import collect_logits_timed, to_probabilities
        record = collect_logits_timed(adapter.model, [prepared], adapter.processor, adapter.tokenizer,
                                      adapter.max_length, adapter.device, 1)[0]
        probabilities = to_probabilities(record['logits'], adapter.temperature()).tolist()
        latency = record['latency_s']
    else:
        raise ValueError(f"Unsupported model: {model_key}")

    probabilities = [float(p) for p in probabilities]
    label = prepared.get('label', sample.get('label'))
    prediction = int(max(range(len(probabilities)), key=probabilities.__getitem__)) if probabilities else -1
    correct = prediction == label
    confidence = probabilities[prediction] if 0 <= prediction < len(probabilities) else None
    brier = sum((p - (i == label))**2 for i, p in enumerate(probabilities)) if label is not None else None
    nll = -math.log(max(probabilities[label], 1e-12)) if label is not None and label < len(probabilities) else None

    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'confidence': confidence,
        'correct': correct,
        'latency_ms': latency * 1000,
        'brier': brier,
        'nll': nll,
    }

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed=urlparse(self.path)
        try:
            if parsed.path=='/api/data':
                dataset=parse_qs(parsed.query).get("dataset", ["slake"])[0]
                inference_mode=parse_qs(parsed.query).get("inference", ["stored"])[0]
                if inference_mode == "live":
                    # For live inference, we need to compute on-demand
                    # This would require a different approach - for now, we'll stick with stored
                    # but the frontend handles live inference differently
                    data=json.dumps(self.server.bundle[dataset], allow_nan=False).encode(); mime='application/json'
                else:
                    data=json.dumps(self.server.bundle[dataset], allow_nan=False).encode(); mime='application/json'
            elif parsed.path=='/api/inference':
                # Live inference for a single sample
                dataset=parse_qs(parsed.query).get("dataset", ["slake"])[0]
                model_key=parse_qs(parsed.query).get("model", ["medjev"])[0]
                sample_id=parse_qs(parsed.query).get("sampleId", [""])[0]
                if not sample_id:
                    self.send_error(400, "sampleId is required")
                    return
                # Find the sample in the bundle
                bundle = self.server.bundle.get(dataset)
                if not bundle:
                    self.send_error(404, f"Dataset {dataset} not found")
                    return
                sample = None
                for s in bundle['samples']:
                    if s['id'] == sample_id:
                        sample = s
                        break
                if not sample:
                    self.send_error(404, f"Sample {sample_id} not found in dataset {dataset}")
                    return
                # Run live inference
                with _INFER_LOCK:
                    result = run_live_inference(sample, model_key)
                data=json.dumps(result, allow_nan=False).encode(); mime='application/json'
            elif parsed.path=='/image':
                ident=parse_qs(parsed.query).get('id',[''])[0]
                path=self.server.image_paths.get(ident)
                if not path: self.send_error(404); return
                if not path.is_relative_to((ROOT/'data').resolve()): self.send_error(403); return
                data=path.read_bytes(); mime=mimetypes.guess_type(path)[0] or 'image/jpeg'
            else:
                name='index.html' if parsed.path=='/' else parsed.path.lstrip('/')
                path=(ROOT/'demo/static'/name).resolve()
                if not path.is_relative_to(ROOT/'demo/static') or not path.is_file(): self.send_error(404); return
                data=path.read_bytes(); mime=mimetypes.guess_type(path)[0] or 'application/octet-stream'
            self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Cache-Control','private, max-age=3600' if parsed.path=='/image' else 'no-store'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        except Exception as exc:
            self.send_error(500,str(exc).replace('\n',' ')[:300])
    def log_message(self, *_): pass

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--port',type=int,default=8765); parser.add_argument('--samples',type=int,default=1000)
    args=parser.parse_args()
    if args.samples<1: parser.error('--samples must be positive')
    bundle=prepare_bundle(args.samples)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler); server.limit=args.samples; server.bundle=bundle
    server.image_paths = {r['id']: Path(r['state']['image']).resolve()
                          for r in rows(ROOT/'data/prepared/test.jsonl')
                          if r['dataset']=='slake' and r.get('state', {}).get('image')}
    print(f'SysOne-Med · Inference Observatory → http://localhost:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
