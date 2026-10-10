#!/usr/bin/env python3
"""Collect measured predictions for the demo using existing project adapters."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from server import rows, select_samples

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['slake','med_qa'],default='slake')
    p.add_argument('--models',nargs='+',choices=['medjev','gen','head','ll','nar'])
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--ar-model',default='google/medgemma-4b-it')
    p.add_argument('--medjev',default='junma/MedJev-Qwen3.5-0.8B')
    p.add_argument('--head',default=str(ROOT/'models/medgemma-direct-headonly'))
    p.add_argument('--nar',default=str(ROOT/'models/laya-med/best.pt'))
    args=p.parse_args()
    if args.samples<1:p.error('--samples must be positive')
    keys=args.models or (['medjev','gen','head','nar'] if args.dataset=='med_qa' else ['gen','head','nar'])
    if args.dataset=='slake' and 'medjev' in keys:p.error('MedJev is text-only; select MedQA for MedJev.')
    samples=select_samples(args.dataset, args.samples)
    if not samples:p.error('No prepared test samples')
    out=ROOT/'demo/runs';out.mkdir(exist_ok=True)
    import fcntl
    lock=(out/'collector.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Another collector is running. Wait for it to finish.')
    def status(message):
        (out/'status.tmp').write_text(message);(out/'status.tmp').replace(out/'status.txt');print(message,flush=True)
    existing=rows(out/'live.jsonl')
    done={(r['dataset'],r['model_key'],r['step']) for r in existing if r['dataset']==args.dataset and r['id']==samples[r['step'] % len(samples)]['id']}
    try:
        for key in keys:
            pending=[i for i in range(args.samples) if (args.dataset,key,i) not in done]
            if not pending:continue
            status(f'Loading {key} for {args.dataset}…')
            adapter=None
            import torch
            if args.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable. Use the project GPU environment.')
            if key=='medjev':
                from med_jevify.medjev_adapter import MedJevAdapter
                adapter=MedJevAdapter(run=args.medjev,device=args.device,dtype='bf16' if args.device.startswith('cuda') else 'fp32')
            elif key=='head':
                from med_jevify.medgemma_direct import MedGemmaDirectAdapter
                adapter=MedGemmaDirectAdapter(run=args.head,device=args.device,dtype='bf16' if args.device.startswith('cuda') else 'fp32')
            elif key=='nar':
                from tta.adapters.laya_med import LayaMedAdapter
                adapter=LayaMedAdapter(args.nar,device=args.device,manifest=str(ROOT/'data/prepared/manifest.json'),batch_size=1)
            else:
                from med_jevify.upstream import load_jevify
                adapter=load_jevify().Jevify.from_transformers(args.ar_model,device=args.device,dtype='auto',no_think=False)
            def score(sample):
                if key in ('medjev','head'):
                    probs,lat=adapter.score([sample],1)
                    return dict(probabilities=probs[0].tolist(),latency_seconds=float(lat[0]))
                if key=='gen':
                    from med_jevify.evaluate import conventional_generate,timed_call
                    result,lat=timed_call(adapter,conventional_generate,sample)
                    return result|dict(options=sample['options']+['__invalid__'],latency_seconds=lat)
                if key=='ll':
                    from med_jevify.evaluate import native_generate,timed_call
                    result,lat=timed_call(adapter,native_generate,sample)
                    return result|dict(options=sample['options']+['__invalid__'],latency_seconds=lat)
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                start=time.perf_counter();logits=adapter.score_logits([sample],1)[0]
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                return dict(logits=logits.tolist(),temperature=adapter.temperature(),latency_seconds=time.perf_counter()-start)
            # Warm each local model outside recorded measurements.
            score(samples[0])
            with (out/'live.jsonl').open('a',buffering=1) as f:
                for i in pending:
                    sample=samples[i%len(samples)]
                    result=score(sample)
                    record=dict(id=sample['id'],dataset=args.dataset,label=sample['label'],options=sample['options'],model_key=key,step=i,device=args.device)
                    record.update(result)
                    from server import normalize
                    normalize(record,sample)
                    f.write(json.dumps(record,allow_nan=False)+'\n')
                    status(f'{args.dataset} · {key} · {i+1}/{args.samples} measured')
            del adapter;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        status(f'Complete · {args.dataset} · {args.samples} steps · '+', '.join(keys))
    except Exception as exc:
        status(f'Collection stopped: {type(exc).__name__}: {exc}');raise

if __name__=='__main__':main()
