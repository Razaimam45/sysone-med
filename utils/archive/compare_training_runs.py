#!/usr/bin/env python3
"""Export epoch histories from local checkpoints without evaluating on test data."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]

def read_run(path):
    # last.pt contains every completed epoch; best.pt only contains its prefix.
    checkpoint=path/'last.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        data={k:saved[k] for k in ['args','history','manifest_sha256','parameters','epoch']}
        del saved
    else:
        checkpoint=path/'metadata.json'
        data=json.loads(checkpoint.read_text())
    history=data['history']
    if not history or len({r['epoch'] for r in history})!=len(history):
        raise ValueError(f'Missing or duplicate epoch history: {path}')
    data['source']=str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    return data

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=2,default=['models/laya-med','models/laya-med-vision-lora'])
    parser.add_argument('--output-dir',default='results/training-comparison')
    args=parser.parse_args();paths=[(ROOT/p).resolve() for p in args.runs]
    runs={p.name:read_run(p) for p in paths};names=list(runs)
    if len(runs)!=2:raise ValueError('Run directory names must differ')
    if len({r['manifest_sha256'] for r in runs.values()})!=1:raise ValueError('Runs use different data manifests')
    out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    summary=[];datasets=[]
    for name,run in runs.items():
        for e in run['history']:
            per=e['validation']['per_dataset'];v=e['validation']
            summary.append(dict(run=name,epoch=e['epoch'],train_loss=e['train_loss'],
                validation_macro_accuracy=v['macro_accuracy'],validation_macro_nll=v['macro_nll'],
                validation_macro_brier=float(np.mean([r['brier'] for r in per.values()])),
                validation_macro_ece=float(np.mean([r['ece'] for r in per.values()]))))
            for dataset,metric in per.items():datasets.append(dict(run=name,epoch=e['epoch'],dataset=dataset,**metric))
    write_csv(out/'epoch_metrics.csv',summary);write_csv(out/'per_dataset_metrics.csv',datasets)
    byrun={name:{r['epoch']:r for r in summary if r['run']==name} for name in names}
    wide=[]
    for epoch in sorted(set().union(*(set(r) for r in byrun.values()))):
        row={'epoch':epoch}
        for name in names:
            e=byrun[name].get(epoch)
            for key in ['train_loss','validation_macro_accuracy','validation_macro_nll']:
                row[name+'/'+key]=e[key] if e else ''
        a,b=[byrun[n].get(epoch) for n in names]
        row['vision_lora_minus_baseline_accuracy_pp']=100*(b['validation_macro_accuracy']-a['validation_macro_accuracy']) if a and b else ''
        wide.append(row)
    write_csv(out/'epoch_comparison.csv',wide)
    a,b=[r['args'] for r in runs.values()]
    differences={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
    export={'exported_at_utc':datetime.now(timezone.utc).isoformat(),'runs':runs,'configuration_differences':differences,
            'note':'Completed-epoch training and validation metrics, before the final held-out probability fit. No test metrics or epoch durations are logged.'}
    (out/'histories.json').write_text(json.dumps(export,indent=2)+'\n')
    sns.set_theme(style='ticks',context='paper',palette=['#527CA3','#238B80'])
    fig,axs=plt.subplots(1,3,figsize=(8.4,2.5),layout='constrained')
    for ax,(metric,label,scale) in zip(axs,[('train_loss','Training loss',1),('validation_macro_accuracy','Validation accuracy (%)',100),('validation_macro_nll','Validation NLL',1)]):
        for name in names:
            rr=byrun[name];epochs=sorted(rr)
            ax.plot(epochs,[rr[e][metric]*scale for e in epochs],marker='o',ms=3,lw=1.4,alpha=.85,label=name)
        ax.set_xlabel('Completed epoch');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True));sns.despine(ax=ax)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.13),ncol=2,frameon=False)
    fig.savefig(out/'epoch_comparison.png',dpi=300,bbox_inches='tight',pad_inches=.03);plt.close(fig)
    lines=['# Epoch comparison','',f'Snapshot: {export["exported_at_utc"]}','','Histories are read from last.pt without modifying training checkpoints. Re-run to refresh after another completed epoch.','',
           '| Run | Completed epochs | Best validation epoch | Best macro accuracy | Effective batch |','|---|---:|---:|---:|---:|']
    for name,r in runs.items():
        best=max(r['history'],key=lambda e:e['validation']['macro_accuracy']);eff=r['args']['batch_size']*r['args']['grad_accumulation']
        lines.append(f'| {name} | {len(r["history"])} | {best["epoch"]} | {100*best["validation"]["macro_accuracy"]:.2f}% | {eff} |')
    lines+=['','## Configuration differences','', '```json',json.dumps(differences,indent=2),'```','',
            'Same manifest; differing batch sizes confound attribution to visual adaptation. The baseline was resumed, so the final configuration alone does not reconstruct its full scheduling history.',
            '', 'CSV accuracy/ECE values are fractions. Plots use accuracy percentages. Missing future epochs remain blank; no interpolation is applied. These are validation comparisons, not test-set or final probability-mapping results.',
            '', 'Refresh from the project root:', '', '```bash','python scripts/compare_training_runs.py','```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:10]));print('Exported:',out)

if __name__=='__main__':main()
