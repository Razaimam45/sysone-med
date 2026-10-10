#!/usr/bin/env python3
"""Six evidence tables and PNG-only paper figures.
Run from any directory. Every computed result is tied to saved prediction IDs.
"""
from pathlib import Path
from collections import defaultdict, Counter
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures2'
TABLES = ROOT / 'latex/tables'
DS = ['pubmedqa','med_qa','medmcqa','vqarad','slake','path_vqa','pmc_vqa']
DN = ['PubMedQA','MedQA','MedMCQA','VQA-RAD','SLAKE','PathVQA','PMC-VQA']
NAMES = dict(zip(DS, DN))
MODELS = ['google/gemma-3-4b-it','google/medgemma-4b-it','kushalpatil/jevify-gemma4-e4b','laya-med']
MN = ['Gemma','MedGemma','Auxiliary checkpoint','Laya-Med']
COLORS = ['#527CA3','#A76578','#BC893D','#238B80']

# Paper methods for tables (includes pending baselines with dashes)
PAPER_METHODS = [
    ('Gemma (AR)','Gemma (AR)','Gemma (AR)','Gemma (AR)'),
    ('MedGemma (AR)','MedGemma (AR)','MedGemma (AR)','MedGemma (AR)'),
    ('MedGemma (LL)','MedGemma (LL)','MedGemma (LL)','MedGemma (LL)'),
    ('MedGemma-Direct (head-only)','MedGemma-Direct (head-only)','MedGemma-Direct (head-only)','MedGemma-Direct (head-only)'),
    ('MedGemma-Direct* (LoRA)','MedGemma-Direct* (LoRA)','MedGemma-Direct* (LoRA)','MedGemma-Direct* (LoRA)'),
    ('Laya-Med','Laya-Med','Laya-Med','Laya-Med'),
]
INPUTS = ['results/details/predictions.jsonl','models/laya-med/evaluation-best-50.predictions.jsonl',
          'models/laya-med/evaluation-best-50.json','models/laya-med/evaluation-full.json',
          'models/laya-med/evaluation-full.predictions.jsonl','data/prepared/manifest.json',
          'data/prepared/audit.json','data/prepared/test.jsonl','results/results.json']

def readj(p): return json.loads((ROOT/p).read_text())
def readjl(p): return [json.loads(x) for x in (ROOT/p).read_text().splitlines() if x.strip()]
def softmax_row(row, temperature):
    z = np.asarray(row['logits'],float)/temperature
    z -= z.max()
    lp = z - np.log(np.exp(z).sum())
    return dict(row, prediction=int(lp.argmax()), probabilities=np.exp(lp).tolist(), log_probabilities=lp.tolist())

def load_data():
    groups = defaultdict(dict)
    for r in readjl(INPUTS[0]):
        key = r['model'],r['task'],r['mode']
        if r['id'] in groups[key]: raise ValueError(f'Duplicate observation: {key} {r["id"]}')
        groups[key][r['id']] = r
    pilot = readj(INPUTS[2]); full = readj(INPUTS[3])
    for r in readjl(INPUTS[1]):
        for mode,t in [('raw',1),('cal',pilot['temperature'])]:
            key = MODELS[3],r['dataset'],mode
            if r['id'] in groups[key]: raise ValueError('Duplicate Laya-Med observation')
            groups[key][r['id']] = softmax_row(r,t)
    test = {r['id']:r for r in readjl('data/prepared/test.jsonl')}
    for d in DS:
        reference = groups[MODELS[3],d,'raw']
        assert len(reference)==50
        for m in MODELS[:3]:
            for mode in ['raw_jevify','calibrated_jevify','conventional_llm','native_generate']:
                rr=groups[m,d,mode]
                assert {k:r['label'] for k,r in rr.items()} == {k:r['label'] for k,r in reference.items()}, (m,d,mode)
        for k,r in reference.items(): assert test[k]['label']==r['label']
    fullpred=readjl(INPUTS[4])
    assert len(fullpred)==len(test)==full['examples']
    assert {r['id'] for r in fullpred}==set(test)
    assert all(test[r['id']]['label']==r['label'] for r in fullpred)
    manifest_sha=hashlib.sha256((ROOT/'data/prepared/manifest.json').read_bytes()).hexdigest()
    assert pilot['manifest_sha256']==full['manifest_sha256']==manifest_sha
    return groups,pilot,full,test,fullpred

def items(g,m,d,cal=True): return list(g[m,d,('cal' if cal else 'raw') if m==MODELS[3] else ('calibrated_jevify' if cal else 'raw_jevify')].values())
def correct(rows): return np.array([r['prediction']==r['label'] for r in rows],float)
def confidence(rows): return np.array([r.get('confidence',max(r['probabilities'])) for r in rows])
def stats(rows):
    c=confidence(rows); y=correct(rows); loss=[]; bs=[]
    for r in rows:
        p=np.array(r['probabilities']); lab=r['label']
        assert np.isfinite(p).all() and np.isclose(p.sum(),1,atol=1e-6) and (p>=0).all()
        loss.append(-r['log_probabilities'][lab] if 'log_probabilities' in r else -np.log(p[lab]))
        target=np.eye(len(p))[lab];bs.append(np.sum((p-target)**2))
    ece=0
    for i in range(15):
        mask=(c>=i/15)&((c<(i+1)/15) if i<14 else (c<=1))
        if mask.any():ece+=mask.mean()*abs(c[mask].mean()-y[mask].mean())
    return dict(accuracy=y.mean(),nll=np.mean(loss),brier=np.mean(bs),ece=ece)

def risk_curve(rows):
    """Expected prefix risk under uniform random ordering within exact-confidence ties."""
    c=confidence(rows); err=1-correct(rows); order=np.argsort(-c,kind='stable')
    c=c[order];err=err[order];expected=np.empty(len(err));i=0
    while i<len(err):
        j=i+1
        while j<len(err) and c[j]==c[i]: j+=1
        expected[i:j]=err[i:j].mean();i=j
    return np.arange(1,len(err)+1)/len(err),np.cumsum(expected)/np.arange(1,len(err)+1)

# Presentation names describe computation; source IDs remain unchanged for auditability.
METHODS = [
    ('Gemma (AR)',MODELS[0],'conventional_llm',COLORS[0]),
    ('MedGemma (AR)',MODELS[1],'conventional_llm',COLORS[1]),
    ('MedGemma (LL)',MODELS[1],'calibrated_jevify',COLORS[2]),
    ('Laya-Med',MODELS[3],'cal',COLORS[3]),
]
def method_items(g,method,d):
    return list(g[method[1],d,method[2]].values())


def table(name,caption,label,headers,rows,widths=None):
    n=len(headers); spec=widths or ('l'+'r'*(n-1))
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{5pt}',r'\renewcommand{\arraystretch}{1.13}',
           '\\caption{'+caption+'}', '\\label{tab:'+label+'}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+spec+'@{}}',r'\toprule',
           ' & '.join(r'\textcolor{PaperInk}{\textbf{'+h+'}}' for h in headers)+r' \\',r'\midrule']
    for row in rows:
        if isinstance(row,str): lines.append(row);continue
        row=[r'\textcolor{PaperTeal}{\textbf{Laya-Med}}' if str(x)=='Laya-Med' else str(x) for x in row]
        lines.append(' & '.join(row)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table*}','']
    (TABLES/name).write_text('\n'.join(lines))

def make_tables(g,pilot,full,test):
    audit=readj('data/prepared/audit.json');counts=audit['counts']
    rows=[]
    for d in DS:
        ks=sorted({len(r['options']) for r in test.values() if r['dataset']==d})
        k=str(ks[0]) if len(ks)==1 else f'{min(ks)}--{max(ks)}'
        rows.append([NAMES[d],'Text' if d in DS[:3] else 'Text + image',k,*[f'{counts[s][d]:,}' for s in ['train','validation','calibration','test']],50])
    rows += [r'\midrule',['Total','', '',*[f'{sum(counts[s].values()):,}' for s in ['train','validation','calibration','test']],350]]
    table('table1_cohorts.tex',r'Experimental cohorts after preprocessing. $K$ is the observed test-set candidate count. Val. selects checkpoints; Cal. fits temperatures. The pilot contains 50 shared test IDs per dataset. Counts describe the prepared corpus, not necessarily the number consumed in a particular training run.','cohorts',['Dataset','Input','$K$','Train','Val.','Cal.','Test','Pilot'],rows,'llrrrrrr')
    # One numbered table with a baseline taxonomy and a matched outcome matrix.
    taxonomy=[['Gemma (AR)','AR VLM','Autoregressive generation','Yes','Available'],
              ['MedGemma (AR)','AR VLM','Autoregressive generation','Yes','Available'],
              ['MedGemma (LL)','AR VLM','Candidate-label likelihood scoring','No','Available'],
              ['MedGemma-Direct (head-only)','AR VLM','Learned direct candidate scoring','No','Pending'],
              ['MedGemma-Direct* (LoRA)','AR VLM','Learned direct candidate scoring','No','Pending'],
              ['Laya-Med','NAR multimodal','Direct decision modeling','No','Available']]
    table('table2_accuracy.tex',r'Baseline taxonomy and matched decision accuracy. AR/NAR describe the backbone; iterative output decoding is a separate property. LL denotes candidate-label likelihood, not complete answer-sequence likelihood. MedGemma-Direct specifies two learned-head variants of the same AR VLM: head-only (frozen backbone, decision head trained on cached backbone features) and LoRA (rank-8 adapters plus the same head). Neither has a saved evaluation, so both are excluded from numeric comparisons. Gemma is the multimodal 4B checkpoint. Accuracy is measured on 50 shared examples per task, including invalid generated answers as errors.','accuracy',['Method','Backbone','Decision mechanism','Token loop','Evidence'],taxonomy,'lllll')
    path=TABLES/'table2_accuracy.tex';txt=path.read_text();txt=txt.replace(r'\begin{tabular*}',r'\textbf{(a) Baseline taxonomy}\par\smallskip'+'\n'+r'\begin{tabular*}',1)
    detail=[r'\par\medskip\textbf{(b) Accuracy (\%, higher is better)}\par\smallskip',
            r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',r'\toprule',
            r'\textbf{Dataset} & \textbf{Gemma (AR)} & \textbf{MedGemma (AR)} & \textbf{MedGemma (LL)} & \textbf{MedGemma-Direct (head-only)} & \textbf{MedGemma-Direct* (LoRA)} & \textbf{Laya-Med} \\',r'\midrule']
    accuracies=[]
    for d in DS:
        vals=[correct(method_items(g,m,d)).mean()*100 for m in METHODS];accuracies.append(vals)
        # Column order: Gemma, MedGemma, MedGemma (LL), head-only (dash), LoRA (dash), Laya-Med
        row_vals = [f'{v:.1f}' for v in vals[:3]] + ['--', '--', f'{vals[3]:.1f}']
        detail.append(NAMES[d]+' & '+' & '.join(row_vals)+r' \\')
    macro_vals = np.mean(accuracies,axis=0)
    detail.append(r'\midrule')
    detail.append('Macro & '+' & '.join(f'{v:.1f}' for v in macro_vals[:3])+' & -- & -- & '+f'{macro_vals[3]:.1f}'+r' \\')
    detail += [r'\bottomrule',r'\end{tabular*}']
    txt=txt.replace(r'\end{table*}','\n'.join(detail)+'\n'+r'\end{table*}');path.write_text(txt)
    rows=[]
    for name,m in [('MedGemma (LL)',MODELS[1]),('MedGemma-Direct (head-only)','--'),('MedGemma-Direct* (LoRA)','--'),('Laya-Med',MODELS[3])]:
        for label,ds in [('Text',DS[:3]),('Image',DS[3:]),('All',DS)]:
            if m == '--':
                rows.append([name,label,'--','--','--','--']); continue
            rs=[stats(items(g,m,d)) for d in ds];cf=np.mean([confidence(items(g,m,d)).mean() for d in ds]);acc=np.mean([r['accuracy'] for r in rs])
            rows.append([name,label,f'{np.mean([r["nll"] for r in rs]):.3f}',f'{np.mean([r["brier"] for r in rs]):.3f}',f'{100*np.mean([r["ece"] for r in rs]):.1f}',f'{100*(cf-acc):+.1f}'])
    table('table3_probability_quality.tex',r'Probabilistic decision quality by input modality. NLL and multiclass Brier are proper scoring rules; ECE uses 15 equal-width bins. The signed confidence gap is mean confidence minus accuracy (percentage points). Entries macro-average tasks within each subset. Only pipelines with distributions over the same supplied candidate support are compared; generated-text probabilities have a different outcome space. MedGemma-Direct (head-only) and MedGemma-Direct* (LoRA) are learned-head variants that produce the same probability support; neither has a saved evaluation.','probability',['Method','Subset','NLL $\\downarrow$','Brier $\\downarrow$','ECE (\\%)','Signed gap'],rows,'llrrrr')
    rows=[]
    for m,mn in zip(MODELS[:2],MN[:2]):
        for group,ds in [('Text',DS[:3]),('Image',DS[3:])]:
            modes={mode:[r for d in ds for r in g[m,d,mode].values()] for mode in ['conventional_llm','native_generate','raw_jevify']}
            def invalid(r):return r['prediction']<0 or r['prediction']>=len(r['options'])-1
            a=modes['conventional_llm'];b=modes['native_generate'];byid={r['id']:r for r in modes['raw_jevify']}
            valid=[r for r in a if not invalid(r)]
            changed=sum(r['prediction']!=byid[r['id']]['prediction'] for r in b)
            vacc=f'{100*correct(valid).mean():.1f}' if valid else '--'
            rows.append([mn,group,len(a),sum(invalid(r) for r in a),sum(invalid(r) for r in b),vacc,changed])
    table('table4_decoding.tex',r'Decoding and interface diagnostics. G32/G1 denote greedy generation with a 32/1-token limit. Invalid outputs include unparseable or out-of-choice labels and count as errors. G32 valid accuracy (\%) conditions on parseable outputs; -- means none were valid. G1--score disagreements count different predicted labels on the same IDs, revealing differences beyond output length. These diagnostics are separate from candidate-selection accuracy.','decoding',['Backbone','Subset','$N$','Invalid G32','Invalid G1','G32 valid acc.','G1--score diff.'],rows,'llrrrrr')
    rows=[]
    runtime=[('Gemma (AR)',MODELS[0],'conventional_llm','AR VLM'),
             ('MedGemma (AR)',MODELS[1],'conventional_llm','AR VLM'),
             ('MedGemma (1-token)',MODELS[1],'native_generate','AR VLM'),
             ('MedGemma (LL)',MODELS[1],'raw_jevify','AR VLM')]
    for name,m,mode,backbone in runtime:
        rr=[r for d in DS for r in g[m,d,mode].values()];lat=np.array([r['latency_seconds'] for r in rr])
        rows.append([name,backbone,1,f'{np.median(lat)*1000:.1f}',f'{np.quantile(lat,.95)*1000:.1f}',f'{len(lat)/lat.sum():.1f}'])
    # Add pending MedGemma-Direct variants with dashes
    for name in ['MedGemma-Direct (head-only)','MedGemma-Direct* (LoRA)']:
        rows.append([name,'AR VLM','--','--','--','--'])
    rows += [r'\midrule',r'\multicolumn{6}{l}{\textit{Direct model: amortized forward latency and separate pipeline throughput}} \\']
    t=pilot['timing'];rows.append(['Laya-Med','NAR multimodal',8,f'{t["forward_ms_per_example_p50"]:.1f}',f'{t["forward_ms_per_example_p95"]:.1f}',f'{t["pipeline_examples_per_second"]:.1f}'])
    mem=t['peak_cuda_allocated_bytes']/1024**3
    table('table5_runtime.tex',r'Inference efficiency on the matched pilot. AR rows use pooled request-level wall-time quantiles and $N/\sum_i t_i$ throughput. The one-token row controls for output length. Laya-Med uses synchronized FP32 forward time amortized at batch 8 and independently measured pipeline throughput. Its recorded peak CUDA allocation is '+f'{mem:.2f}'+r' GiB; comparable baseline memory traces are unavailable. Loading and fitting are excluded. MedGemma-Direct (head-only) and MedGemma-Direct* (LoRA) are learned-head variants of the same backbone; neither has a saved evaluation, so their latency is not measured here. Timing scope, batching and precision differ, so these are observed costs rather than controlled architectural speedups.','runtime',['Method','Backbone','Batch','p50 (ms)','p95 (ms)','Examples/s'],rows,'llrrrr')
    rows=[]
    for d in DS:
        v=full['calibrated']['per_dataset'][d]
        rows.append([NAMES[d],f'{v["n"]:,}',f'{100*v["accuracy"]:.1f}',f'{v["nll"]:.3f}',f'{v["brier"]:.3f}',f'{100*v["ece"]:.1f}'])
    rows += [r'\midrule',['Macro','--',f'{100*full["calibrated"]["macro_accuracy"]:.1f}',
              *[f'{np.mean([full["calibrated"]["per_dataset"][d][metric] for d in DS])*(100 if metric=="ece" else 1):.{1 if metric=="ece" else 3}f}' for metric in ['nll','brier','ece']]]]
    table('table6_full_evaluation.tex',r'Laya-Med final-pipeline results on the full prepared test cohort, from \texttt{evaluation-full.json}. Accuracy/ECE are percentages; lower NLL, Brier and ECE is better. This independent export is interpreted under the provenance constraints in Sec.~\ref{sec:protocol}; full-cohort baseline results are unavailable.','full',['Dataset','$N$','Accuracy','NLL','Brier','ECE'],rows)
    names=[f'table{i}_{n}.tex' for i,n in enumerate(['cohorts','accuracy','probability_quality','decoding','runtime','full_evaluation'],1)]
    (TABLES/'tables.tex').write_text('% Optional tables-only aggregator; do not also input section.tex.\n'+'\n'.join('\\input{latex/tables/'+n+'}' for n in names)+'\n')
    return names

def style():
    sns.set_theme(style='ticks',context='paper',palette=COLORS,font='DejaVu Sans',rc={
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'axes.linewidth':.6,'lines.linewidth':1.5,'pdf.fonttype':42,'ps.fonttype':42,
        'savefig.pad_inches':.025,'axes.spines.top':False,'axes.spines.right':False,'grid.linewidth':.45,
        'grid.alpha':.35,'figure.facecolor':'white','axes.facecolor':'white'})

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=.025,dpi=450)
    plt.close(fig)

def schematic():
    from matplotlib.patches import Circle, Polygon, Rectangle
    fig,axs=plt.subplots(3,1,figsize=(3.48,3.85),layout='constrained')
    def arrow(ax,a,b,col,rad=0):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=.9,color=col,connectionstyle=f'arc3,rad={rad}',shrinkA=1,shrinkB=1))
    def chip(ax,x,y,w,h,label,col,fs=6.5):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009,rounding_size=.025',facecolor=col,edgecolor='white',alpha=.8,lw=.6))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color='#243747')
    def input_icon(ax,col):
        ax.add_patch(Rectangle((.025,.36),.13,.29,facecolor='#EFF3F6',edgecolor=col,lw=.7))
        ax.plot([.047,.12],[.58,.58],color=col,lw=1);ax.plot([.047,.10],[.53,.53],color=col,lw=1)
        ax.add_patch(Polygon([[.045,.39],[.074,.47],[.092,.43],[.125,.49],[.14,.39]],facecolor=col,alpha=.45))
        ax.text(.09,.26,'Evidence',ha='center',fontsize=6.2)
    for ax in axs:ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax=axs[0];c=COLORS[0];ax.text(.01,.96,'(a) Autoregressive generation',fontsize=8.5,fontweight='bold',va='top');input_icon(ax,c)
    # Decoder stack, serial token sequence, and a feedback arc.
    for dx,dy in [(.016,.025),(.008,.012),(0,0)]:chip(ax,.22+dx,.32+dy,.22,.34,'AR VLM' if dx==0 else '',sns.light_palette(c,5)[3])
    arrow(ax,(.16,.50),(.21,.50),c)
    for i,t in enumerate(['y₁','y₂','y₃']):
        x=.51+i*.15;chip(ax,x,.40,.10,.18,t,sns.light_palette(c,6)[i+2],7)
        arrow(ax,(.45 if i==0 else x-.05,.49),(x-.01,.49),c)
    arrow(ax,(.86,.62),(.39,.66),c,.18);ax.text(.66,.77,'sequential feedback',ha='center',fontsize=6,color=c)
    ax.text(.59,.15,'Token sequence → parsed answer',ha='center',fontsize=6.6)
    ax=axs[1];c=COLORS[2];ax.text(.01,.96,'(b) Candidate-label likelihood',fontsize=8.5,fontweight='bold',va='top');input_icon(ax,c)
    chip(ax,.23,.34,.25,.31,'AR VLM',sns.light_palette(c,5)[2]);arrow(ax,(.16,.50),(.22,.50),c)
    for i,(lab,w) in enumerate([('A',.09),('B',.22),('C',.055)]):
        y=.65-i*.16;arrow(ax,(.49,.49),(.61,y),c);ax.text(.65,y,lab,ha='center',va='center',fontsize=6.6)
        ax.add_patch(Rectangle((.70,y-.043),w,.085,facecolor=c,alpha=.4+i*.13,lw=0))
    ax.text(.8,.12,'Normalize choice scores',ha='center',fontsize=6.5)
    ax.text(.35,.14,'One scoring step',ha='center',fontsize=6.5)
    ax=axs[2];c=COLORS[3];ax.text(.01,.96,'(c) Direct multimodal decisions',fontsize=8.5,fontweight='bold',va='top')
    for i,label in enumerate(['Text + a₁','Text + a₂','Text + aₖ']):chip(ax,.02,.62-i*.18,.22,.14,label,sns.light_palette(c,6)[i+1],6)
    chip(ax,.31,.39,.23,.25,'NAR encoders',sns.light_palette(c,5)[2]);ax.text(.425,.23,'+ shared image',ha='center',fontsize=6)
    for y in [.69,.51,.33]:arrow(ax,(.25,y),(.30,.515),c)
    ax.add_patch(Circle((.66,.515),.075,facecolor=sns.light_palette(c,5)[3],edgecolor=c,lw=.7));ax.text(.66,.515,'fθ',ha='center',va='center',fontsize=9)
    arrow(ax,(.55,.515),(.58,.515),c)
    for i,w in enumerate([.10,.04,.14]):
        y=.68-i*.16;arrow(ax,(.74,.515),(.82,y),c);ax.add_patch(Rectangle((.84,y-.04),w,.08,facecolor=c,alpha=.5+i*.16))
    ax.text(.66,.17,'Shared scorer',ha='center',fontsize=6.4);ax.text(.5,.015,'Parallel candidates · no output-token decoding',ha='center',fontsize=6.8,color=c)
    save(fig,'figure2_inference_paths')

def timeline():
    """Single-row four-stage timeline from autoregressive generation to NAR decisions.

    Conceptual only: no stage encodes a measured quantity, and each card is a reduction of
    the matching panel of figure2. Colours come from latex/paper_style.tex -- PaperInk
    #223F52, PaperTeal #238B80, PaperAccent #EAF3F0, PaperHeader #E7EDF2.
    """
    from matplotlib.patches import Arc, Circle, Ellipse, Polygon, Rectangle
    ink,teal,accent,header='#223F52','#238B80','#EAF3F0','#E7EDF2'
    blue,slate='#527CA3','#8299A8'
    SPEC=[dict(title='Autoregressive\ngeneration',tag='AR',tc=ink,face=accent,ec='#B4CFDC',mc=blue,nc=blue,
               note='Iterative token-by-token\ngeneration'),
          dict(title='Candidate-label\nlikelihood',tag='AR',tc=ink,face='#FFFFFF',ec='#C6DCD7',mc=teal,nc=teal,
               note='Direct candidate scoring\nwithout generation'),
          dict(title='Learned decision\nhead',tag='AR',tc=ink,face=header,ec='#C1CDD7',mc=slate,nc=slate,
               note='Direct decision mechanism\nover AR representations'),
          dict(title='Non-autoregressive\ndecision modeling',tag='NAR',tc=teal,face='#FFFFFF',ec=teal,mc=teal,
               nc=teal,note='Direct prediction over\npredefined candidates')]
    widths=[.225,.222,.222,.245];lefts=[.010]
    for w in widths[:-1]:lefts.append(lefts[-1]+w+.022)
    cents=[l+w/2 for l,w in zip(lefts,widths)]
    fig,ax=plt.subplots(figsize=(7.0,2.45),layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

    def chip(x,y,w,h,label,col,fs=5.2,tc='#243747',a=.8,r=.022,ls='-',fc=None,ec=None):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=.005,rounding_size={r}',facecolor=fc or col,
            edgecolor=ec or col,lw=.65,alpha=a,linestyle=ls,zorder=2))
        if label:ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color=tc,linespacing=1.25,zorder=3)

    def arrow(a,b,col,lw=.85,rad=0,ms=6.5):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=lw,color=col,shrinkA=0,shrinkB=0,
            mutation_scale=ms,connectionstyle=f'arc3,rad={rad}'),zorder=3)

    def evidence(x,y,w,col):
        """Clinical chart icon: record rules above a monotone ECG trace."""
        h=.78*w;ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=col,lw=.7,zorder=2))
        ax.plot([x+.14*w,x+.70*w],[y+.82*h]*2,color=col,lw=.75,zorder=3)
        ax.plot([x+.14*w,x+.52*w],[y+.66*h]*2,color=col,lw=.75,zorder=3)
        px=np.array([.10,.20,.32,.40,.48,.56,.66,.78,.90]);py=np.array([.26,.26,.26,.48,.08,.26,.26,.26,.26])
        ax.plot(x+px*w,y+py*h,color=col,lw=.9,solid_joinstyle='round',zorder=3)

    def stethoscope(cx,cy,s,col):
        """Medical anchor at the origin of the timeline."""
        ax.add_patch(Arc((cx,cy+.05*s),.62*s,.60*s,theta1=190,theta2=350,color=col,lw=1,zorder=3))
        for sgn in (-1,1):
            ax.plot([cx+sgn*.305*s]*2,[cy,cy+.40*s],color=col,lw=1,zorder=3)
            ax.add_patch(Circle((cx+sgn*.305*s,cy+.42*s),.06*s,facecolor=col,edgecolor='none',zorder=3))
        ax.add_patch(Circle((cx,cy-.25*s),.17*s,facecolor=col,alpha=.35,edgecolor=col,lw=.8,zorder=4))

    def branches(cx,cy,s,col):
        """Decision anchor at the end of the timeline."""
        ax.plot([cx,cx],[cy+.06*s,cy-.02*s],color=col,lw=.9,zorder=3)
        for dx in (-.42,0,.42):
            ax.plot([cx,cx+dx*s],[cy-.02*s,cy-.22*s],color=col,lw=.9,zorder=3)
            ax.add_patch(Circle((cx+dx*s,cy-.27*s),.09*s,facecolor=col,alpha=.55,edgecolor='none',zorder=3))
        ax.add_patch(Circle((cx,cy+.10*s),.11*s,facecolor='white',edgecolor=col,lw=.9,zorder=3))

    def card(l,w,s):
        """Panel body, stage title and the speech bubble that names the mechanism."""
        cx=l+w/2
        ax.add_patch(FancyBboxPatch((l,.045),w,.750,boxstyle='round,pad=.006,rounding_size=.025',
            facecolor=s['face'],edgecolor=s['ec'],lw=.8,zorder=1))
        ax.add_patch(Polygon([[cx-.011,.191],[cx+.011,.191],[cx,.220]],closed=True,facecolor='white',
            edgecolor=s['ec'],lw=.7,zorder=2))
        ax.add_patch(FancyBboxPatch((l+.011,.058),w-.022,.135,boxstyle='round,pad=.005,rounding_size=.02',
            facecolor='white',edgecolor=s['ec'],lw=.7,zorder=2))
        ax.text(cx,.125,s['note'],ha='center',va='center',fontsize=5.5,color=s['tc'],linespacing=1.35,zorder=3)
        ax.text(cx,.712,s['title'],ha='center',va='center',fontsize=6.4,color=s['tc'],fontweight='bold',
            linespacing=1.35,zorder=3)

    def motif_ar(l,w,mc):
        """Decoder stack emitting a serial token chain, with a feedback arc closing the loop."""
        evidence(l+.012,.508,.050,mc);arrow((l+.062,.527),(l+.072,.527),mc)
        chip(l+.074,.482,.062,.075,'AR VLM',sns.light_palette(mc,6)[3],5.2)
        for i,t in enumerate(['y₁','y₂','y₃']):
            x=l+.076+i*.040;chip(x,.318,.026,.046,t,sns.light_palette(mc,6)[i+2],5.6)
            if i:arrow((x-.014,.341),(x-.001,.341),mc,ms=5.5)
        arrow((l+.169,.368),(l+.105,.480),mc,rad=.35,ms=6)
        ax.text(l+.150,.428,'feedback',ha='center',fontsize=4.6,color=mc)
        ax.text(l+.128,.283,'token sequence',ha='center',fontsize=4.6,color=mc)

    def motif_ll(l,w,mc):
        """One scoring pass: the backbone scores the allowed labels without decoding them."""
        evidence(l+.012,.508,.050,mc);arrow((l+.062,.527),(l+.072,.527),mc)
        chip(l+.074,.482,.062,.075,'AR VLM',sns.light_palette(mc,6)[3],5.2)
        arrow((l+.105,.480),(l+.105,.408),mc,ms=5.5)
        chip(l+.074,.358,.062,.046,'p(label)',mc,4.2,tc=mc,fc='white',ls='--',a=1)
        for i,(lab,y,a,tc) in enumerate([('A',.470,.30,'#243747'),('B',.390,.85,'white'),('C',.310,.18,'#243747')]):
            arrow((l+.138,.381),(l+.160,y+.021),mc,ms=5)
            chip(l+.162,y,.044,.042,lab,mc,4.8,tc=tc,a=a)
        ax.text(l+.100,.290,'one scoring pass',ha='center',fontsize=4.6,color=mc)

    def motif_head(l,w,mc):
        """Frozen backbone plus a trained decision head, read out as one choice distribution."""
        evidence(l+.012,.510,.046,mc);arrow((l+.060,.531),(l+.070,.531),mc)
        chip(l+.072,.505,.052,.075,'AR VLM',sns.light_palette(mc,6)[2],5.0)
        arrow((l+.126,.542),(l+.134,.542),mc,ms=5.5)
        chip(l+.136,.505,.052,.075,'decision\nhead',sns.light_palette(teal,6)[2],4.4,tc='#173F3A')
        ax.plot([l+.162,l+.162],[.503,.470],color=teal,lw=.7,ls='--',zorder=3)
        chip(l+.136,.434,.052,.034,'LoRA r=8','#FFFFFF',4.2,tc=teal,fc='white',ec=teal,a=1,r=.014)
        x=l+.024
        for sw,col in [(.088,sns.light_palette(teal,7)[1]),(.048,sns.light_palette(teal,7)[4]),(.016,teal)]:
            ax.add_patch(Rectangle((x,.318),sw,.044,facecolor=col,edgecolor='white',lw=.6,zorder=3));x+=sw
        arrow((l+.180,.340),(l+.196,.340),teal,ms=5.5)
        ax.text(l+.188,.372,'argmax',ha='center',fontsize=4.2,color=teal)
        ax.text(l+.024,.288,'P(choice) over candidates',ha='left',fontsize=4.6,color=teal)

    def motif_nar(l,w,mc):
        """Candidate-conditioned encodings read in parallel by one shared scorer."""
        for i,lab in enumerate(['Text + a₁','Text + a₂','Text + aₖ']):
            chip(l+.012,.470-i*.056,.058,.038,lab,sns.light_palette(teal,7)[3+i],4.2,tc='#173F3A')
        chip(l+.012,.302,.058,.034,'+ image',sns.light_palette(teal,7)[1],4.2)
        chip(l+.086,.352,.054,.136,'NAR\nencoders',sns.light_palette(teal,6)[3],4.6,tc='#173F3A')
        for y in (.489,.433,.377):arrow((l+.072,y),(l+.084,.420),mc,ms=5)
        arrow((l+.070,.319),(l+.084,.396),mc,ms=5)
        arrow((l+.140,.420),(l+.150,.420),mc,ms=5)
        ax.add_patch(Ellipse((l+.174,.420),.048,.058,facecolor=teal,edgecolor=teal,lw=.6,alpha=.92,zorder=2))
        ax.text(l+.174,.420,'fθ',ha='center',va='center',fontsize=5.6,color='white',zorder=3)
        for i,(bw,a) in enumerate([(.024,.45),(.010,.30),(.022,.85)]):
            y=.476-i*.056;arrow((l+.200,.420),(l+.206,y+.014),mc,ms=4.5)
            ax.add_patch(Rectangle((l+.208,y-.002),bw,.028,facecolor=teal,alpha=a,lw=0,zorder=3))
        ax.text(l+.174,.362,'shared head',ha='center',fontsize=4.4,color=teal)

    for l,w,s,motif in zip(lefts,widths,SPEC,[motif_ar,motif_ll,motif_head,motif_nar]):
        card(l,w,s);motif(l,w,s['mc'])
    save(fig,'figure7_ar_to_nar_timeline')

    ax.plot([.055,.945],[.905]*2,color='#D6E1E8',lw=1.4,solid_capstyle='round',zorder=1)
    stethoscope(.030,.902,.062,ink);branches(.968,.898,.055,teal)
    for i,(cx,s) in enumerate(zip(cents,SPEC)):
        if i:arrow((cents[i-1]+.024,.905),(cx-.024,.905),s['nc'],lw=1,ms=7)
        ax.add_patch(Circle((cx,.905),.0175,facecolor=s['nc'],edgecolor='white',lw=.9,zorder=4))
        ax.text(cx,.905,str(i+1),ha='center',va='center',fontsize=5.4,color='white',fontweight='bold',zorder=5)
        chip(cx-.018,.806,.036,.032,s['tag'],'#FFFFFF',4.6,tc=s['nc'],ec=s['nc'],fc='white',a=1,r=.014)

def error_overlap(g):
    values=[]
    for d in DS:
        a=g[MODELS[1],d,'raw_jevify'];b=g[MODELS[3],d,'raw']
        joint=Counter((a[k]['prediction']==a[k]['label'],b[k]['prediction']==b[k]['label']) for k in a)
        values.append([100*joint[x]/len(a) for x in [(True,True),(False,True),(True,False),(False,False)]])
    fig,ax=plt.subplots(figsize=(3.48,2.38),layout='constrained');values=np.array(values);left=np.zeros(7)
    cs=['#AAC6D9','#319887','#BB7892','#D9BD91'];labels=['Both correct','Laya-Med only','MedGemma (LL) only','Both wrong']
    for j,(col,label) in enumerate(zip(cs,labels)):
        ax.barh(np.arange(7),values[:,j],left=left,color=col,alpha=.84,height=.76,label=label,edgecolor='white',linewidth=.45)
        for k,v in enumerate(values[:,j]):
            if v>=8:ax.text(left[k]+v/2,k,f'{v:.0f}',ha='center',va='center',fontsize=7)
        left+=values[:,j]
    ax.set(yticks=np.arange(7),yticklabels=DN,xlim=(0,100),ylim=(6.5,-.5),xlabel='Paired examples (%)');ax.set_xticks([0,25,50,75,100]);ax.margins(0)
    ax.legend(loc='lower center',bbox_to_anchor=(.48,1.005),ncol=2,frameon=False,columnspacing=.8,handlelength=1)
    sns.despine(ax=ax,left=True);save(fig,'figure3_error_overlap')
    return values.tolist()

def reliability(g):
    fig,axs=plt.subplots(2,2,figsize=(3.48,2.9),layout='constrained',sharex=True,sharey=True);export={}
    for row,(m,title,col) in enumerate([(MODELS[1],'MedGemma (LL)',COLORS[2]),(MODELS[3],'Laya-Med',COLORS[3])]):
        for j,(ds,modality) in enumerate([(DS[:3],'Text'),(DS[3:],'Image')]):
            ax=axs[row,j];ax.plot([0,1],[0,1],color='#999999',lw=.8,ls=':')
            rr=[r for d in ds for r in items(g,m,d)];c=confidence(rr);y=correct(rr);pts=[]
            for k in range(10):
                mask=(c>=k/10)&((c<(k+1)/10) if k<9 else (c<=1))
                if mask.any():pts.append((c[mask].mean(),y[mask].mean(),int(mask.sum())))
            p=np.asarray(pts);export[title+'/'+modality]=pts
            ax.plot(p[:,0],p[:,1],color=col,lw=1.1,alpha=.7)
            ax.scatter(p[:,0],p[:,1],s=12+np.sqrt(p[:,2])*3,c=p[:,2],cmap=sns.light_palette(col,as_cmap=True),vmin=-20,vmax=p[:,2].max(),alpha=.85,edgecolors=col,linewidth=.6,clip_on=False,zorder=3)
            ax.set(xlim=(0,1),ylim=(0,1));ax.set_title(title+' · '+modality,fontsize=7.4);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.grid(alpha=.18)
    for ax in axs[1]:ax.set_xlabel('Confidence')
    for ax in axs[:,0]:ax.set_ylabel('Accuracy')
    save(fig,'figure4_reliability');return export

def selective(g):
    fig,axs=plt.subplots(1,2,figsize=(3.48,2.08),layout='constrained',sharex=True,sharey=True);export={}
    for ax,ds,title in zip(axs,[DS[:3],DS[3:]],['Text tasks','Image tasks']):
        for method in METHODS:
            mn,m,mode,col=method
            curves=[risk_curve(method_items(g,method,d))[1] for d in ds];y=np.mean(curves,axis=0)*100;x=np.arange(1,51)*2
            # No smoothing or invented uncertainty bands; fill encodes observed risk only.
            sns.lineplot(x=x,y=y,ax=ax,color=col,label=mn,legend=False,alpha=.86,linewidth=1.5,estimator=None)
            export[title+'/'+mn]={'coverage':(x/100).tolist(),'risk':(y/100).tolist(),'aurc':float(np.mean(y)/100)}
        ax.set(title=title,xlim=(2,100),ylim=(0,102),xlabel='Coverage (%)');ax.set_xticks([20,60,100]);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Selective risk (%)');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.14),ncol=2,frameon=False,handlelength=1.3,columnspacing=.8)
    save(fig,'figure5_selective_risk');return export

def latency(g):
    import pandas as pd
    records=[]
    for mode,mn in [('conventional_llm','AR generation'),('native_generate','AR, 1 token'),('raw_jevify','Label likelihood')]:
        for d in DS:
            for r in g[MODELS[1],d,mode].values():records.append({'Dataset':NAMES[d],'Model':mn,'Latency (ms)':1000*r['latency_seconds']})
    df=pd.DataFrame(records);fig,ax=plt.subplots(figsize=(3.48,2.42),layout='constrained')
    sns.boxplot(data=df,x='Latency (ms)',y='Dataset',hue='Model',order=DN,palette=[sns.desaturate(c,.85) for c in COLORS[:3]],
        width=.77,gap=.1,linewidth=.65,fliersize=1,showfliers=True,whis=(10,90),ax=ax,boxprops={'alpha':.76})
    ax.set_xscale('log');ax.set_ylabel('');ax.set_xlabel('MedGemma latency (ms, log scale)');ax.grid(axis='x',alpha=.18)
    ax.set_ylim(6.5,-.5);ax.margins(x=.015)
    ax.legend(loc='lower center',bbox_to_anchor=(.45,1.01),ncol=3,frameon=False,fontsize=6.3,handlelength=1,columnspacing=.6)
    sns.despine(ax=ax,left=True);save(fig,'figure6_latency_distribution')

def main():
    OUT.mkdir(exist_ok=True);TABLES.mkdir(exist_ok=True,parents=True);(ROOT/'latex/artifacts').mkdir(exist_ok=True,parents=True)
    g,pilot,full,test,fullpred=load_data();names=make_tables(g,pilot,full,test)
    style();schematic();timeline();overlap=error_overlap(g);rel=reliability(g);risk=selective(g);latency(g)
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}
    report={'sources_sha256':sources,'matched_pilot_ids':350,'full_evaluation_ids':len(fullpred),
            'pilot_temperature':pilot['temperature'],'full_temperature':full['temperature'],
            'tables':names,'error_overlap_percent':dict(zip(DS,overlap)),
            'reliability_bins':rel,'selective_curves':risk,
            'definitions':{'ece':'15 equal-width bins per task, then equal-task mean',
              'reliability_plot':'10 equal-width bins; equal-size pilot tasks pooled; marker area = 12 + 3 sqrt(n); shade also encodes occupancy within each panel',
              'selective':'Exact confidence ties receive expected errors under random within-tie ordering. Each task ranked separately, then equal-task mean. AURC is mean prefix risk.',
              'runtime':'AR raw per-item pooled latencies. Laya-Med timing comes only from its pilot report, not merged summary rows.'}}
    (ROOT/'latex/artifacts/analysis_data.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Generated six tables and six PNG figures; figure 1 uses supplied architecture PNG.')
    print('Verified 350 paired pilot IDs; 19,586 full-cohort IDs; shared manifest fingerprint.')
if __name__=='__main__':main()
