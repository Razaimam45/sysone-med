"""Auditable, coverage-constrained summaries of saved TTA operating points.

Selection is descriptive on exported test evaluations, not validation-set tuning.
No model inference or calibration fitting occurs here.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATASETS = ['pubmedqa','med_qa','medmcqa','vqarad','slake','path_vqa','pmc_vqa']
NAMES = dict(zip(DATASETS,['PubMedQA','MedQA','MedMCQA','VQA-RAD','SLAKE','PathVQA','PMC-VQA']))
SOURCES = {
    'MedGemma-LL':'medgemma_ll_tta',
    'MedGemma-Head':'medgemma_direct_headonly_tta',
    'MedGemma-Adapt':'medgemma_direct_lora_tta',
    'MedGemma-NAR':'laya_med_vl_tta',
    'Qwen-LL':'qwen_ll_tta',
    'Qwen-Head':'qwen_direct_headonly_tta',
    'Qwen-Adapt':'qwen_direct_lora_tta',
}
GROUPS = [('Text Medical QA',DATASETS[:3],'Text Average'),
          ('Medical VQA',DATASETS[3:],'Image Average')]

def read_json(path):
    return json.loads(Path(path).read_text())

def checked_frame(rows, source):
    f=pd.DataFrame(rows).copy()
    if 'dataset' not in f:f['dataset']=f['task']
    f['id']=f.id.astype(str)
    if f.duplicated(['dataset','id']).any() or set(f.dataset)!=set(DATASETS):
        raise ValueError(f'{source}: duplicate IDs or missing datasets')
    return f

def summarize(frame, accepted, correct, datasets):
    """Equal dataset weights; accuracy = accepted-correct mass / accepted mass."""
    f=frame.assign(accepted=np.asarray(accepted,dtype=bool), correct=np.asarray(correct,dtype=bool))
    f=f[f.dataset.isin(datasets)]
    g=f.assign(accepted_correct=f.accepted & f.correct).groupby('dataset').agg(
        n=('accepted','size'), accepted=('accepted','sum'), accepted_correct=('accepted_correct','sum'))
    if set(g.index)!=set(datasets):raise ValueError('Incomplete averaging group')
    coverage=float((g.accepted/g.n).mean())
    accuracy=float((g.accepted_correct/g.n).mean()/coverage) if coverage else np.nan
    return dict(accuracy=100*accuracy,coverage=100*coverage,n=int(g.n.sum()),accepted_n=int(g.accepted.sum()))

def collect(root, floor=70.):
    root=Path(root)
    candidates=[];audit=[];routing={};conformal={};baseline={}
    for method,folder in SOURCES.items():
        route_path=root/'tta_results'/folder/'routing.json'
        conf_path=root/'tta_results'/folder/'conformal.json'
        r=read_json(route_path);c=read_json(conf_path)
        rf=checked_frame(r['results'][0]['rows'],route_path)
        # These ledgers contain every test example, not only accepted examples.
        if len(rf)!=r['test_n']:raise ValueError(f'{method}: incomplete routing ledger')
        routing[method]=rf
        baseline[method]=rf.assign(correct=rf.prediction==rf.label)
        for result in r['results']:
            frame=checked_frame(result['rows'],route_path)
            accepted=frame.score >= result['tau']
            correct=frame.prediction==frame.label
            if not np.isclose(accepted.mean(),result['coverage']):raise ValueError('Routing coverage mismatch')
            if accepted.any() and not np.isclose(correct[accepted].mean(),result['accepted_accuracy']):raise ValueError('Routing accuracy mismatch')
            record=dict(method=method,policy='Routing',parameter=float(result['tau']),source=str(route_path.relative_to(root)))
            record.update(summarize(frame,accepted,correct,DATASETS))
            candidates.append(record)
        conformal[method]=[]
        ref=rf.set_index(['dataset','id'])
        for result in c['results']:
            frame=checked_frame(result['rows'],conf_path)
            cf=frame.set_index(['dataset','id'])
            if set(cf.index)!=set(ref.index) or not np.array_equal(cf.loc[ref.index,'label'],ref.label):
                raise ValueError(f'{method}: routing/conformal cohort or labels differ')
            accepted=frame.set_size==1
            # A singleton is correct exactly when its set contains the true label.
            correct=frame.covered.astype(bool)
            if not np.isclose(accepted.mean(),1-result['abstention_rate']):raise ValueError('Conformal coverage mismatch')
            if accepted.any() and not np.isclose(correct[accepted].mean(),result['decided_accuracy']):raise ValueError('Conformal accuracy mismatch')
            record=dict(method=method,policy='Conformal',parameter=float(result['alpha']),source=str(conf_path.relative_to(root)))
            record.update(summarize(frame,accepted,correct,DATASETS))
            candidates.append(record)
            conformal[method].append((float(result['alpha']),frame))
        mismatch=int((cf.loc[ref.index,'prediction']!=ref.prediction).sum())
        audit.append(dict(method=method,n=len(ref),routing_conformal_prediction_disagreements=mismatch,
                          routing_source=str(route_path.relative_to(root)),conformal_source=str(conf_path.relative_to(root))))
    # Gen has baseline prediction ledgers but no saved conformal evaluation.
    for family,folder in [('MedGemma','medgemma_ar'),('Qwen','qwen_ar')]:
        rows=[json.loads(line) for line in (root/'results'/folder/'details/predictions.jsonl').read_text().splitlines() if line.strip()]
        rows=[r for r in rows if r.get('mode')=='conventional_llm']
        f=checked_frame(rows,folder)
        f['score']=f.confidence
        routing[family+'-Gen']=f

    candidate_frame=pd.DataFrame(candidates)
    selected=[]
    for (method,policy),frame in candidate_frame.groupby(['method','policy'],sort=False):
        eligible=frame[frame.coverage>=floor-1e-9]
        if len(eligible):
            chosen=eligible.sort_values(['accuracy','coverage','parameter'],ascending=[False,False,True]).iloc[0].to_dict()
            chosen['selection_status']='target_met'
        else:
            # Transparent fallback: maximal saved coverage; never call it target-compliant.
            chosen=frame.sort_values(['coverage','accuracy'],ascending=False).iloc[0].to_dict()
            chosen['selection_status']='target_unmet'
        chosen['coverage_floor']=floor
        selected.append(chosen)
    selected=pd.DataFrame(selected)
    records=[]
    views=[(NAMES[d],[d],False) for d in DATASETS]
    views += [('Text Average',DATASETS[:3],True),('Image Average',DATASETS[3:],True),('Overall Average',DATASETS,True)]
    for method in SOURCES:
        for label,ds,is_average in views:
            base=baseline[method]
            records.append(dict(method=method,policy='Base',dataset=label,is_average=is_average,
                                parameter=np.nan,selection_status='baseline',**summarize(base,np.ones(len(base),bool),base.correct,ds)))
            for policy in ['Conformal','Routing']:
                choice=selected[(selected.method==method)&(selected.policy==policy)].iloc[0]
                if policy=='Routing':
                    f=routing[method];accepted=f.score>=choice.parameter;correct=f.prediction==f.label
                else:
                    f=next(f for alpha,f in conformal[method] if alpha==choice.parameter)
                    accepted=f.set_size==1;correct=f.covered
                records.append(dict(method=method,policy=policy,dataset=label,is_average=is_average,
                                    parameter=choice.parameter,selection_status=choice.selection_status,
                                    **summarize(f,accepted,correct,ds)))
    return dict(candidates=candidate_frame,selected=selected,table=pd.DataFrame(records),audit=pd.DataFrame(audit),routing=routing,conformal=conformal)


def table_tex(bundle):
    """One horizontal backbone layout; coverage superscripts; averages colored."""
    palette=[(248,194,187),(250,224,181),(244,236,184),(181,225,181),(139,211,174)]
    def color(value,lo,hi):
        t=.5 if hi==lo else np.clip((value-lo)/(hi-lo),0,1)
        x=t*(len(palette)-1);i=min(int(x),len(palette)-2);f=x-i
        v=[round(palette[i][k]*(1-f)+palette[i+1][k]*f) for k in range(3)]
        return '{rgb,255:red,%d;green,%d;blue,%d}'%tuple(v)
    lines=[r'\begin{tikzpicture}[x=1cm,y=1cm,font=\normalfont\sffamily\fontsize{9}{11}\selectfont]']
    def text(x,y,s,opts=''):
        lines.append(rf'\node[align=center,inner sep=0pt{","+opts if opts else ""}] at ({x:.3f},{y:.3f}) {{{s}}};')
    def rect(left,top,right,bottom,fill):
        lines.append(rf'\fill[fill={fill},rounded corners=2pt] ({left:.3f},{top:.3f}) rectangle ({right:.3f},{bottom:.3f});')
    table=bundle['table'];selected=bundle['selected']
    widths={'Base':1.0,'Conformal':1.6,'Routing':1.4}
    columns=[];dividers=[];cursor=2.55
    for family,forms in [('MedGemma',['LL','Head','Adapt']),('Qwen',['LL','Head','Adapt']),('NAR',['NAR'])]:
        method_family='MedGemma' if family=='NAR' else family
        if family in ('Qwen','NAR'):
            dividers.append((cursor+.2,True));cursor+=.4
        left=cursor
        for j,form in enumerate(forms):
            if j:
                dividers.append((cursor+.1,False));cursor+=.2
            form_left=cursor
            for policy in ['Base','Conformal','Routing']:
                width=widths[policy];x=cursor+width/2
                columns.append((method_family+'-'+form,policy,x,width))
                text(x,-1.17,'Base' if policy=='Base' else '+ '+policy,
                     r'font=\normalfont\sffamily\fontsize{8.5}{10}\selectfont')
                if policy!='Base':
                    choice=selected[(selected.method==method_family+'-'+form)&(selected.policy==policy)].iloc[0]
                    greek=r'\alpha' if policy=='Conformal' else r'\tau'
                    text(x,-1.52,rf'${greek}={choice.parameter:g}$',r'font=\normalfont\sffamily\fontsize{8}{10}\selectfont')
                cursor+=width
            text((form_left+cursor)/2,-.79,rf'\textbf{{{form if family != "NAR" else "MedGemma"}}}')
        rect(left,0,cursor,-.48,'gray!9')
        text((left+cursor)/2,-.24,rf'\textbf{{{family}}}',r'font=\sffamily\bfseries\fontsize{11}{13}\selectfont')
    text(.05,-1.17,r'\textbf{Dataset}','anchor=west')
    text(.05,-.45,r'Accuracy\textsuperscript{Coverage} (\%)',r'anchor=west,font=\sffamily\fontsize{8}{10}\selectfont')
    y=-1.88;row_h=.63
    groups=[[NAMES[d] for d in DATASETS[:3]]+['Text Average'],
            [NAMES[d] for d in DATASETS[3:]]+['Image Average'],['Overall Average']]
    for datasets in groups:
        for dataset in datasets:
            cy=y-row_h/2;average='Average' in dataset
            text(.05,cy,rf'\textbf{{{dataset}}}' if average else dataset,'anchor=west')
            subset=table[table.dataset==dataset]
            if average:
                lines.append(rf'\draw[gray!40,line width=.35pt] (0,{y:.3f}) -- ({cursor:.3f},{y:.3f});')
            for method,policy,x,width in columns:
                r=subset[(subset.method==method)&(subset.policy==policy)].iloc[0]
                family=method.split('-')[0]
                values=subset[subset.method.str.startswith(family)].accuracy
                if average and np.isfinite(r.accuracy):
                    rect(x-width/2+.04,cy+.23,x+width/2-.04,cy-.23,color(r.accuracy,values.min(),values.max()))
                acc='--' if not np.isfinite(r.accuracy) else f'{r.accuracy:.1f}'
                if np.isfinite(r.accuracy) and np.isclose(r.accuracy,subset[subset.method==method].accuracy.max()):
                    acc=rf'\textbf{{{acc}}}'
                if policy!='Base':
                    acc+=rf'\textsuperscript{{\normalfont\sffamily\mdseries\fontsize{{6}}{{7}}\selectfont {r.coverage:.1f}}}'
                text(x,cy,acc,r'text=black,font=\normalfont\sffamily\mdseries\fontsize{9}{11}\selectfont')
            y-=row_h
        y-=.13
    for x,strong in dividers:
        lines.append(rf'\draw[gray!50,line width={.7 if strong else .4}pt] ({x:.3f},{0 if strong else -.59:.3f}) -- ({x:.3f},{y+.13:.3f});')
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)
