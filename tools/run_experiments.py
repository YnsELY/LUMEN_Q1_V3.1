#!/usr/bin/env python3
"""Fresh synthetic comparisons, no additional bytes for the claimant.

Same candidate grid. Full setup/search times reported, not equal compute.
This is a small-matrix experiment, NOT a Qwen performance claim.
"""
from pathlib import Path
import argparse, sys, time, json, platform, hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from lumen_ref.quantization import quantize_ldlq
from lumen_ref.calibration import asymmetric_target
from lumen_ref.linalg import silu
from lumen_ref.entropy import decode_matrix
SEEDS=(2003,2011,2027,2029,2039)
MULTIPLIERS=(.80,.825,.85,.875,.90,.95,1.0)
PENALTIES=(0.,.0001)
def error(y,p):return float(np.sum((y-p)**2)/np.sum(y*y))
def norm(x):return x/np.sqrt(np.mean(x*x,axis=0,keepdims=True)+1e-6)

def trial(seed,outdir):
    start=time.perf_counter();rng=np.random.default_rng(seed);n,m,o=128,32,48
    pre=rng.normal(scale=1/np.sqrt(n),size=(n,n));qpre=quantize_ldlq(pre,np.eye(n),.9,.0001)
    wg=rng.normal(scale=1/np.sqrt(n),size=(m,n));wu=rng.normal(scale=1/np.sqrt(n),size=(m,n));down=rng.normal(scale=1/np.sqrt(m),size=(o,m))
    data={}
    for split,N in (('cal',512),('select',384),('confirm',768)):
        x0=rng.normal(size=(n,N));xf=norm(silu(pre@x0));xq=norm(silu(qpre.weights@x0));data[split]=(xf,xq)
    xf,xq=data['cal'];h=xq@xq.T/512+.02*np.eye(n)
    qgate=quantize_ldlq(wg,h,.85,.0001);q0=quantize_ldlq(wu,h,.85,.0001);cap=len(q0.artifact)
    shared_seconds=time.perf_counter()-start
    t=time.perf_counter();target,_=asymmetric_target(wu,xf,xq,ridge=.01);target_seconds=time.perf_counter()-t
    def loss(weights,split):
        f,x=data[split];y=down@(silu(wg@f)*(wu@f));p=down@(silu(qgate.weights@x)*(weights@x))
        return error(y,p)
    families={};times={}
    for family,w in (('control',wu),('asymmetric',target)):
        candidates=[];t0=time.perf_counter()
        for mult in MULTIPLIERS:
            for penalty in PENALTIES:
                q=quantize_ldlq(w,h,mult,penalty);decoded,_,_,_=decode_matrix(q.artifact)
                if not np.array_equal(decoded,q.weights):raise AssertionError('Artifact mismatch')
                candidates.append({'object':q,'multiplier':mult,'penalty':penalty,'bytes':len(q.artifact),'select_nmse':loss(decoded,'select')})
        families[family]=candidates
        times[family]={'grid_candidates':len(candidates),'grid_search_seconds_cpu':time.perf_counter()-t0,'target_seconds_cpu':target_seconds if family=='asymmetric' else 0.}
        times[family]['total_method_seconds_cpu']=times[family]['grid_search_seconds_cpu']+times[family]['target_seconds_cpu']
    # Freeze choices before reading any confirmation metric.
    selections=[]
    for cap_b in sorted(set(int(round(cap*f)) for f in (.90,.95,1.))):
        available=[q for q in families['control'] if q['bytes']<=cap_b]
        if not available:
            selections.append({'ceiling_bytes':cap_b,'status':'no_control_feasible_in_fixed_grid'});continue
        control=min(available,key=lambda q:(q['select_nmse'],q['bytes']))
        common=[q for q in families['asymmetric'] if q['bytes']<=cap_b]
        smaller=[q for q in families['asymmetric'] if q['bytes']<=control['bytes']]
        raw=min(smaller,key=lambda q:(q['select_nmse'],q['bytes'])) if smaller else None
        accepted=raw is not None and raw['select_nmse']<control['select_nmse'];chosen=raw if accepted else control
        selections.append({'ceiling_bytes':cap_b,'status':'selected','control':control,'candidate':raw,'chosen':chosen,'accepted_on_select_without_extra_bytes':accepted,
                           'common_ceiling_candidate':min(common,key=lambda q:(q['select_nmse'],q['bytes'])) if common else None})
    result=[]
    for row in selections:
        if row['status']!='selected':result.append(row);continue
        control=row['control'];chosen=row['chosen'];base=loss(control['object'].weights,'confirm');final=loss(chosen['object'].weights,'confirm')
        def describe(q):return None if q is None else {k:v for k,v in q.items() if k!='object'}
        entry={k:v for k,v in row.items() if k not in ('control','candidate','chosen','common_ceiling_candidate')}
        entry.update(control=describe(control),candidate_without_extra_bytes=describe(row['candidate']),retained=describe(chosen),common_ceiling_candidate=describe(row['common_ceiling_candidate']),
                     control_confirm_nmse=base,retained_confirm_nmse=final,confirm_relative_change_pct=100*(final/base-1),byte_difference_retained_vs_control=chosen['bytes']-control['bytes'])
        if chosen['bytes']>control['bytes']:raise AssertionError('Claimant received extra bytes')
        result.append(entry)
        if row['ceiling_bytes']==cap:
            for label,item in (('control',control),('retained',chosen)):
                path=outdir/f'matched_{seed}_{label}.lmat';path.write_bytes(item['object'].artifact)
                entry[label+'_artifact']={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    serial={f:[{k:v for k,v in q.items() if k!='object'} for q in seq] for f,seq in families.items()}
    return {'seed':seed,'status':'synthetic_not_Qwen','dimensions':{'n':n,'m':m,'output':o,'cal':512,'select':384,'confirm':768},'base_ceiling_bytes':cap,
            'comparisons':result,'candidate_tables':serial,'timings':times,'shared_setup_seconds_cpu':shared_seconds,'total_seconds_cpu':time.perf_counter()-start}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'matched_budget');p.add_argument('--seeds',default=','.join(map(str,SEEDS)))
    a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True);seeds=tuple(int(s) for s in a.seeds.split(','));rows=[]
    for seed in seeds:
        row=trial(seed,out);rows.append(row);(out/f'trial_{seed}.json').write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf-8')
        best=row['comparisons'][-1];print(seed,best.get('accepted_on_select_without_extra_bytes'),best.get('confirm_relative_change_pct'),best.get('byte_difference_retained_vs_control'),flush=True)
    report={'status':'fresh_synthetic_not_Qwen','seeds_fixed_before_run':list(seeds),'multipliers':list(MULTIPLIERS),'penalties':list(PENALTIES),
            'selection_rule':'candidate <= selected control actual bytes, lower select NMSE; else retain control',
            'note':'Same grid of 14 candidates per method; timings include target solve. Compute not equalized. Historical +3.54% not promoted.',
            'python':platform.python_version(),'numpy':np.__version__,'trials':rows,
            'code_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'lumen_ref').glob('*.py'))}}
    report['code_sha256']['tools/run_experiments.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'experiences_taille_controlee.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
