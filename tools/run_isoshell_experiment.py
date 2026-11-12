#!/usr/bin/env python3
"""New hypothesis from measured size rejections. Fresh fixed seeds, no Qwen.

All moves preserve each shell (and coset for conditioned coding). Same physical
byte count is tested, not assumed. Selection and confirmation remain distinct.
"""
from pathlib import Path
import sys,json,time,platform,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
from run_experiments import norm,relative_error
from lumen_ref.linalg import silu
from lumen_ref.quantization import quantize_ldlq
from lumen_ref.calibration import refine_swiglu_up

def trial(seed):
 rng=np.random.default_rng(seed);n=128;m=32;out=48
 pre=rng.normal(scale=1/np.sqrt(n),size=(n,n));qpre=quantize_ldlq(pre,np.eye(n),.9,.0001)
 wg=rng.normal(scale=1/np.sqrt(n),size=(m,n));wu=rng.normal(scale=1/np.sqrt(n),size=(m,n));d=rng.normal(scale=1/np.sqrt(m),size=(out,m))
 data={}
 for s,N in [('cal',512),('select',384),('confirm',768)]:
  x0=rng.normal(size=(n,N));data[s]=(norm(silu(pre@x0)),norm(silu(qpre.weights@x0)))
 xf,xq=data['cal'];h=xq@xq.T/xq.shape[1]+.02*np.eye(n)
 qg=quantize_ldlq(wg,h,.85,.0001);base=quantize_ldlq(wu,h,.85,.0001);ceiling=len(base.artifact)
 def error(q,s):
  f,x=data[s];return relative_error(d@(silu(wg@f)*(wu@f)),d@(silu(qg.weights@x)*(q.weights@x)))
 best=base;bestdev=error(base,'select');search=[]
 for penalty in (0.,.0001,.001):
  for mult in (.8,.9,1.,1.15):
   c=quantize_ldlq(wu,h,mult,penalty);v=error(c,'select');search.append({'penalty':penalty,'multiplier':mult,'bytes':len(c.artifact),'select_nmse':v})
   if len(c.artifact)<=ceiling and v<bestdev:best=c;bestdev=v
 gates=silu(qg.weights@xq);y=d@(silu(wg@xf)*(wu@xf))
 refined,trace=refine_swiglu_up(best,xq,gates,d,y,ridge=.01,rate_lambda=0.,sweeps=2,preserve_shell=True)
 if len(refined.artifact)!=len(best.artifact):raise RuntimeError('Byte-neutrality failed')
 canddev=error(refined,'select');accept=canddev<bestdev;chosen=refined if accept else best
 return {'seed':seed,'synthetic':True,'dimensions':[128,32,48],'cal_select_confirm_positions':[512,384,768],
  'control_search':search,'control_bytes':len(best.artifact),'candidate_bytes':len(refined.artifact),
  'byte_difference':len(refined.artifact)-len(best.artifact),'control_select_nmse':bestdev,'candidate_select_nmse':canddev,
  'accepted_on_select':accept,'control_confirm_nmse':error(best,'confirm'),'retained_confirm_nmse':error(chosen,'confirm'),
  'candidate_confirm_nmse_diagnostic_only':error(refined,'confirm'),
  'retained_confirm_gain_pct':100*(1-error(chosen,'confirm')/error(best,'confirm')),
  'accepted_code_moves':trace['accepted_moves'],'fixed_table_cal_before':trace['fixed_table_objective_trace'][0],
  'fixed_table_cal_after':trace['fixed_table_objective_trace'][-1]}
def main():
 t=time.perf_counter();rows=[trial(s) for s in (719,809,907,1009,1103)]
 d={'status':'Synthetic fresh-seed follow-up; not a Qwen gain or a global optimality claim',
  'hypothesis':'Exact physical byte-neutrality for fixed scales, per-vector shells, and the delivered shell-uniform format; then evaluate true output improvement independently',
  'seeds_fixed_before_run':[719,809,907,1009,1103],'numpy':np.__version__,'python':platform.python_version(),'trials':rows,'seconds_cpu':time.perf_counter()-t}
 d['code_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'lumen_ref').glob('*.py'))}
 d['code_sha256']['tools/run_isoshell_experiment.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (ROOT/'results'/'experience_isocoquille.json').write_text(json.dumps(d,indent=2,allow_nan=False))
 for r in rows: print(r['seed'],'moves',r['accepted_code_moves'],'bytes delta',r['byte_difference'],'accepted',r['accepted_on_select'],'confirm',r['retained_confirm_gain_pct'])
if __name__=='__main__':main()
