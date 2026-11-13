"""Bounded LOCAL recovery on actual module inputs/targets, not global 27B QAT."""
from __future__ import annotations
import copy,time
import numpy as np
import torch
from .common import integer,finite

def recover_local(module,x_cal,y_cal,x_select,y_select,*,steps=30,batch_size=32,lr=1e-3,
                  proximal=1e-5,max_seconds=60,seed=0,max_trainable_params=2_000_000):
    steps=integer(steps,'steps',1);batch_size=integer(batch_size,'batch_size',1)
    lr=finite(lr,'lr',1e-12);proximal=finite(proximal,'proximal',0);max_seconds=finite(max_seconds,'max_seconds',1e-6)
    pcount=sum(p.numel() for p in module.parameters())
    if pcount>integer(max_trainable_params,'max_trainable_params',1):raise MemoryError('Local training parameter cap')
    arrays=[np.asarray(a,dtype=np.float32) for a in (x_cal,y_cal,x_select,y_select)]
    if any(a.ndim!=2 or not a.size or not np.isfinite(a).all() for a in arrays):raise ValueError('Invalid recovery arrays')
    if len(arrays[0])!=len(arrays[1]) or len(arrays[2])!=len(arrays[3]):raise ValueError('Unpaired recovery data')
    device=next(module.parameters()).device
    trained=copy.deepcopy(module).float().to(device);trained.eval()
    for p in trained.parameters():p.requires_grad_(True)
    anchor={n:p.detach().clone() for n,p in trained.named_parameters()}
    xc,yc,xv,yv=[torch.from_numpy(a).to(device) for a in arrays]
    opt=torch.optim.AdamW(trained.parameters(),lr=lr,weight_decay=0.)
    rng=np.random.default_rng(seed);trace=[];t0=time.perf_counter()
    def score():
        with torch.no_grad():return float(torch.mean((trained(xv)-yv)**2).item())
    best_score=score();initial=best_score;best=copy.deepcopy(trained.state_dict());accepted_step=0
    for step in range(1,steps+1):
        if time.perf_counter()-t0>=max_seconds:break
        ix=torch.tensor(rng.choice(len(xc),size=min(batch_size,len(xc)),replace=False),device=device)
        pred=trained(xc[ix]);loss=torch.mean((pred-yc[ix])**2)
        loss=loss+proximal*sum(torch.sum((p-anchor[n])**2) for n,p in trained.named_parameters())/max(1,pcount)
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trained.parameters(),1.)
        opt.step();s=score();trace.append({'step':step,'train_objective':float(loss.item()),'select_mse':s})
        if np.isfinite(s) and s<best_score:best_score=s;best=copy.deepcopy(trained.state_dict());accepted_step=step
    trained.load_state_dict(best);trained.to(dtype=next(module.parameters()).dtype).eval()
    return trained,{'scope':'local_module_reconstruction_NOT_global_QAT','steps_executed':len(trace),
                    'selected_step':accepted_step,'initial_select_mse':initial,'best_float32_select_mse':best_score,
                    'export_dtype':str(next(trained.parameters()).dtype),'elapsed_seconds':time.perf_counter()-t0,
                    'trace':trace,'note':'Final cast and re-quantization require a new end-to-end evaluation.'}
