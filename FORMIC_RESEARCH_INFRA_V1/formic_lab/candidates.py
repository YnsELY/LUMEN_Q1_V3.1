"""Reversible examples, NOT a prescribed research architecture.

Plugins may replace these examples. Base weights must never be mutated in-place.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import torch
from torch import nn
from .common import integer,load_plugin

class FactorizedLinear(nn.Module):
    def __init__(self,left,right):super().__init__();self.left=left;self.right=right
    def forward(self,x):return self.right(self.left(x))

def lowrank_linear(original,rank,*,max_elements=2_000_000,randomized=False,seed=0):
    if not isinstance(original,nn.Linear):raise TypeError('Low-rank example requires nn.Linear')
    m,n=original.weight.shape;rank=integer(rank,'rank',1)
    if rank>min(m,n):raise ValueError('Rank exceeds matrix dimensions')
    if m*n>integer(max_elements,'max_elements',1):raise MemoryError('SVD safety cap; inspect resources before changing it')
    dtype=original.weight.dtype;device=original.weight.device
    w=original.weight.detach().float()
    if randomized and rank<min(m,n):
        devices=[] if device.type=='cpu' else [device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed);u,s,v=torch.svd_lowrank(w,q=min(rank+8,min(m,n)),niter=2)
        vh=v.T
    else:u,s,vh=torch.linalg.svd(w,full_matrices=False)
    left=nn.Linear(n,rank,bias=False,device=device,dtype=dtype)
    right=nn.Linear(rank,m,bias=original.bias is not None,device=device,dtype=dtype)
    with torch.no_grad():
        left.weight.copy_(vh[:rank]);right.weight.copy_(u[:,:rank]*s[:rank])
        if original.bias is not None:right.bias.copy_(original.bias)
    return FactorizedLinear(left,right)

def prune_swiglu(original,keep_indices):
    for k in ('gate_proj','up_proj','down_proj'):
        if not isinstance(getattr(original,k,None),nn.Linear):raise TypeError('Explicit SwiGLU layout required')
    g,u,d=original.gate_proj,original.up_proj,original.down_proj
    if g.weight.shape!=u.weight.shape or d.in_features!=u.out_features:raise ValueError('Incompatible coupled channels')
    ids=[integer(i,'channel') for i in keep_indices]
    if not ids or len(set(ids))!=len(ids) or max(ids)>=u.out_features:raise ValueError('Invalid retained channels')
    # Keep the original class/forward and its nonlinearity. Replace only matrices.
    result=copy.deepcopy(original);dev=g.weight.device;dt=g.weight.dtype
    result.gate_proj=nn.Linear(g.in_features,len(ids),g.bias is not None,device=dev,dtype=dt)
    result.up_proj=nn.Linear(u.in_features,len(ids),u.bias is not None,device=dev,dtype=dt)
    result.down_proj=nn.Linear(len(ids),d.out_features,d.bias is not None,device=dev,dtype=dt)
    with torch.no_grad():
        result.gate_proj.weight.copy_(g.weight[ids]);result.up_proj.weight.copy_(u.weight[ids]);result.down_proj.weight.copy_(d.weight[:,ids])
        if g.bias is not None:result.gate_proj.bias.copy_(g.bias[ids])
        if u.bias is not None:result.up_proj.bias.copy_(u.bias[ids])
        if d.bias is not None:result.down_proj.bias.copy_(d.bias)
    if hasattr(result,'intermediate_size'):result.intermediate_size=len(ids)
    return result

def build_candidate(original,spec):
    kind=spec.get('kind','identity')
    if kind=='identity':return original,{'kind':'identity'}
    if kind=='lowrank_linear':
        new=lowrank_linear(original,spec['rank'],max_elements=spec.get('max_elements',2_000_000),randomized=spec.get('randomized',False),seed=spec.get('seed',0))
        return new,{'kind':kind,'rank':spec['rank'],'initialization':'SVD of original; truncated when rank < full'}
    if kind=='prune_swiglu':
        new=prune_swiglu(original,spec['keep_indices'])
        return new,{'kind':kind,'kept_channels':spec['keep_indices'],'initialization':'selected original coupled channels'}
    if kind=='plugin':
        factory,meta=load_plugin(spec['factory']);isolated=copy.deepcopy(original);new=factory(isolated,spec)
        if not isinstance(new,nn.Module):raise TypeError('Candidate plugin must return nn.Module')
        if new is original:raise ValueError('Custom plugin returned the protected original')
        return new,{'kind':kind,'plugin':meta}
    raise ValueError('Unknown candidate kind; add a plugin or modify the registry')

def replace_module(root,path,new):
    if not path:raise ValueError('Root replacement is not supported by this helper')
    parent,_,name=path.rpartition('.')
    root.get_submodule(parent)._modules[name]=new

@contextmanager
def substitutions(model,changes):
    paths=[s['target'] for s in changes]
    if len(paths)!=len(set(paths)):raise ValueError('Duplicate targets')
    for a in paths:
        if any(a!=b and (a.startswith(b+'.') or b.startswith(a+'.')) for b in paths):
            raise ValueError('Nested targets: use one coherent plugin, not ambiguous replacement order')
    backups=[];info=[]
    try:
        for spec in changes:
            path=spec['target'];old=model.get_submodule(path);new,meta=build_candidate(old,spec)
            backups.append((path,old));replace_module(model,path,new);new.eval()
            info.append({**meta,'path':path,'old_params':sum(p.numel() for p in old.parameters()),
                         'new_params':sum(p.numel() for p in new.parameters())})
        yield info
    finally:
        for path,old in reversed(backups):replace_module(model,path,old)
