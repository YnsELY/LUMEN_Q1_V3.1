"""Budget primitives with explicit assumptions and actual byte accounting.

No claim to implement RCO or to measure Qwen sensitivities. The exact sparse
knapsack is for small additive option tables; it deliberately stops rather
than silently approximating a large frontier. Joint loss must be re-evaluated.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math, numbers
import numpy as np


def integer(value, name="value", minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer, not a bool")
    value = int(value)
    if value < minimum:
        raise ValueError(name + " is below its minimum")
    return value


def align(n: int, a: int) -> int:
    n = integer(n, "bytes"); a = integer(a, "alignment", 1)
    return ((n+a-1)//a)*a


def lowrank_cost(m: int, n: int, r: int, bits: int=8,
                 scale_bytes: int=0, metadata_bytes: int=0):
    m=integer(m,"rows",1); n=integer(n,"columns",1); r=integer(r,"rank",1)
    bits=integer(bits,"bits",1); scale_bytes=integer(scale_bytes,"scale_bytes")
    metadata_bytes=integer(metadata_bytes,"metadata_bytes")
    raw=(r*(m+n)*bits+7)//8  # exact integer arithmetic, no float rounding
    total=raw+scale_bytes+metadata_bytes
    return {'raw_bytes':raw, 'total_bytes':total,
            'equivalent_added_bpw':8*total/(m*n)}


def continuous_waterfill(counts, omega, budget_bits, floors=0., ceilings=16.):
    """Continuous convex surrogate only; supplied sensitivities are not measured here.

    A returned point is feasible under the stated float64 dot convention.
    Exact integer-byte feasibility belongs to allocate_options/account_files.
    """
    n=np.asarray(counts,dtype=float); w=np.asarray(omega,dtype=float)
    if n.ndim!=1 or n.size==0 or w.shape!=n.shape:
        raise ValueError("Expected nonempty 1D allocation arrays")
    lo=np.broadcast_to(np.asarray(floors,dtype=float),n.shape)
    hi=np.broadcast_to(np.asarray(ceilings,dtype=float),n.shape)
    if (not all(np.all(np.isfinite(v)) for v in (n,w,lo,hi)) or
            np.any(n<=0) or np.any(w<=0) or np.any(lo<0) or np.any(hi<lo)):
        raise ValueError("Invalid allocation inputs")
    if isinstance(budget_bits,(bool,np.bool_)) or not np.isscalar(budget_bits) or not math.isfinite(budget_bits):
        raise ValueError("Invalid budget")
    budget_bits=float(budget_bits)
    with np.errstate(over='ignore',invalid='ignore'):
        min_b=float(n@lo); max_b=float(n@hi)
    if not math.isfinite(min_b) or not math.isfinite(max_b):
        raise ValueError("Budget products overflow float64")
    if budget_bits<min_b: raise ValueError("Budget below mandatory floor")
    if budget_bits>=max_b: return hi.copy()
    offset=.5*np.log2(w)
    base_lo=float(np.min(lo-offset))-2
    base_hi=float(np.max(hi-offset))+2
    feasible=lo.copy()
    for _ in range(100):
        base=base_lo*.5+base_hi*.5
        r=np.clip(base+offset,lo,hi)
        if float(n@r)<=budget_bits:
            base_lo=base; feasible=r.copy()
        else:
            base_hi=base
    if float(n@feasible)>budget_bits: raise ArithmeticError("Feasibility check failed")
    return feasible


@dataclass(frozen=True)
class Option:
    name:str
    bytes:int
    loss:float


def allocate_options(groups:dict[str,list[Option]],ceiling_bytes:int,fixed_bytes=0,max_states=100000):
    """Exact Pareto DP on ADDITIVE measured/proxy losses and integer byte costs."""
    def _is_int(v): return isinstance(v,numbers.Integral) and not isinstance(v,(bool,np.bool_))
    if not (_is_int(ceiling_bytes) and _is_int(fixed_bytes) and _is_int(max_states)) or int(ceiling_bytes)<int(fixed_bytes) or int(fixed_bytes)<0 or int(max_states)<1:
        raise ValueError("Infeasible fixed budget")
    ceiling_bytes=int(ceiling_bytes); fixed_bytes=int(fixed_bytes); max_states=int(max_states)
    states={fixed_bytes:(0.,{})}
    for gid,opts in groups.items():
        if not isinstance(gid,str) or not gid or not opts or len({o.name for o in opts})!=len(opts) or any(not isinstance(o.name,str) or not o.name or not _is_int(o.bytes) or int(o.bytes)<0 or not math.isfinite(o.loss) for o in opts):
            raise ValueError("Invalid choices")
        nxt={}
        for cost,(loss,chosen) in states.items():
            for o in opts:
                c=cost+int(o.bytes);l=loss+o.loss
                if not math.isfinite(l): raise ValueError("Accumulated loss overflow")
                if c<=ceiling_bytes and (c not in nxt or l<nxt[c][0]):
                    nxt[c]=(l,{**chosen,gid:o.name})
        best=float('inf');states={}
        for c in sorted(nxt):
            if nxt[c][0]<best:
                states[c]=nxt[c];best=nxt[c][0]
        if not states:raise ValueError("No feasible allocation")
        if len(states)>max_states:raise RuntimeError("Frontier limit; use a documented approximation, not silent pruning")
    cost=min(states,key=lambda c:(states[c][0],c));loss,chosen=states[cost]
    return {'bytes':cost,'additive_loss':loss,'choices':chosen,'slack_bytes':ceiling_bytes-cost,'status':'exact_for_supplied_additive_table_only'}


def account_files(root:Path,paths:list[str],ceiling_bytes:int):
    """Count physical relative files once. Does not merge equal hashes for free."""
    ceiling_bytes=integer(ceiling_bytes,'ceiling_bytes')
    root=Path(root).resolve();seen=set();records=[];total=0
    for name in paths:
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file():raise ValueError('Invalid path: '+name)
        if p in seen:raise ValueError('Duplicate package entry: '+name)
        seen.add(p);length=0;digest=hashlib.sha256()
        with p.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):
                length+=len(block);digest.update(block)
        total+=length
        records.append({'path':p.relative_to(root).as_posix(),'bytes':length,'sha256':digest.hexdigest()})
    return {'files':records,'actual_bytes':total,'ceiling_bytes':ceiling_bytes,'feasible':total<=ceiling_bytes,'slack_bytes':ceiling_bytes-total}
