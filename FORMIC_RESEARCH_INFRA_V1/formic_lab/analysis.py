"""Diagnostics, not causal redundancy or full-model sensitivity claims."""
from __future__ import annotations
import numpy as np
from .common import integer
from .metrics import relative_mse

def redundancy(weights,max_dimension=1024):
    w=np.asarray(weights,dtype=np.float64)
    if w.ndim!=2 or not w.size or not np.isfinite(w).all():raise ValueError('Finite matrix required')
    cap=integer(max_dimension,'max_dimension',1)
    # Never silently downsample and pretend to characterize the full matrix.
    if max(w.shape)>cap:raise MemoryError('SVD diagnostic cap. Select and label a slice or implement a randomized diagnostic.')
    s=np.linalg.svd(w,compute_uv=False);energy=s*s;total=float(energy.sum())
    probs=energy/total if total else np.zeros_like(energy)
    nonzero=probs>0;erank=float(np.exp(-np.sum(probs[nonzero]*np.log(probs[nonzero])))) if total else 0.
    return {'shape':list(w.shape),'singular_values':s.tolist(),'energy_effective_rank':erank,
            'cumulative_energy_fraction':(np.cumsum(energy)/total).tolist() if total else [0.]*len(s),
            'interpretation':'Weight-only descriptive spectrum; does not prove removable capacity or rank of quantization error.'}

def sensitivity(w,q,x,y=None):
    w=np.asarray(w,dtype=np.float64);q=np.asarray(q,dtype=np.float64);x=np.asarray(x,dtype=np.float64)
    if w.shape!=q.shape or w.ndim!=2 or x.ndim!=2 or x.shape[1]!=w.shape[1]:raise ValueError('Incompatible weights/inputs')
    ref=x@w.T if y is None else np.asarray(y,dtype=np.float64)
    return {**relative_mse(ref,x@q.T),'scope':'isolated_linear_projection_not_model_KL',
            'positions':len(x),'error_weight_frobenius':float(np.linalg.norm(w-q))}
