"""Corrected small-matrix E8 LDLQ and monotone finite-candidate refinement.

No global optimum claim: the lattice search is a finite heuristic. Rate units
are bits per E8 VECTOR in the objective, not bits per weight. Physical byte
counts are always computed separately by entropy.encode_matrix.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numbers
from . import e8
from .entropy import ShellModel, encode_matrix, decode_matrix, matrix_codec_metadata
from .linalg import as_finite, checked_spd, upper_ldl, block_ldl


@dataclass
class Quantized:
    weights:np.ndarray
    codes:np.ndarray
    steps:np.ndarray
    model:ShellModel
    artifact:bytes

    @property
    def encoded_bpw(self): return len(self.artifact)*8/self.weights.size


def validate_quantized(initial: Quantized):
    """Artifact is authoritative: reject inconsistent in-memory fields."""
    w=as_finite(initial.weights,2)
    decoded,codes,steps,model=decode_matrix(initial.artifact,max_weights=max(8,w.size))
    if (not np.array_equal(w,decoded) or not np.array_equal(initial.codes,codes) or
            not np.array_equal(initial.steps,steps) or initial.model!=model):
        raise ValueError("Quantized fields do not match the serialized artifact")
    return matrix_codec_metadata(initial.artifact,max_weights=max(8,w.size))


def row_steps(w,multiplier):
    w=as_finite(w,2)
    if not np.isfinite(multiplier) or multiplier<=0:raise ValueError("Multiplier must be positive")
    if not w.size or not w.shape[0] or not w.shape[1]: raise ValueError("Empty weights")
    magnitude=np.max(np.abs(w),axis=1)
    scaled=w/np.where(magnitude==0,1.,magnitude)[:,None]
    with np.errstate(over='ignore'):
        raw=(multiplier*magnitude)*np.sqrt(np.mean(scaled*scaled,axis=1))
    # A zero row uses scale 1 and zero codes; nonzero underflow is rejected.
    raw=np.where(raw==0,1.,raw)
    with np.errstate(over='ignore'):
        s=raw.astype(np.float16).astype(np.float64)
    if np.any(s<=0) or not np.all(np.isfinite(s)):
        raise ValueError("Requested scales cannot be represented as FP16")
    return s


def select_code(center,step,metric,rate_lambda,model,incumbent=None,neighbors=False,preserve_shell=False):
    metric=checked_spd(metric)
    if metric.shape!=(8,8) or not np.isfinite(rate_lambda) or rate_lambda<0:raise ValueError("Invalid RDOQ parameters")
    c=e8.candidates(center,step,incumbent,neighbors)
    if preserve_shell:
        if incumbent is None: raise ValueError("Shell-preserving search needs an incumbent")
        old_shell=e8.shell(incumbent);old_parity=int(incumbent[0])&1
        c=np.asarray([u for u in c if e8.shell(u)==old_shell and (not model.mode or (int(u[0])&1)==old_parity)],dtype=np.int64)
    err=np.asarray(center)[None,:]-.5*step*c
    values=np.einsum('ki,ij,kj->k',err,metric,err)
    values+=rate_lambda*np.array([model.bits(u) for u in c])
    j=int(np.argmin(values))
    return c[j].copy(),float(values[j])


def quantize_ldlq(w,h,step_multiplier=.8,rate_lambda=0.,model=None,output_metric=None,group_size=128):
    """Block-8 input feedback, optional Kronecker output feedback.

    `h` and `output_metric` are the actual surrogate factors in the units of
    the objective. We do not renormalize one factor without its Lagrange
    multiplier. No hidden EM changes the objective inside a sweep.
    """
    w=as_finite(w,2);h=checked_spd(h);m,n=w.shape
    if not m or not n or n%8 or h.shape!=(n,n):raise ValueError("Invalid matrix/Hessian shapes")
    g=np.eye(m) if output_metric is None else checked_spd(output_metric)
    if g.shape!=(m,m):raise ValueError("Invalid output factor")
    steps=row_steps(w,step_multiplier)
    model=ShellModel.gaussian(1/step_multiplier) if model is None else model
    ui,factors=block_ldl(h,8); uo,do=upper_ldl(g)
    q=np.zeros_like(w);error=np.zeros_like(w);codes=np.zeros((m,n//8,8),dtype=np.int64)
    for wave in range(m+n//8-1):
        for i in range(m):
            b=wave-i
            if not 0<=b<n//8:continue
            j=8*b;s=slice(j,j+8);inv,metric=factors[b]
            previous_rows=uo[:i,i]@error[:i,:] if i else np.zeros(n)
            raw=(error[i,:j]+previous_rows[:j])@ui[:j,s]
            center=w[i,s]+previous_rows[s]+raw@inv
            code,_=select_code(center,steps[i],do[i]*metric,rate_lambda,model)
            codes[i,b]=code;q[i,s]=code*steps[i]/2;error[i,s]=w[i,s]-q[i,s]
    # Fit a physical local table AFTER the sweep; objective vs physical rate
    # are separate outputs. Promotion must use final bytes, not the old table.
    final_model=ShellModel.fit(codes.reshape(-1,8),limit=model.limit,mode=model.mode,precision=sum(model.frequencies).bit_length()-1)
    artifact=encode_matrix(codes,steps,final_model,group_size=group_size)
    decoded,dc,ds,dm=decode_matrix(artifact,max_weights=max(w.size,8))
    if not np.array_equal(decoded,q):raise ArithmeticError("Export reconstruction mismatch")
    return Quantized(q,codes,steps,final_model,artifact)


def quadratic_objective(target,q,h,g,rate_lambda,model,codes):
    e=np.asarray(target)-np.asarray(q)
    return float(np.sum(e*(g@e@h))+rate_lambda*model.total_model_bits(np.asarray(codes).reshape(-1,8)))


def refine_quadratic(target,initial:Quantized,h,output_metric=None,rate_lambda=0.,sweeps=2,neighbors=True,preserve_shell=False):
    """Exact conditional objective tests, finite candidate search, frozen table.

    Alternating sweep direction; includes incumbent. Monotonicity applies only
    to THIS calibration surrogate with a FIXED probability table, not KL,
    fresh tables, held-out data or physical length.
    """
    if (not np.isfinite(rate_lambda) or rate_lambda<0 or isinstance(sweeps,(bool,np.bool_)) or
            not isinstance(sweeps,numbers.Integral) or sweeps<1):
        raise ValueError("Invalid refinement penalty or sweeps")
    codec=validate_quantized(initial)
    w=as_finite(target,2);h=checked_spd(h);m,n=w.shape
    g=np.eye(m) if output_metric is None else checked_spd(output_metric)
    if initial.weights.shape!=w.shape or h.shape!=(n,n) or g.shape!=(m,m):
        raise ValueError("Shapes do not agree")
    q=initial.weights.copy();c=initial.codes.copy();model=initial.model
    trace=[quadratic_objective(w,q,h,g,rate_lambda,model,c)]
    accepted=0
    for k in range(sweeps):
        indices=[(i,b) for i in range(m) for b in range(n//8)]
        if k%2:indices.reverse()
        for i,b in indices:
            j=8*b;s=slice(j,j+8);e=w-q
            grad=(g@e@h)[i,s];a=g[i,i]*h[s,s]
            center=q[i,s]+np.linalg.solve(a,grad)
            new,_=select_code(center,initial.steps[i],a,rate_lambda,model,c[i,b],neighbors,preserve_shell)
            d=.5*initial.steps[i]*(new-c[i,b])
            delta=float(d@a@d-2*d@grad)+rate_lambda*(model.bits(new)-model.bits(c[i,b]))
            if delta < -1e-12*max(1.,abs(trace[-1])):
                q[i,s]+=d;c[i,b]=new;accepted+=1
                value=quadratic_objective(w,q,h,g,rate_lambda,model,c)
                if value>trace[-1]+1e-9*max(1.,abs(trace[-1])):
                    raise ArithmeticError("Conditional objective increased")
                trace.append(value)
    fm=model if preserve_shell else ShellModel.fit(c.reshape(-1,8),limit=model.limit,mode=model.mode,precision=sum(model.frequencies).bit_length()-1)
    blob=encode_matrix(c,initial.steps,fm,group_size=codec["group_size"])
    qr,_,_,_=decode_matrix(blob,max_weights=max(w.size,8))
    if preserve_shell and len(blob)!=len(initial.artifact):
        raise ArithmeticError("Shell-preserving export changed physical length")
    return Quantized(qr,c,initial.steps.copy(),fm,blob),{'objective_trace_fixed_table':trace,'accepted_moves':accepted,'table_refitted_after_sweep':not preserve_shell,'shell_preserved':preserve_shell,'group_size':codec['group_size'],'physical_bytes_before':len(initial.artifact),'physical_bytes_after':len(blob)}


def quantize_scalar_ldlq(w,h,step_multiplier=.8,levels=31):
    """Plain scalar LDLQ reference. Not a reproduction of historical S0/GSQ."""
    w=as_finite(w,2);h=checked_spd(h);m,n=w.shape
    if h.shape!=(n,n):raise ValueError("Invalid shape")
    u,_=upper_ldl(h);steps=row_steps(w,step_multiplier);q=np.zeros_like(w)
    for i in range(m):
        for j in range(n):
            center=w[i,j]+(w[i,:j]-q[i,:j])@u[:j,j]
            q[i,j]=np.clip(np.rint(center/steps[i]),-levels,levels)*steps[i]
    return q,steps
