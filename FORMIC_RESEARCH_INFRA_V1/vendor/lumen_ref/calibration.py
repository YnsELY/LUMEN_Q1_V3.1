"""Asymmetric and gated reconstruction, with explicit objectives and costs.

The CG solver is matrix-free with respect to the Hessian, NOT with respect to
activations. Data are in RAM in this reference. Streaming/batching is a GPU
integration task. No universal N/n threshold is assumed.
"""
from __future__ import annotations
import numpy as np
import numbers
from .linalg import as_finite, checked_spd, pcg
from .quantization import select_code, Quantized, validate_quantized
from .entropy import ShellModel, encode_matrix, decode_matrix


def asymmetric_target(w,x_reference,x_student,ridge:float):
    """Min ||B Xq - W Xfp||_F²/N + ridge ||B-W||_F².

    Exact regularized normal-equation target on the supplied data, not the
    full published GPTAQ algorithm. Biases must be accounted for by caller.
    """
    w=as_finite(w,2);xf=as_finite(x_reference,2);xq=as_finite(x_student,2)
    if xf.shape!=xq.shape or w.shape[1]!=len(xf) or xf.shape[1]==0 or not np.isfinite(ridge) or ridge<=0:
        raise ValueError("Invalid dimensions or ridge")
    n=xq.shape[1];h=xq@xq.T/n+ridge*np.eye(len(xq))
    b=(w@xf)@xq.T/n+ridge*w
    return np.linalg.solve(h,b.T).T,h


def bgcc_row_target(anchor,xq,gate_quantized,target_product,ridge:float):
    """Exact single-row ridge target; keeps the weighted metric for rounding."""
    a=as_finite(anchor,1);x=as_finite(xq,2);g=as_finite(gate_quantized,1);y=as_finite(target_product,1)
    if x.shape!=(len(a),len(g)) or len(y)!=len(g) or len(g)==0 or not np.isfinite(ridge) or ridge<=0:
        raise ValueError("Invalid BGCC inputs")
    f=x*g[None,:];n=len(g)
    h=f@f.T/n+ridge*np.eye(len(a));b=f@y/n+ridge*a
    return np.linalg.solve(h,b),h


def bgcc_output_operator(xq,gates,down,ridge):
    """A(V)=((D.T D (g*(V X)))*g) X.T/N + ridge V.

    This is the exact quadratic operator for a frozen gate and down projection.
    D.T@D is NEVER materialized. Coupling between up rows is retained.
    """
    x=as_finite(xq,2);g=as_finite(gates,2);d=as_finite(down,2)
    if x.shape[1]!=g.shape[1] or d.shape[1]!=g.shape[0] or not np.isfinite(ridge) or ridge<=0 or not x.shape[1]:
        raise ValueError("Invalid BGCC operator dimensions")
    n=x.shape[1]
    def mv(v):
        v=as_finite(v,2)
        if v.shape!=(g.shape[0],x.shape[0]):raise ValueError("Wrong parameter shape")
        tmp=g*(v@x)
        return (g*(d.T@(d@tmp)))@x.T/n+ridge*v
    diagonal=np.sum(d*d,axis=0)[:,None]*((g*g)@(x*x).T)/n+ridge
    return mv,diagonal


def bgcc_output_cg(anchor,xq,gates,down,target_output,ridge=1e-3,rtol=1e-9,maxiter=200):
    """Min ||D(g*(W Xq))-Y||²/N + ridge||W-anchor||², conditional convex problem."""
    anchor=as_finite(anchor,2);x=as_finite(xq,2);g=as_finite(gates,2);d=as_finite(down,2);y=as_finite(target_output,2)
    if anchor.shape!=(g.shape[0],x.shape[0]) or y.shape!=(d.shape[0],x.shape[1]):
        raise ValueError("Incompatible anchor/target")
    mv,diag=bgcc_output_operator(x,g,d,ridge)
    residual=y-d@(g*(anchor@x))
    rhs=(g*(d.T@residual))@x.T/x.shape[1]
    result=pcg(mv,rhs,diag,rtol,maxiter)
    return anchor+result.solution,result


def swiglu_objective(w,x,g,down,target,anchor,ridge):
    r=down@(g*(w@x))-target
    return float(np.sum(r*r)/x.shape[1]+ridge*np.sum((w-anchor)**2))


def refine_swiglu_up(initial:Quantized,xq,gates,down,target_output,ridge=1e-3,rate_lambda=0.,sweeps=1,neighbors=True,preserve_shell=False):
    """8-coordinate exact conditional quadratics on FULL MLP output.

    Gate and down are frozen. No m separate n-by-n Gram matrices are needed.
    The current code is a candidate. Probability table is fixed per sweep set.
    Complexity can still be high: this is not a throughput claim.
    """
    codec=validate_quantized(initial)
    if isinstance(sweeps,(bool,np.bool_)) or not isinstance(sweeps,numbers.Integral):
        raise ValueError("sweeps must be an integer")
    x=as_finite(xq,2);g=as_finite(gates,2);d=as_finite(down,2);y=as_finite(target_output,2)
    q=initial.weights.copy();anchor=q.copy();codes=initial.codes.copy();m,n=q.shape;N=x.shape[1]
    if x.shape[0]!=n or g.shape!=(m,N) or d.shape[1]!=m or y.shape!=(d.shape[0],N) or not np.isfinite(ridge) or not np.isfinite(rate_lambda) or ridge<=0 or rate_lambda<0 or sweeps<1 or N==0:
        raise ValueError("Invalid refinement shapes")
    model=initial.model
    def objective():
        return swiglu_objective(q,x,g,d,y,anchor,ridge)+rate_lambda*model.total_model_bits(codes.reshape(-1,8))
    residual=y-d@(g*(q@x));trace=[objective()];accepted=0
    for sweep in range(sweeps):
        pairs=[(i,b) for i in range(m) for b in range(n//8)]
        if sweep%2:pairs.reverse()
        for i,b in pairs:
            s=slice(8*b,8*b+8);f=x[s,:]*g[i][None,:]
            a=(float(d[:,i]@d[:,i])*(f@f.T)/N)+ridge*np.eye(8)
            grad=f@(d[:,i]@residual)/N-ridge*(q[i,s]-anchor[i,s])
            center=q[i,s]+np.linalg.solve(a,grad)
            new,_=select_code(center,initial.steps[i],a,rate_lambda,model,codes[i,b],neighbors,preserve_shell)
            delta=.5*initial.steps[i]*(new-codes[i,b])
            change=float(delta@a@delta-2*delta@grad)+rate_lambda*(model.bits(new)-model.bits(codes[i,b]))
            if change < -1e-12*max(1.,abs(trace[-1])):
                q[i,s]+=delta;codes[i,b]=new
                residual-=np.outer(d[:,i],g[i]*(delta@x[s,:]))
                value=objective()
                if value>trace[-1]+1e-9*max(1.,abs(trace[-1])):
                    raise ArithmeticError("Full MLP conditional objective increased")
                trace.append(value);accepted+=1
    fm=model if preserve_shell else ShellModel.fit(codes.reshape(-1,8),limit=model.limit,mode=model.mode,precision=sum(model.frequencies).bit_length()-1)
    blob=encode_matrix(codes,initial.steps,fm,group_size=codec["group_size"])
    qr,_,_,_=decode_matrix(blob,max_weights=max(8,m*n))
    if preserve_shell and len(blob)!=len(initial.artifact):
        raise ArithmeticError("Shell-preserving export changed physical length")
    return Quantized(qr,codes,initial.steps.copy(),fm,blob),{'fixed_table_objective_trace':trace,'accepted_moves':accepted,'shell_preserved':preserve_shell,'table_refitted_after_sweep':not preserve_shell,'group_size':codec['group_size'],'physical_bytes_before':len(initial.artifact),'physical_bytes_after':len(blob)}


def fresh_quadratic_target(w,gradient,h,g,trust_radius:float):
    """Our reference completion of the square, not a reproduction of G2PTQ.

    Objective <gradient,Delta> + .5 tr(G Delta H Delta.T).
    Bound the G/H norm of the unconstrained displacement before quantization.
    To round this objective with quantize_ldlq (quadratic coefficient 1), use
    output_metric=0.5*g with the same rate penalty, or g with twice that penalty.
    This primitive alone is NOT a gradient-collection or G2PTQ pipeline.
    """
    w=as_finite(w,2);b=as_finite(gradient,2);h=checked_spd(h);g=checked_spd(g)
    if b.shape!=w.shape or h.shape!=(w.shape[1],w.shape[1]) or g.shape!=(w.shape[0],w.shape[0]) or not np.isfinite(trust_radius) or trust_radius<0:
        raise ValueError("Invalid quadratic/trust region")
    direction=-np.linalg.solve(h,np.linalg.solve(g,b).T).T
    norm=float(np.sqrt(max(0,np.sum(direction*(g@direction@h)))))
    scale=1. if norm==0 else min(1.,trust_radius/norm)
    return w+scale*direction,{'unconstrained_metric_norm':norm,'scale':scale}
