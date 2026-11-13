"""Checked float64 linear algebra for independent numerical references."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numbers
import numpy as np


def as_finite(a, ndim: int | None = None) -> np.ndarray:
    if np.iscomplexobj(a):
        raise ValueError("Real arrays are required")
    try:
        a = np.asarray(a, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Expected a real numeric array") from exc
    if (ndim is not None and a.ndim != ndim) or not np.all(np.isfinite(a)):
        raise ValueError("Wrong dimension or non-finite values")
    return a


def checked_spd(a: np.ndarray) -> np.ndarray:
    a = as_finite(a, 2)
    if not a.shape[0] or a.shape[0] != a.shape[1] or not np.allclose(a, a.T, rtol=1e-11, atol=1e-12):
        raise ValueError("A symmetric square matrix is required")
    a = 0.5 * a + 0.5 * a.T  # avoid overflow in a+a.T
    if not np.all(np.isfinite(a)):
        raise ValueError("Nonfinite symmetric part")
    np.linalg.cholesky(a)  # Fail on non-SPD inputs; no silent clipping of pivots.
    return a


def upper_ldl(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A = U diag(d) U.T, U upper triangular with unit diagonal."""
    a = checked_spd(a)
    c = np.linalg.cholesky(a[::-1, ::-1])
    s = np.diag(c)
    l = c / s[None, :]
    return l[::-1, ::-1].copy(), (s * s)[::-1].copy()


def inverse_unit_upper(u: np.ndarray) -> np.ndarray:
    """Backward substitution; explicitly covers the term k=j missed in v2."""
    u = as_finite(u, 2)
    n = len(u)
    if u.shape != (n,n) or not np.allclose(u, np.triu(u), rtol=0, atol=1e-12) or not np.allclose(np.diag(u), 1, rtol=0, atol=1e-12):
        raise ValueError("Expected unit upper triangular U")
    v = np.eye(n)
    for i in range(n-1,-1,-1):
        for j in range(i+1,n):
            v[i,j] = -np.dot(u[i,i+1:j+1], v[i+1:j+1,j])
    return v


def block_ldl(a: np.ndarray, block: int = 8):
    a=checked_spd(a)
    if isinstance(block, (bool, np.bool_)) or not isinstance(block, numbers.Integral) or block < 1 or len(a) % block:
        raise ValueError("Input width must be divisible by block size")
    u,d=upper_ldl(a)
    factors=[]
    for b in range(0,len(a),block):
        us=u[b:b+block,b:b+block]
        factors.append((inverse_unit_upper(us), (us*d[b:b+block][None,:])@us.T))
    return u,factors


def hadamard(n: int) -> np.ndarray:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or n < 1 or n & (n-1):
        raise ValueError("This reference requires a power-of-two size")
    h=np.ones((1,1))
    while len(h)<n:
        h=np.block([[h,h],[h,-h]])
    return h / np.sqrt(n)


def silu(x):
    """Stable SiLU without clipping the input or changing the activation."""
    x=as_finite(x)
    s=np.empty_like(x)
    pos=x>=0
    s[pos]=1/(1+np.exp(-x[pos]))
    e=np.exp(x[~pos]); s[~pos]=e/(1+e)
    return x*s


def rmsnorm_slice(x_slice, gamma_slice, full_mean_square, eps: float=1e-6):
    """Inputs [n_slice,N], effective gamma (already 1+w if required), full RMS."""
    x=as_finite(x_slice,2); g=as_finite(gamma_slice,1); ms=as_finite(full_mean_square,1)
    if x.shape!=(len(g),len(ms)) or np.any(ms<0) or not np.isfinite(eps) or eps<=0:
        raise ValueError("Invalid shapes, full mean-square or epsilon")
    return g[:,None]*x/np.sqrt(ms[None,:]+eps)


@dataclass
class CGResult:
    solution: np.ndarray
    iterations: int
    relative_residual: float
    converged: bool


def stable_norm(a) -> float:
    """Euclidean norm with scaling; unlike a naive dot, handles tiny RHSs."""
    a = as_finite(a)
    scale = float(np.max(np.abs(a), initial=0.0))
    if scale == 0.0:
        return 0.0
    return scale * float(np.linalg.norm(a / scale))


def pcg(matvec: Callable, rhs: np.ndarray, diagonal=None, rtol=1e-10,
        maxiter=200) -> CGResult:
    """Jacobi-preconditioned CG; convergence is checked on a TRUE residual.

    Normalize by max(abs(rhs)), not norm(rhs): the latter underflows on 1e-200.
    A recursive residual reaching tolerance triggers a true-residual check and
    a restart when needed, not an early unconverged return. Exhausting maxiter
    remains a legitimate nonconvergence result; no arbitrary guarantee for
    ill-conditioned systems. The operator must be real, linear and SPD.
    """
    b = as_finite(rhs)
    if b.size == 0:
        raise ValueError("Empty RHS")
    if (isinstance(maxiter, (bool, np.bool_)) or
            not isinstance(maxiter, numbers.Integral) or maxiter < 1 or
            not np.isscalar(rtol) or not np.isfinite(rtol) or rtol <= 0):
        raise ValueError("Invalid stopping criteria")
    maxiter = int(maxiter)
    diag = np.ones_like(b) if diagonal is None else as_finite(diagonal)
    if diag.shape != b.shape or np.any(diag <= 0):
        raise ValueError("Preconditioner must have RHS shape and be positive")
    scale = float(np.max(np.abs(b)))
    if scale == 0.0:
        return CGResult(np.zeros_like(b), 0, 0.0, True)
    bu = b / scale
    normbu = stable_norm(bu)
    x = np.zeros_like(bu)

    def apply(v):
        out = as_finite(matvec(v))
        if out.shape != b.shape:
            raise ValueError("Operator output has the wrong shape")
        return out

    def dot(a, c):
        # float64 BLAS values; wider accumulation helps near-zero products.
        value = np.sum(a.astype(np.longdouble)*c.astype(np.longdouble))
        if not np.isfinite(value):
            raise ArithmeticError("Nonfinite CG scalar product")
        return value

    def finish(k):
        with np.errstate(over='ignore', under='ignore'):
            solution = x * scale
        if not np.all(np.isfinite(solution)):
            raise ArithmeticError("Solution not representable in float64")
        # Check the rescaled, delivered solution, not only the internal one.
        rel = stable_norm(bu - apply(solution) / scale) / normbu
        return CGResult(solution, k, float(rel), bool(rel <= rtol))

    r = bu.copy()
    z = as_finite(r / diag)
    p = z.copy()
    rz = dot(r, z)
    for k in range(1, maxiter + 1):
        ap = apply(p)
        denom = dot(p, ap)
        if denom <= 0:
            result = finish(k - 1)
            if result.converged:
                return result
            raise ValueError("Operator is not numerically SPD on the CG direction")
        alpha = float(rz / denom)
        if not np.isfinite(alpha):
            raise ArithmeticError("Nonfinite CG step")
        x = as_finite(x + alpha * p)
        r = as_finite(r - alpha * ap)
        if stable_norm(r) / normbu <= rtol:
            true_r = bu - apply(x)
            if stable_norm(true_r) / normbu <= rtol:
                result = finish(k)
                if result.converged:
                    return result
            # Reliable update: continue with the true residual, budget retained.
            r = true_r
            z = as_finite(r / diag)
            p = z.copy()
            rz = dot(r, z)
            continue
        z = as_finite(r / diag)
        new_rz = dot(r, z)
        if rz <= 0 or new_rz <= 0:
            result = finish(k)
            if result.converged:
                return result
            raise ArithmeticError("CG numerical breakdown before convergence")
        beta = float(new_rz / rz)
        p = as_finite(z + beta * p)
        rz = new_rz
    return finish(maxiter)
