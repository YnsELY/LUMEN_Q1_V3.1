"""E8 geometry and an exact CPU shell rank/unrank reference.

Representation: u = 2*z, z in E8. A valid u has eight coordinates of identical
parity and sum(u) divisible by four. No vector codebook is stored. The dynamic
programming memoization IS memory, and is not advertised as a table-free GPU
implementation. Shells outside the declared limit must use an escape format.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations, product
import math, numbers
import numpy as np


def valid(u) -> bool:
    try:
        seq=list(u)
    except TypeError:
        return False
    if len(seq)!=8:
        return False
    v=[]
    for x in seq:
        if isinstance(x,(bool,np.bool_)): return False
        if isinstance(x,numbers.Integral): v.append(int(x))
        elif isinstance(x,(float,np.floating)) and math.isfinite(float(x)) and float(x).is_integer(): v.append(int(x))
        else: return False
    return all((c&1)==(v[0]&1) for c in v) and sum(v)%4==0


def checked(u) -> tuple[int,...]:
    if not valid(u):
        raise ValueError("Invalid doubled E8 vector")
    return tuple(int(c) for c in u)


def shell(u) -> int:
    u=checked(u)
    return sum(c*c for c in u)//8


def shell_size(m: int) -> int:
    if isinstance(m,(bool,np.bool_)) or not isinstance(m,numbers.Integral) or int(m)<0:
        raise ValueError("Shell index must be a nonnegative integer")
    m=int(m)
    if m==0: return 1
    if m>1_000_000: raise ValueError("Shell enumeration resource limit; use ESC")
    total=0
    for d in range(1,math.isqrt(m)+1):
        if m%d==0:
            total+=d**3
            if d*d!=m: total+=(m//d)**3
    return 240*total


def roots() -> np.ndarray:
    a=[]
    for i,j in combinations(range(8),2):
        for x,y in product((-2,2), repeat=2):
            u=[0]*8; u[i]=x; u[j]=y; a.append(u)
    for t in product((-1,1),repeat=8):
        if sum(t)%4==0: a.append(list(t))
    return np.asarray(a,dtype=np.int64)

ROOTS=roots()


def nearest_d8(v):
    v=np.asarray(v,dtype=np.float64)
    if v.shape!=(8,) or not np.all(np.isfinite(v)) or np.max(np.abs(v))>1e8+1.0:
        raise ValueError("Invalid vector or outside reference integer range")
    q=np.rint(v).astype(np.int64)
    if int(q.sum())&1:
        j=int(np.argmax(np.abs(v-q)))
        q[j]+=1 if v[j]>=q[j] else -1
    return q


def nearest(v):
    """Nearest E8 point to v, returned in the doubled integer representation."""
    v=np.asarray(v,dtype=np.float64)
    if v.shape!=(8,) or not np.all(np.isfinite(v)) or np.max(np.abs(v))>1e8:
        raise ValueError("Invalid vector or outside reference integer range")
    a=2*nearest_d8(v); b=2*nearest_d8(v-.5)+1
    return a if np.sum((v-a/2)**2)<=np.sum((v-b/2)**2) else b


@lru_cache(maxsize=100_000)
def _count(k:int, energy:int, residue:int, parity:int) -> int:
    """Count k coordinates with sum mod 4 = residue and squared norm = energy."""
    if energy<0: return 0
    if k==0: return int(energy==0 and residue%4==0)
    bound=math.isqrt(energy)
    return sum(_count(k-1,energy-c*c,(residue-c)%4,parity)
               for c in range(-bound,bound+1) if (c&1)==parity)


def coset_size(m:int, parity:int) -> int:
    if isinstance(m,(bool,np.bool_)) or not isinstance(m,numbers.Integral) or int(m)<0 or int(m)>255 or isinstance(parity,(bool,np.bool_)) or not isinstance(parity,numbers.Integral) or parity not in (0,1):
        raise ValueError("Enumeration reference supports shells 0..255")
    return _count(8,8*int(m),0,parity)


def rank(u, separate_coset:bool=False) -> tuple[int,int,int]:
    """Return (shell, parity, rank). In uniform mode both cosets share a rank."""
    u=checked(u); m=shell(u); p=u[0]&1
    if m>255:
        raise ValueError("Use an escape for large shells")
    result=0 if separate_coset or p==0 else coset_size(m,0)
    energy=8*m; residue=0
    for i,c in enumerate(u):
        bound=math.isqrt(energy)
        for a in range(-bound,c):
            if (a&1)==p:
                result+=_count(7-i,energy-a*a,(residue-a)%4,p)
        energy-=c*c; residue=(residue-c)%4
    if energy or residue: raise ValueError("Rank failed validity check")
    return m,p,result


def unrank(m:int, r:int, parity:int|None=None) -> tuple[int,...]:
    if (isinstance(m,(bool,np.bool_)) or not isinstance(m,numbers.Integral) or
            isinstance(r,(bool,np.bool_)) or not isinstance(r,numbers.Integral) or m<0 or m>255):
        raise ValueError("Unsupported shell or noninteger rank")
    m=int(m); r=int(r)
    if parity is not None and (isinstance(parity,(bool,np.bool_)) or not isinstance(parity,numbers.Integral)):
        raise ValueError("Parity must be 0 or 1")
    c0=coset_size(m,0)
    if parity is None:
        total=c0+coset_size(m,1)
        if not 0<=r<total: raise ValueError("Rank out of range")
        p=0 if r<c0 else 1
        if p: r-=c0
    else:
        p=parity
        if p not in (0,1) or not 0<=r<coset_size(m,p):
            raise ValueError("Rank out of range for coset")
    out=[]; energy=8*m; residue=0
    for i in range(8):
        bound=math.isqrt(energy)
        for c in range(-bound,bound+1):
            if (c&1)!=p: continue
            count=_count(7-i,energy-c*c,(residue-c)%4,p)
            if r<count:
                out.append(c); energy-=c*c; residue=(residue-c)%4
                break
            r-=count
        else:
            raise ValueError("Invalid enumerative rank")
    return checked(out)


def candidates(center, step:float, incumbent=None, neighbors:bool=False):
    """Finite candidate heuristic, NOT the exact anisotropic lattice argmin."""
    if not np.isfinite(step) or step<=0: raise ValueError("Step must be positive")
    v=np.asarray(center,dtype=np.float64)/step
    if v.shape!=(8,) or not np.all(np.isfinite(v)) or np.max(np.abs(v))>1e8:
        raise ValueError("Candidate center outside the supported reference range")
    pool=[np.zeros(8,dtype=np.int64)]
    for shrink in (1.,.85,.70):
        a=v*shrink
        pool.extend((2*nearest_d8(a), 2*nearest_d8(a-.5)+1))
    if incumbent is not None:
        inc=np.asarray(checked(incumbent),dtype=np.int64); pool.append(inc)
        if neighbors: pool.extend(inc[None,:]+ROOTS)
    if neighbors:
        base=nearest(v); pool.extend(base[None,:]+ROOTS)
    uniq=sorted(set(tuple(int(c) for c in a) for a in pool))
    return np.asarray(uniq,dtype=np.int64)
