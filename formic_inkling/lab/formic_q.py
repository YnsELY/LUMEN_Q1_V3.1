"""GPU port of the LUMEN Q1-v3.1 E8 quantizer (lumen_ref/e8.py, quantization.py, entropy.py).

Conventions identical to the CPU reference:
  * doubled representation u = 2 z, z in E8 ; decoded weight = u * step / 2
  * shell index m = sum(u^2) / 8 ; mode-0 shell model: rANS(shell) + uniform rank in shell
  * per-row FP16 step  (reference rule: step_i = multiplier * RMS(row_i))
Extension (flagged, NOT in v3.1): optional per-column scale c (FP16, counted in bytes):
  W ~= diag(r) * (U/2) * diag(1/c).  With c = 1 this is exactly the reference parametrisation.
Bits are the *ideal* cost of the mode-0 codec (empirical shell entropy + log2 shell size
+ escapes at 256 bits) plus scale vectors; physical overhead is validated separately on
sub-matrices with the real CPU codec (see tests/test_gpu_port.py).
"""
import math
import torch

LIMIT = 128  # shell limit of the table (max allowed by ShellModel)


def _shell_sizes(limit=LIMIT):
    out = []
    for m in range(limit):
        if m == 0:
            out.append(1); continue
        tot = 0
        for d in range(1, math.isqrt(m) + 1):
            if m % d == 0:
                tot += d ** 3
                if d * d != m: tot += (m // d) ** 3
        out.append(240 * tot)
    return out

_LOG2_SHELL = torch.tensor([math.log2(s) for s in _shell_sizes()], dtype=torch.float64)


def set_limit(limit):
    """V7 codec extension: shell table limit above the v3.1 reference's 128 (needed above ~3.6 bits/weight, where
    shells > 127 would otherwise all be 256-bit escapes). Ideal cost only: the CPU reference codec stops at 128."""
    global LIMIT, _LOG2_SHELL
    LIMIT = int(limit)
    _LOG2_SHELL = torch.tensor([math.log2(s) for s in _shell_sizes(LIMIT)], dtype=torch.float64)


def nearest_d8(v):
    q = torch.round(v)
    odd = (q.sum(-1).to(torch.int64) & 1).bool()
    d = v - q
    j = d.abs().argmax(-1, keepdim=True)
    dj = torch.gather(d, -1, j)
    adj = torch.where(dj >= 0, 1.0, -1.0).to(v.dtype) * odd.unsqueeze(-1).to(v.dtype)
    return q.scatter_add(-1, j, adj)


def e8_nearest(v):
    """v [...,8] float (units of step) -> doubled integer codes (float tensor)."""
    a = 2 * nearest_d8(v)
    b = 2 * nearest_d8(v - 0.5) + 1
    da = ((v - a / 2) ** 2).sum(-1)
    db = ((v - b / 2) ** 2).sum(-1)
    return torch.where((da <= db).unsqueeze(-1), a, b)


def e8_candidates(v):
    """Finite candidate pool of the reference (zero + 3 shrinks x 2 cosets). v [m,8] -> [m,7,8]."""
    pool = [torch.zeros_like(v)]
    for shrink in (1.0, 0.85, 0.70):
        a = v * shrink
        pool.append(2 * nearest_d8(a)); pool.append(2 * nearest_d8(a - 0.5) + 1)
    return torch.stack(pool, 1)


def shell_hist(U, limit=None):
    """U [..., 8] doubled codes -> histogram over shells (index `limit` = escapes)."""
    limit = LIMIT if limit is None else limit
    m = (U.double() ** 2).sum(-1) / 8
    m = m.round().to(torch.int64).clamp_(max=limit).flatten()
    return torch.bincount(m, minlength=limit + 1).double()


def bits_from_hist(h, limit=None):
    """Ideal mode-0 bits for a histogram of shells (own fitted table)."""
    limit = len(h) - 1 if limit is None else limit
    tot = h.sum()
    p = h / tot
    nz = h > 0
    ent = -(h[nz] * torch.log2(p[nz])).sum()
    rank = (h[:limit] * _LOG2_SHELL.to(h.device)[:limit]).sum()
    esc = h[limit] * 256.0
    return float(ent + rank + esc)


def row_steps(W, multiplier):
    """Reference rule: step_i = multiplier * RMS(row_i), rounded to FP16."""
    rms = W.float().pow(2).mean(1).sqrt()
    s = (multiplier * rms).clamp_min(1e-12)
    s = torch.where(rms == 0, torch.ones_like(s), s)
    return s.to(torch.float16).float().clamp_min(6e-8)


def fp16(x):
    return x.to(torch.float16).float()


@torch.no_grad()
def quantize_rtn(W, r, c=None, chunk=4096):
    """No-feedback E8 rounding. W [m,n] (n%8==0); r [m] row steps; c [n] column scales or None.
    Returns (Q fp32 [m,n], hist)."""
    m, n = W.shape
    Q = torch.empty(m, n, dtype=torch.float32, device=W.device)
    hist = torch.zeros(LIMIT + 1, dtype=torch.float64, device=W.device)
    for i in range(0, m, chunk):
        w = W[i:i + chunk].float()
        if c is not None: w = w * c
        w = w / r[i:i + chunk, None]
        u = e8_nearest(w.view(w.shape[0], n // 8, 8))
        hist += shell_hist(u)
        q = (u.view(w.shape[0], n) * 0.5) * r[i:i + chunk, None]
        if c is not None: q = q / c
        Q[i:i + chunk] = q
    return Q, hist


def upper_ldl(A):
    """A = U diag(d) U^T, U unit upper (same as lumen_ref.linalg.upper_ldl). A float64 on GPU."""
    C = torch.linalg.cholesky(A.flip(0, 1))
    s = torch.diagonal(C)
    L = C / s[None, :]
    return L.flip(0, 1), (s * s).flip(0)


def ldl_prepare(H, c=None, damp=0.01):
    """Factor the (column-rescaled, damped) Gram once. Returns dict(U fp32, inv [n/8,8,8], metric [n/8,8,8]).
    Memory-lean: works IN PLACE on H when it is already float64 (the caller must not reuse H); only the
    lower triangle is read (the Gram is symmetric up to rounding)."""
    H = H.double()
    if c is not None:
        cd = c.double()
        H.div_(cd[:, None]); H.div_(cd[None, :])
    H.diagonal().add_(damp * H.diagonal().mean())
    Hf = H.flip(0, 1); del H
    C = torch.linalg.cholesky(Hf); del Hf            # = upper_ldl of the reference, on the flipped matrix
    sdiag = torch.diagonal(C).clone()
    C.div_(sdiag[None, :])
    U64 = C.flip(0, 1); del C
    d64 = (sdiag * sdiag).flip(0)
    n = U64.shape[0]
    idx = torch.arange(0, n, 8, device=U64.device)
    blocks = torch.stack([U64[j:j + 8, j:j + 8] for j in range(0, n, 8)])      # [n/8,8,8]
    inv = torch.linalg.inv(blocks).float()
    dd = d64.view(n // 8, 8)
    metric = ((blocks * dd[:, None, :]) @ blocks.transpose(1, 2)).float()
    U = U64.float()
    dsum = float(d64.sum()); dlog = float(d64.log().mean())
    del U64, d64, blocks
    return dict(U=U, inv=inv, metric=metric, dsum=dsum, dlogmean=dlog)


@torch.no_grad()
def ldl_pivots(H, damp=0.01):
    """LDL pivots d (float64) of the damped Gram, same ordering/convention as ldl_prepare; H is not modified."""
    Hf = H.double().flip(0, 1).clone()
    Hf.diagonal().add_(damp * Hf.diagonal().mean())
    C = torch.linalg.cholesky(Hf); del Hf
    d = (torch.diagonal(C) ** 2).flip(0).clone(); del C
    return d


@torch.no_grad()
def ldlq_rows(Wp, prep, superblock=256, use_candidates=True):
    """Wp [m,n] fp32 already in normalised coordinates (W*c/r). Returns (codes/2 [m,n], hist)."""
    m, n = Wp.shape; dev = Wp.device
    U = prep["U"]; INV = prep["inv"]; MET = prep["metric"]
    E = torch.zeros(m, n, dtype=torch.float32, device=dev)
    Qp = torch.empty(m, n, dtype=torch.float32, device=dev)
    hist = torch.zeros(LIMIT + 1, dtype=torch.float64, device=dev)
    ar = torch.arange(m, device=dev)
    for j0 in range(0, n, superblock):
        j1 = min(n, j0 + superblock)
        Fprev = E[:, :j0] @ U[:j0, j0:j1] if j0 else None
        for j in range(j0, j1, 8):
            s = slice(j, j + 8); b = j // 8
            center = Wp[:, s]
            raw = None
            if Fprev is not None: raw = Fprev[:, j - j0:j - j0 + 8]
            if j > j0:
                t = E[:, j0:j] @ U[j0:j, s]
                raw = t if raw is None else raw + t
            if raw is not None: center = center + raw @ INV[b]
            if use_candidates:
                cand = e8_candidates(center)
                err = center[:, None, :] - 0.5 * cand
                val = torch.einsum('mki,ij,mkj->mk', err, MET[b], err)
                u = cand[ar, val.argmin(1)]
            else:
                u = e8_nearest(center)
            hist += shell_hist(u)
            q = 0.5 * u
            Qp[:, s] = q
            E[:, s] = Wp[:, s] - q
    del E
    return Qp, hist


@torch.no_grad()
def quantize_ldlq(W, H, r, c=None, damp=0.01, superblock=256, use_candidates=True, row_chunk=20480, prep=None):
    """Block-8 LDLQ with E8 codes (reference algebra; rows are independent without output feedback,
    so they are processed in parallel / in row chunks).  W [m,n]; H [n,n] Gram of the inputs in
    ORIGINAL coordinates; r [m] row steps; c [n] column scales or None.  Returns (Q fp32, hist, info)."""
    m, n = W.shape
    if prep is None: prep = ldl_prepare(H, c, damp)
    Q = torch.empty(m, n, dtype=torch.float32, device=W.device)
    hist = torch.zeros(LIMIT + 1, dtype=torch.float64, device=W.device)
    for i in range(0, m, row_chunk):
        Wp = W[i:i + row_chunk].float()
        if c is not None: Wp = Wp * c
        Wp = Wp / r[i:i + row_chunk, None]
        Qp, h = ldlq_rows(Wp, prep, superblock, use_candidates)
        hist += h
        Qp = Qp * r[i:i + row_chunk, None]
        if c is not None: Qp = Qp / c
        Q[i:i + row_chunk] = Qp
        del Wp, Qp
    return Q, hist, {}


def matrix_bits(hist, m, n, has_col_scale=False, header_bits=None):
    """Total ideal bits for one matrix: symbols + FP16 row steps (+ FP16 column scales) + header/table.
    Header = 656 bytes with the reference 129-entry table; the table grows by 4 bytes per shell used beyond 128."""
    if header_bits is None:
        used = int(torch.nonzero(hist[:-1]).max()) + 1 if hist[:-1].any() else 1
        header_bits = 8 * (656 + 4 * max(0, used - 128))
    return bits_from_hist(hist) + 16 * m + (16 * n if has_col_scale else 0) + header_bits
