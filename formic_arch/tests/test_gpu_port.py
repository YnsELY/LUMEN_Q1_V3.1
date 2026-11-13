"""Validation of the GPU port lab/formic_q.py against the CPU reference lumen_ref.

Run: python3 -B tests/test_gpu_port.py   (CUDA if available, else CPU; small GPU footprint).
"""
import os, sys, time, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lab'))
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/FORMIC_RESEARCH_INFRA_V1/vendor')

import formic_q as fq
from lumen_ref import e8 as re8
from lumen_ref import quantization as rq
from lumen_ref import linalg as rl
from lumen_ref.entropy import ShellModel, encode_matrix, decode_matrix

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS = {}


def T(x, dt=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dt, device=DEV)


def record(name, ok, msg):
    RESULTS[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)


def valid_codes(u):
    u = u.to(torch.int64)
    par = (u & 1)
    same = (par == par[..., :1]).all(-1)
    s4 = (u.sum(-1) % 4) == 0
    return bool((same & s4).all())


def correlated_X(rng, n, N, rho=0.9):
    """AR(1)-correlated features with per-feature scale variation."""
    X = np.empty((n, N))
    X[0] = rng.standard_normal(N)
    for i in range(1, n):
        X[i] = rho * X[i - 1] + math.sqrt(1 - rho ** 2) * rng.standard_normal(N)
    scale = np.exp(0.5 * rng.standard_normal(n))
    return X * scale[:, None]


def make_H(rng, n, N=None, ridge=1e-3):
    N = N or 4 * n
    X = correlated_X(rng, n, N)
    H = X @ X.T / N
    H = 0.5 * (H + H.T)
    H += ridge * np.mean(np.diag(H)) * np.eye(n)
    return H


def objective(W, Q, H):
    E = np.asarray(W, np.float64) - np.asarray(Q, np.float64)
    return float(np.sum((E @ H) * E))


# ---------------------------------------------------------------- 1
def test_e8_nearest():
    rng = np.random.default_rng(1)
    V = np.concatenate([s * rng.standard_normal((6000, 8)) for s in (0.3, 1, 3, 10)])
    ref = np.array([re8.nearest(v) for v in V])
    u = fq.e8_nearest(T(V, torch.float32))
    ok_valid = valid_codes(u)
    u = u.cpu().numpy().astype(np.int64)
    dref = ((V - ref / 2) ** 2).sum(1)
    dp = ((V - u / 2) ** 2).sum(1)  # distance in float64 of the port's chosen point
    dd = np.abs(dp - dref)
    tol = 1e-5 * (1 + dref)
    same = np.mean(np.all(u == ref, 1))
    # float64 input path too
    u64 = fq.e8_nearest(T(V, torch.float64)).cpu().numpy().astype(np.int64)
    same64 = np.mean(np.all(u64 == ref, 1))
    ok = ok_valid and bool(np.all(dd <= tol))
    record('1 e8_nearest', ok, f"N={len(V)} valid={ok_valid} identical(fp32)={same:.6f} identical(fp64)={same64:.6f} "
           f"max|d2_port-d2_ref|={dd.max():.3e} worse_than_ref={int(np.sum(dp > dref + tol))}")


# ---------------------------------------------------------------- 2
def test_candidates():
    rng = np.random.default_rng(2)
    C = np.concatenate([s * rng.standard_normal((125, 8)) for s in (0.3, 1, 3, 10)])
    P = fq.e8_candidates(T(C, torch.float64)).cpu().numpy().astype(np.int64)
    P32 = fq.e8_candidates(T(C, torch.float32)).cpu().numpy().astype(np.int64)
    bad = bad32 = 0
    for i, c in enumerate(C):
        ref = set(map(tuple, re8.candidates(c, step=1.0)))
        bad += set(map(tuple, P[i])) != ref
        bad32 += set(map(tuple, P32[i])) != ref
    record('2 e8_candidates', bad == 0 and bad32 <= 2,
           f"centers={len(C)} set mismatches fp64={bad} fp32={bad32}")


# ---------------------------------------------------------------- 3
def test_row_steps():
    rng = np.random.default_rng(3)
    mats = [rng.standard_normal((200, 256)),
            rng.standard_t(2, (200, 256)),
            rng.standard_t(1, (200, 256)) * np.exp(1.5 * rng.standard_normal((200, 1))),
            np.vstack([np.zeros((2, 64)), 1e-3 * rng.standard_normal((50, 64))])]
    tot = mism = 0; maxrel = 0.0
    for W in mats:
        for mult in (0.4, 0.8, 1.3):
            r = rq.row_steps(W, mult)
            p = fq.row_steps(T(W, torch.float64), mult).cpu().numpy().astype(np.float64)
            tot += len(r); mism += int(np.sum(r != p))
            maxrel = max(maxrel, float(np.max(np.abs(p - r) / r)))
    # Port computes RMS in fp32 -> may differ by 1 FP16 ulp at rounding boundaries.
    record('3 row_steps', mism <= max(1, tot // 1000) and maxrel < 1e-3,
           f"rows={tot} mismatches={mism} max_rel={maxrel:.2e}")


# ---------------------------------------------------------------- 4
def test_upper_ldl():
    rng = np.random.default_rng(4)
    worst_u = worst_d = 0.0
    for k in range(5):
        A = make_H(rng, 64)
        U, d = rl.upper_ldl(A)
        Up, dp = fq.upper_ldl(T(A, torch.float64))
        Up = Up.cpu().numpy(); dp = dp.cpu().numpy()
        worst_u = max(worst_u, np.abs(Up - U).max() / np.abs(U).max())
        worst_d = max(worst_d, np.max(np.abs(dp - d) / d))
    record('4 upper_ldl', worst_u < 1e-9 and worst_d < 1e-9, f"max rel err U={worst_u:.2e} d={worst_d:.2e}")


# ---------------------------------------------------------------- 5
def test_ldlq_parity():
    rng = np.random.default_rng(5)
    rows = []; ok = True
    for (m, n) in ((6, 32), (16, 64)):
        for dist in ('gauss', 'student_t3'):
            W = rng.standard_normal((m, n)) if dist == 'gauss' else rng.standard_t(3, (m, n))
            W *= 0.02
            H = make_H(rng, n)
            for mult in (0.5, 0.8):
                ref = rq.quantize_ldlq(W, H, step_multiplier=mult)
                Jr = objective(W, ref.weights, H)
                Jrtn = None
                r = fq.row_steps(T(W, torch.float64), mult)
                for sb in (16, 256):
                    Q, hist, _ = fq.quantize_ldlq(T(W), T(H, torch.float64), r=r, c=None, damp=0.0,
                                                  superblock=sb, use_candidates=True)
                    Qn = Q.cpu().numpy().astype(np.float64)
                    finite = bool(np.all(np.isfinite(Qn)))
                    codes = np.rint(2 * Qn / ref.steps[:, None]).astype(np.int64).reshape(m, n // 8, 8)
                    blk = float(np.mean(np.all(codes == ref.codes, -1)))
                    Jp = objective(W, Qn, H)
                    rel = (Jp - Jr) / Jr
                    hist_ok = int(hist.sum().item()) == m * n // 8
                    good = finite and hist_ok and abs(rel) < 0.01 and blk > 0.9 and valid_codes(torch.from_numpy(codes))
                    ok &= good
                    rows.append(f"    m={m:2d} n={n:2d} {dist:10s} mult={mult} sb={sb:3d}: blocks_identical={blk:.4f} "
                                f"J_port/J_ref-1={rel:+.2e} finite={finite} {'ok' if good else 'BAD'}")
    record('5 ldlq parity', ok, "\n" + "\n".join(rows))


# ---------------------------------------------------------------- 6
def test_rate():
    rng = np.random.default_rng(6)
    lines = []; ok = True
    data = []  # (m, n, ideal_bits, physical_bits)
    maxdiff_all = 0.0
    cases = [(m, n, d, mult) for (m, n) in ((64, 256), (128, 512)) for d in ('gauss', 't4') for mult in (0.4, 0.6, 0.9)]
    cases += [(1, 8, 'gauss', 0.6), (2, 64, 'gauss', 0.6), (8, 128, 't4', 0.6)]
    for (m, n, d, mult) in cases:
        W = rng.standard_normal((m, n)) if d == 'gauss' else rng.standard_t(4, (m, n))
        W = (0.02 * W).astype(np.float32)
        Wt = T(W)
        r = fq.row_steps(Wt, mult)
        Q, hist = fq.quantize_rtn(Wt, r)
        rn = r.cpu().numpy().astype(np.float64)
        codes = np.rint(2 * Q.cpu().numpy().astype(np.float64) / rn[:, None]).astype(np.int64).reshape(m, n // 8, 8)
        ideal = fq.bits_from_hist(fq.shell_hist(torch.from_numpy(codes).to(DEV)))
        ideal_h = fq.bits_from_hist(hist)
        mb = fq.matrix_bits(hist, m, n)
        model = ShellModel.fit(codes.reshape(-1, 8), limit=128, mode=0, precision=14)
        blob = encode_matrix(codes, rn, model, group_size=128)
        phys = len(blob) * 8
        dec, dcodes, dsteps, _ = decode_matrix(blob, max_weights=m * n)
        diff = float(np.max(np.abs(dec - Q.cpu().numpy().astype(np.float64))))
        rel_to_step = float(np.max(np.abs(dec - Q.cpu().numpy()) / rn[:, None]))
        maxdiff_all = max(maxdiff_all, rel_to_step)
        rt = np.array_equal(dcodes, codes) and np.array_equal(dsteps, rn) and rel_to_step < 1e-6 and abs(ideal - ideal_h) < 1e-6
        ok &= rt
        data.append((m, n, ideal, phys))
        lines.append(f"    {m:4d}x{n:<4d} {d:5s} mult={mult}: ideal={ideal/(m*n):.4f} bpw  physical={phys/(m*n):.4f} bpw  "
                     f"matrix_bits={mb/(m*n):.4f} bpw  overhead={(phys-ideal)/(m*n):+.4f} bpw ({100*(phys-ideal)/ideal:+.2f}%)  "
                     f"phys-matrix_bits={(phys-mb)/8:+.0f} B  decode max|diff|={diff:.2e} (rel step {rel_to_step:.1e}) roundtrip={'ok' if rt else 'BAD'}")
    # Overhead model: phys = ideal + 8*FIXED + 16*m + prop*(m*n); FIXED = exact container bytes
    FIXED = 24 + 32 + 48 + 4 * 129 + 32 + 4  # matrix hdr, sha256, E8 hdr, freq table (limit 128), sha256, rANS state
    props = [(x[3] - x[2] - 8 * FIXED - 16 * x[0]) / (x[0] * x[1]) for x in data]
    for (m, n, idl, ph), pr in zip(data, props):
        pass
    big_props = [p for x, p in zip(data, props) if x[0] * x[1] >= 16384]
    lines.append(f"    fixed container = {FIXED} B (+2 B/row FP16 steps); port matrix_bits header = {8*(64+32+4*129)//8} B "
                 f"(under-counts by {FIXED-(64+32+4*129)} B/matrix)")
    lines.append("    proportional residual (phys-ideal-fixed-steps)/weight per case: " + " ".join(f"{p:.5f}" for p in props))
    pm = float(np.mean(big_props)); pmax = float(np.max(big_props))
    lines.append(f"    proportional part on >=16k-weight cases: mean {pm:.5f} bpw, max {pmax:.5f} bpw "
                 f"(= {100*pm/ (np.mean([x[2]/(x[0]*x[1]) for x in data if x[0]*x[1]>=16384])):.3f}% of ideal)")
    big = (17408, 5120); Wn = big[0] * big[1]
    for bpw in (2.2, 2.8, 3.4):
        ov = (8 * FIXED + 16 * big[0]) / Wn + pm
        lines.append(f"    extrapolated {big[0]}x{big[1]} @ {bpw} bpw ideal symbols: total overhead {ov:.5f} bpw "
                     f"({100*ov/bpw:.3f}%), of which steps {16*big[0]/Wn:.5f}, fixed {8*FIXED/Wn:.6f}, proportional {pm:.5f}")
    record('6 rate accounting / roundtrip', ok, f"max decode |diff|/step = {maxdiff_all:.2e}\n" + "\n".join(lines))


# ---------------------------------------------------------------- 7
def test_speed():
    g = torch.Generator(device=DEV).manual_seed(7)
    m, n, N = 4096, 2048, 8192
    # correlated X: AR(1) along feature axis
    Z = torch.randn(n, N, device=DEV, generator=g, dtype=torch.float32)
    rho = 0.9
    X = torch.empty_like(Z); X[0] = Z[0]
    # vectorised AR(1) via cumulative filter would be faster; loop is fine for n=2048
    for i in range(1, n):
        X[i] = rho * X[i - 1] + math.sqrt(1 - rho * rho) * Z[i]
    X *= torch.exp(0.5 * torch.randn(n, 1, device=DEV, generator=g))
    H = (X.double() @ X.double().T) / N
    del X, Z
    H += 1e-3 * H.diagonal().mean() * torch.eye(n, device=DEV, dtype=torch.float64)
    W = 0.02 * torch.randn(m, n, device=DEV, generator=g)
    r = fq.row_steps(W, 0.6)

    def J(Q):
        E = (W - Q).double()
        return float(((E @ H) * E).sum())
    Qr, _ = fq.quantize_rtn(W, r); Jr = J(Qr); del Qr
    lines = []; ok = True
    for uc in (True, False):
        fq.quantize_ldlq(W[:64], H, r=r[:64], damp=0.0, superblock=256, use_candidates=uc)  # warmup
        if DEV.type == 'cuda': torch.cuda.synchronize()
        t = time.time()
        Q, hist, _ = fq.quantize_ldlq(W, H, r=r, c=None, damp=0.0, superblock=256, use_candidates=uc)
        if DEV.type == 'cuda': torch.cuda.synchronize()
        dt = time.time() - t
        Jl = J(Q); ratio = Jl / Jr
        ok &= bool(torch.isfinite(Q).all()) and ratio < 0.9
        lines.append(f"    use_candidates={uc}: {dt:.2f} s  J_LDLQ/J_RTN={ratio:.4f}  ideal bpw={fq.bits_from_hist(hist)/(m*n):.4f}")
        del Q
    if DEV.type == 'cuda':
        lines.append(f"    peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    record('7 speed 4096x2048', ok, f"device={DEV}\n" + "\n".join(lines))


if __name__ == '__main__':
    t0 = time.time()
    for f in (test_e8_nearest, test_candidates, test_row_steps, test_upper_ldl, test_ldlq_parity, test_rate, test_speed):
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc()
            record(f.__name__, False, f"EXCEPTION {e!r}")
    print(f"\nSUMMARY ({time.time()-t0:.0f}s): " + ", ".join(f"{k.split()[0]}={'PASS' if v else 'FAIL'}" for k, v in RESULTS.items()))
    assert all(RESULTS.values()), "some checks failed"
