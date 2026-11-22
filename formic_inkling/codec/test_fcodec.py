"""Tests + measurements for fcodec (CPU only). Run:  CUDA_VISIBLE_DEVICES="" python test_fcodec.py
Prints a summary; the full output is also written to test_fcodec.log next to this file."""
import os, sys, json, time, shutil, tempfile, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_arch/lab")
import fcodec as fc, formic_q as fq, package as pk

torch.set_num_threads(4)
LOG = open(os.path.join(HERE, "test_fcodec.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

fq.set_limit(16384)       # ideal cost with a 16384-shell table (no 256-bit escapes) for every case
ROWS = []


def codes(W, mult):
    r = fq.row_steps(W, mult); Q, h = fq.quantize_rtn(W, r)
    U = torch.round(2 * Q / r[:, None]).to(torch.int32).numpy()
    return U, r.half().numpy(), Q.numpy(), fq.bits_from_hist(h) + 16 * W.shape[0]


def calibrate(W, target, lo=0.01, hi=4.0):
    Ws = W[: min(256, W.shape[0])]
    for _ in range(18):
        mid = (lo * hi) ** 0.5
        _, h = fq.quantize_rtn(Ws, fq.row_steps(Ws, mid))
        b = fq.bits_from_hist(h) / Ws.numel()
        lo, hi = (lo, mid) if b < target else (mid, hi)
    return (lo * hi) ** 0.5


def check(tag, U, steps, Q=None, ideal=None, extra=None, timing=True):
    t0 = time.perf_counter(); blob = fc.encode_matrix(U, steps, extra); te = time.perf_counter() - t0
    t0 = time.perf_counter(); U2, s2, ex2 = fc.decode_matrix(blob); td = time.perf_counter() - t0
    Uf = U.reshape(U.shape[0], -1)
    assert U2.shape == Uf.shape and np.array_equal(U2, Uf), tag + ": U mismatch"
    assert np.array_equal(s2.view(np.uint16), steps.view(np.uint16)), tag + ": steps mismatch"
    if extra:
        for k, v in extra.items():
            assert np.array_equal(np.asarray(ex2[k]), np.asarray(v)) and np.asarray(ex2[k]).dtype == np.asarray(v).dtype, k
    if Q is not None:
        Wd = U2.astype(np.float32) * s2.astype(np.float32)[:, None] / 2
        assert np.array_equal(Wd, Q), tag + ": decoded weights differ from Q"
    alt = {}
    for mode, nm in ((fc.MODE_SHELL, "shell"), (fc.MODE_PROD, "prod")):
        b2 = fc.encode_matrix(U, steps, extra, mode=mode)
        assert np.array_equal(fc.decode_matrix(b2)[0], Uf), tag + " forced mode " + nm
        alt[nm] = len(b2)
    nw = Uf.size; real = 8 * len(blob)
    row = dict(tag=tag, m=Uf.shape[0], n=Uf.shape[1], nw=nw, ideal=ideal, real=real, mode=fc.blob_info(blob)["mode"],
               shell=8 * alt["shell"], prod=8 * alt["prod"], enc=nw / te / 1e6, dec=nw / td / 1e6)
    ROWS.append(row)
    ib = f"{ideal / nw:.4f}" if ideal else "   -  "
    ratio = f"{real / ideal:.5f}" if ideal else "   -   "
    log(f"{tag:34s} {Uf.shape[0]:5d}x{Uf.shape[1]:<5d} ideal {ib} real {real / nw:.4f} bpw ratio {ratio} "
        f"[shell {8 * alt['shell'] / nw:.4f} prod {8 * alt['prod'] / nw:.4f} -> {row['mode']}] "
        f"enc {row['enc']:6.1f} dec {row['dec']:6.1f} Mw/s")
    return blob


def rand_e8(rng, k, scale):
    """k valid doubled-E8 points with coordinates ~ scale."""
    z = rng.integers(-scale, scale, size=(k, 8)).astype(np.int64)
    out = np.empty_like(z)
    even = rng.random(k) < 0.5
    ze = z.copy(); ze[:, 7] += ze.sum(1) & 1; out[even] = 2 * ze[even]          # z in D8 -> u = 2z
    zo = 2 * z + 1; zo[:, 7] += (-(zo.sum(1)) % 4); out[~even] = zo[~even]     # odd coords, sum%4==0
    return out


def main():
    rng = np.random.default_rng(0); g = torch.Generator().manual_seed(0)
    log("== 1. synthetic ==   (ideal = bits_from_hist with set_limit(16384) + 16 bits/row; real = blob bytes*8)")
    Wg = torch.randn(2048, 4096, generator=g)
    for target in (2.0, 2.8, 3.5, 4.5, 6.5):
        mult = calibrate(Wg, target)
        U, s, Q, ideal = codes(Wg, mult)
        check(f"gauss ~{target} bpw (mult {mult:.4f})", U, s, Q, ideal)
    Wt = torch.distributions.StudentT(3.0).sample((2048, 4096))
    for mult in (0.5917, calibrate(Wt, 4.5)):
        U, s, Q, ideal = codes(Wt, mult); check(f"student-t3 mult {mult:.4f}", U, s, Q, ideal)
    Wz = torch.randn(512, 4096, generator=g); Wz[[0, 7, 300, 511]] = 0
    U, s, Q, ideal = codes(Wz, 0.5917)
    assert (U[[0, 7, 300, 511]] == 0).all()
    check("zero rows", U, s, Q, ideal)
    U, s, Q, ideal = codes(torch.randn(512, 4096, generator=g), 0.5917)
    V = U.reshape(-1, 8).copy(); idx = rng.choice(V.shape[0], 40, replace=False)
    V[idx] = rand_e8(rng, 40, 50000); V[idx[0]] = [2 ** 30, 2 ** 30, 0, 0, 0, 0, 0, 0]
    V[idx[1]] = [-(2 ** 31 - 1)] * 4 + [2 ** 31 - 1] * 4
    Uo = V.reshape(U.shape)
    assert np.abs(Uo).max() > 1e5
    check("huge outliers (|u|~1e5..2^31)", Uo, s, None, None)
    for (m, n) in ((1, 8), (2, 16)):
        Wsm = torch.randn(m, n, generator=g)
        U, s, Q, ideal = codes(Wsm, 0.5917); check(f"tiny {m}x{n}", U, s, Q, ideal)
    U, s, Q, ideal = codes(torch.randn(16, 17408, generator=g), 0.5917); check("wide 16x17408", U, s, Q, ideal)
    # all-zero matrix, int8 / 3-D input, extras
    check("all-zero 4x64", np.zeros((4, 8, 8), np.int8), np.ones(4, np.float16))
    U, s, Q, ideal = codes(torch.randn(64, 512, generator=g), 0.5917)
    ex = dict(bias=rng.standard_normal(64).astype(np.float16), col_scale=rng.random(512).astype(np.float16),
              perm=rng.permutation(512).astype(np.uint16), note={"src": "test", "k": 3})
    ideal_ex = ideal + 8 * sum(v.nbytes for v in ex.values() if isinstance(v, np.ndarray))   # + raw side arrays
    blob = check("extras (bias,col_scale,perm,json)", U.astype(np.int16).reshape(64, 64, 8), s, Q, ideal_ex, ex)
    # corruption + invalid input are rejected
    bad = bytearray(blob); bad[-5] ^= 1
    try: fc.decode_matrix(bytes(bad)); raise AssertionError("corruption not detected")
    except ValueError: pass
    Ubad = U.copy(); Ubad[0, 0] += 1
    try: fc.encode_matrix(Ubad, s); raise AssertionError("invalid E8 accepted")
    except ValueError: pass
    log("corruption detected, invalid E8 input rejected: OK")

    log("== 2. real weights (qwen3.8-27b, lazily sliced) ==")
    from safetensors import safe_open
    D = "/root/models/qwen3.8-27b/"; IDX = json.load(open(D + "model.safetensors.index.json"))["weight_map"]
    L = "model.language_model."
    cases = [(L + "layers.30.mlp.gate_proj.weight", 0, 2048, False), (L + "layers.30.mlp.down_proj.weight", 0, 1024, False),
             (L + "layers.30.linear_attn.in_proj_qkv.weight", 0, 2048, False), (L + "layers.31.self_attn.q_proj.weight", 0, 2048, False),
             (L + "layers.31.self_attn.o_proj.weight", 0, 2048, True), (L + "embed_tokens.weight", 1000, 3048, False),
             ("lm_head.weight", 1000, 3048, True)]
    real_rows = []
    for name, r0, r1, hi in cases:
        with safe_open(D + IDX[name], "pt") as f:
            W = f.get_slice(name)[r0:r1].float()
        short = name.replace(L, "").replace(".weight", "")
        mults = [0.5917] + ([calibrate(W, 4.5)] if hi else [])
        for mult in mults:
            U, s, Q, ideal = codes(W, mult)
            check(f"{short} m{mult:.4f}", U, s, Q, ideal); real_rows.append(ROWS[-1])
        del W

    log("== 3. package + parallel workers ==")
    tmp = tempfile.mkdtemp(dir=HERE, prefix="_tmp_pkg_")
    try:
        ind, pkg = os.path.join(tmp, "in"), os.path.join(tmp, "pkg"); os.makedirs(ind)
        ref = {}
        for i in range(6):
            U, s, Q, _ = codes(torch.randn(256, 1024, generator=g), 0.5917)
            nm = f"layers.{i}.mlp/weight"; ref[nm] = (U, s)
            np.savez(os.path.join(ind, f"t{i}.npz"), U=U.astype(np.int8), steps=s, name=np.array(nm),
                     bias=s.copy())
        procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "package.py"), "worker", ind, pkg],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) for _ in range(3)]
        time.sleep(1.0); open(os.path.join(ind, "_END"), "w").close()
        for p in procs: p.wait(); assert p.returncode == 0, p.stdout.read().decode()
        assert not [f for f in os.listdir(ind) if f != "_END"], os.listdir(ind)
        pk.write_raw(pkg, "model.norm.weight", rng.standard_normal(5120).astype(np.float16))
        for nm, (U, s) in ref.items():
            U2, s2, ex = pk.read_tensor(pkg, nm)
            assert np.array_equal(U2, U) and np.array_equal(s2, s) and np.array_equal(ex["bias"], s)
        out = subprocess.run([sys.executable, os.path.join(HERE, "package.py"), "verify", pkg], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        log("package/workers OK:", out.stdout.strip(), "| manifest entries:", len(pk.load_manifest(pkg)))
    finally:
        shutil.rmtree(tmp)

    log("== summary ==")
    log(f"{'case':34s} {'weights':>9s} {'ideal_bpw':>9s} {'real_bpw':>9s} {'ratio':>8s} {'mode':>5s} {'enc':>6s} {'dec':>6s}")
    for r in ROWS:
        if r["ideal"]:
            log(f"{r['tag']:34s} {r['nw']:9d} {r['ideal'] / r['nw']:9.4f} {r['real'] / r['nw']:9.4f} {r['real'] / r['ideal']:8.5f} "
                f"{r['mode']:>5s} {r['enc']:6.1f} {r['dec']:6.1f}")
    big = [r for r in ROWS if r["ideal"] and r["nw"] >= 1 << 20]
    for nm, rows in (("synthetic+real (>=1M weights)", big), ("real weights", real_rows)):
        I = sum(r["ideal"] for r in rows); R = sum(r["real"] for r in rows); N = sum(r["nw"] for r in rows)
        log(f"overall {nm}: real/ideal = {R / I:.6f} (overhead {100 * (R / I - 1):+.4f} %), "
            f"enc {N / sum(r['nw'] / r['enc'] for r in rows):.1f} Mw/s, dec {N / sum(r['nw'] / r['dec'] for r in rows):.1f} Mw/s (1 core)")
    I = sum(r["ideal"] for r in real_rows); R = sum(r["real"] for r in real_rows)
    ext = 26.893e9 * 2.808 * (R / I) / 8
    log(f"extrapolated: 26.893e9 weights @ 2.808 ideal bpw -> {26.893e9 * 2.808 / 8 / 1e9:.4f} GB ideal, "
        f"{ext / 1e9:.4f} GB real (x{R / I:.5f}); encode time on 30 cores at measured speed ≈ "
        f"{26.893e9 / (30 * 1e6 * min(r['enc'] for r in real_rows)) / 60:.1f} min (worst case)")
    log("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
