"""FORMIC E8 codec: real, exact, fast entropy coding of doubled-E8 codes U [m, n] (n % 8 == 0) + FP16 row steps.

decoded weight = U.astype(float32) * steps.astype(float32)[:, None] / 2

Two models, chosen per matrix from an exact cost estimate (see README.md):
  mode 0 SHELL: shell index m = |u|^2/8 (categorical table stored in the blob) + the point inside the shell coded
                EXACTLY uniformly by enumerative range coding (= the ideal `bits_from_hist` cost + ~0 overhead).
  mode 1 PROD : coset bit, 7 coordinates v=(u-c)>>1 with a per-coset table, last coordinate's parity implied by the
                E8 constraint (sum v even) so only k = v_7 >> 1 is coded (table per coset x parity).
The core is a 64-bit range coder in C (fe8c.c, compiled on first import with gcc, cached next to this file).
"""
import ctypes, hashlib, json, os, struct, subprocess, tempfile, zlib
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "fe8c.c")
MAGIC = b"FE8CODC1"
VERSION = 1
HDR = struct.Struct("<8sHBBIIQI")       # magic, version, mode, P, m, n, payload_len, crc32(payload)
P = 18                                   # categorical tables are normalised to 2^P
W = 2048                                 # product-mode coordinate window (outside -> escape + raw 32 bit)
RMAX_SHELL = 300                         # count table range; usable shells limited by N(m) < 2^32
MODE_SHELL, MODE_PROD = 0, 1


def _load_lib():
    src = open(_SRC, "rb").read()
    tag = hashlib.sha1(src).hexdigest()[:12]
    so = os.path.join(_DIR, f"_fe8c_{tag}.so")
    if not os.path.exists(so):
        fd, tmp = tempfile.mkstemp(suffix=".so", dir=_DIR); os.close(fd)
        subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", "-o", tmp, _SRC])
        os.replace(tmp, so)              # atomic: safe with several workers compiling at once
    lib = ctypes.CDLL(so)
    vp, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    lib.fe8_init_counts.argtypes = [i32]; lib.fe8_init_counts.restype = i32
    lib.fe8_count.argtypes = [i32, i32, i32, i32]; lib.fe8_count.restype = ctypes.c_uint64
    lib.fe8_stats.argtypes = [vp, i64, i32, vp, i32, vp]; lib.fe8_stats.restype = i64
    lib.fe8_encode.argtypes = [i32, vp, i64, i32, vp, vp, i32, vp, i64]; lib.fe8_encode.restype = i64
    lib.fe8_decode.argtypes = [i32, vp, i64, i64, i32, vp, vp, i32, vp]; lib.fe8_decode.restype = i32
    if lib.fe8_init_counts(8 * RMAX_SHELL) != 0: raise MemoryError("count table")
    return lib


_lib = _load_lib()
_NSHELL = np.array([_lib.fe8_count(0, 8, 8 * k, 0) + _lib.fe8_count(1, 8, 8 * k, 0) for k in range(RMAX_SHELL + 1)],
                   dtype=np.float64)
KMAX = int(np.argmax(_NSHELL >= 2.0 ** 32)) if (_NSHELL >= 2.0 ** 32).any() else RMAX_SHELL + 1  # shells 0..KMAX-1
_LOG2N = np.log2(np.maximum(_NSHELL, 1))
assert _NSHELL[1] == 240 and _NSHELL[2] == 2160


def _ptr(a): return a.ctypes.data_as(ctypes.c_void_p)


# ---------------------------------------------------------------- tables
def _normalise(cnt):
    """counts (int64) -> integer freqs summing to 2^P, >0 exactly where cnt>0 (deterministic)."""
    cnt = np.asarray(cnt, dtype=np.int64); tot = int(cnt.sum())
    f = np.zeros(len(cnt), dtype=np.int64)
    if tot == 0:
        f[0] = 1 << P; return f
    nz = cnt > 0
    f[nz] = np.maximum(1, np.floor(cnt[nz].astype(np.float64) * (1 << P) / tot)).astype(np.int64)
    d = (1 << P) - int(f.sum())
    order = np.argsort(-cnt, kind="stable")
    i = 0
    while d != 0:                        # fix rounding on the most frequent symbols
        j = order[i % int(nz.sum())]
        step = 1 if d > 0 else -1
        if f[j] + step >= 1: f[j] += step; d -= step
        i += 1
    return f


def _bits(cnt, f):
    nz = cnt > 0
    return float((cnt[nz] * (P - np.log2(f[nz]))).sum())


def _uvarint(out, x):
    while True:
        b = x & 0x7F; x >>= 7
        if x: out.append(b | 0x80)
        else: out.append(b); return


def _read_uvarint(buf, pos):
    x = 0; sh = 0
    while True:
        b = buf[pos]; pos += 1; x |= (b & 0x7F) << sh; sh += 7
        if not b & 0x80: return x, pos


def _pack_tables(tabs):
    """tabs: list of (lo, freqs). freqs RLE: value>0 -> varint(value); zeros -> 0, varint(run)."""
    out = bytearray([len(tabs)])
    for lo, f in tabs:
        _uvarint(out, (lo << 1) ^ (lo >> 63) if lo < 0 else lo << 1)
        _uvarint(out, len(f))
        i = 0; f = [int(x) for x in f]
        while i < len(f):
            if f[i]: _uvarint(out, f[i]); i += 1
            else:
                j = i
                while j < len(f) and f[j] == 0: j += 1
                out.append(0); _uvarint(out, j - i); i = j
    return bytes(out)


def _unpack_tables(buf, pos):
    nt = buf[pos]; pos += 1; tabs = []
    for _ in range(nt):
        z, pos = _read_uvarint(buf, pos); lo = (z >> 1) ^ -(z & 1)
        ns, pos = _read_uvarint(buf, pos); f = []
        while len(f) < ns:
            v, pos = _read_uvarint(buf, pos)
            if v: f.append(v)
            else:
                run, pos = _read_uvarint(buf, pos); f.extend([0] * run)
        if len(f) != ns or sum(f) != (1 << P): raise ValueError("corrupt model table")
        tabs.append((lo, np.array(f, dtype=np.int64)))
    return tabs, pos


def _tab_arrays(tabs):
    desc = []; cums = []; off = 0
    for lo, f in tabs:
        c = np.concatenate([[0], np.cumsum(f)]).astype(np.uint32)
        desc += [lo, len(f), off]; cums.append(c); off += len(c)
    return np.array(desc, dtype=np.int32), np.ascontiguousarray(np.concatenate(cums), dtype=np.uint32)


# ---------------------------------------------------------------- extras
def _pack_extra(extra):
    out = bytearray(); items = sorted((extra or {}).items())
    out += struct.pack("<H", len(items))
    for name, val in items:
        nb = name.encode()
        if isinstance(val, np.ndarray):
            a = np.ascontiguousarray(val); dt = a.dtype.str.encode()
            out += struct.pack("<BB", 0, len(nb)) + nb + struct.pack("<B", len(dt)) + dt + struct.pack("<B", a.ndim)
            out += struct.pack(f"<{a.ndim}Q", *a.shape) + struct.pack("<Q", a.nbytes) + a.tobytes()
        else:
            js = json.dumps(val).encode()
            out += struct.pack("<BB", 1, len(nb)) + nb + struct.pack("<Q", len(js)) + js
    return bytes(out)


def _unpack_extra(buf, pos):
    (cnt,) = struct.unpack_from("<H", buf, pos); pos += 2; ex = {}
    for _ in range(cnt):
        kind, ln = struct.unpack_from("<BB", buf, pos); pos += 2
        name = bytes(buf[pos:pos + ln]).decode(); pos += ln
        if kind == 0:
            (ld,) = struct.unpack_from("<B", buf, pos); pos += 1
            dt = np.dtype(bytes(buf[pos:pos + ld]).decode()); pos += ld
            (nd,) = struct.unpack_from("<B", buf, pos); pos += 1
            shape = struct.unpack_from(f"<{nd}Q", buf, pos); pos += 8 * nd
            (nbytes,) = struct.unpack_from("<Q", buf, pos); pos += 8
            ex[name] = np.frombuffer(bytes(buf[pos:pos + nbytes]), dtype=dt).reshape(shape).copy(); pos += nbytes
        else:
            (ln2,) = struct.unpack_from("<Q", buf, pos); pos += 8
            ex[name] = json.loads(bytes(buf[pos:pos + ln2]).decode()); pos += ln2
    return ex, pos


# ---------------------------------------------------------------- models
def _stats(U32):
    nvec = U32.shape[0]
    sh = np.zeros(KMAX + 2, dtype=np.int64)        # [0..KMAX] exact, [KMAX+1] beyond
    ph = np.zeros((7, 2 * W + 2), dtype=np.int64)
    bad = _lib.fe8_stats(_ptr(U32), nvec, KMAX, _ptr(sh), W, _ptr(ph))
    if bad >= 0:
        raise ValueError(f"vector {bad} is not a doubled-E8 point: {U32[bad].tolist()}")
    return sh, ph


def _shell_model(sh):
    """symbols m = 0..K-1 exact (K <= KMAX), symbol K = escape (raw 256 bits)."""
    used = np.nonzero(sh[:KMAX])[0]
    K = int(used.max()) + 1 if len(used) else 1
    cnt = np.concatenate([sh[:K], [sh[K:].sum()]])
    f = _normalise(cnt)
    bits = _bits(cnt, f) + float((sh[:K] * _LOG2N[:K]).sum()) + 256.0 * cnt[-1]
    return [(0, f)], bits


def _prod_model(ph):
    tabs = []; bits = 0.0
    cc = ph[0, :2]; f = _normalise(cc); tabs.append((0, f)); bits += _bits(cc, f)
    for t in range(1, 7):
        h = ph[t]; nz = np.nonzero(h[:2 * W + 1])[0]
        lo, hi = (int(nz.min()), int(nz.max())) if len(nz) else (W, W)
        cnt = np.concatenate([h[lo:hi + 1], [h[2 * W + 1]]])
        f = _normalise(cnt); tabs.append((lo - W, f)); bits += _bits(cnt, f) + 32.0 * cnt[-1]
    return tabs, bits


def _as_codes(U):
    U = np.asarray(U)
    if U.ndim == 3:
        if U.shape[2] != 8: raise ValueError("U must be (m, n/8, 8) or (m, n)")
        U = U.reshape(U.shape[0], -1)
    if U.ndim != 2 or U.shape[1] % 8: raise ValueError("U must be 2-D with n % 8 == 0")
    if not np.issubdtype(U.dtype, np.integer): raise TypeError("U must be an integer array")
    if U.dtype.itemsize > 4 or U.dtype == np.uint32:
        if U.size and (U.min() < -2 ** 31 or U.max() >= 2 ** 31): raise ValueError("codes must fit in int32")
    return np.ascontiguousarray(U, dtype=np.int32)


def encode_matrix(U, steps, extra=None, mode=None):
    """-> bytes. mode None = automatic (smaller estimated size), or MODE_SHELL / MODE_PROD."""
    U = _as_codes(U); m, n = U.shape
    steps = np.ascontiguousarray(np.asarray(steps), dtype=np.float16).reshape(-1)
    if steps.dtype != np.float16 or steps.shape[0] != m: raise ValueError("steps must be float16 (m,)")
    V = U.reshape(-1, 8); nvec = V.shape[0]
    sh, ph = _stats(V)
    st, sb = _shell_model(sh)
    pt, pb = _prod_model(ph)
    sb += 8 * len(_pack_tables(st)); pb += 8 * len(_pack_tables(pt))     # model tables count in the size
    if mode is None: mode = MODE_SHELL if sb <= pb else MODE_PROD
    tabs, est = (st, sb) if mode == MODE_SHELL else (pt, pb)
    desc, cums = _tab_arrays(tabs)
    cap = int(est / 8 * 1.02) + 4096 + nvec // 64
    while True:
        buf = np.empty(cap, dtype=np.uint8)
        ln = _lib.fe8_encode(mode, _ptr(V), nvec, len(tabs), _ptr(desc), _ptr(cums), P, _ptr(buf), cap)
        if ln >= 0: break
        cap = cap * 2 + 1024
    payload = b"".join([steps.tobytes(), _pack_extra(extra), _pack_tables(tabs), struct.pack("<Q", ln),
                        buf[:ln].tobytes()])
    return HDR.pack(MAGIC, VERSION, mode, P, m, n, len(payload), zlib.crc32(payload)) + payload


def decode_matrix(blob):
    """-> (U int32 (m, n), steps float16 (m,), extra dict)."""
    mv = memoryview(blob)
    magic, ver, mode, p, m, n, plen, crc = HDR.unpack_from(mv, 0)
    if magic != MAGIC or ver != VERSION or p != P: raise ValueError("not an FE8 codec blob / unsupported version")
    payload = mv[HDR.size:HDR.size + plen]
    if len(payload) != plen or zlib.crc32(payload) != crc: raise ValueError("checksum mismatch (corrupt blob)")
    pos = 0
    steps = np.frombuffer(payload, dtype=np.float16, count=m, offset=0).copy(); pos += 2 * m
    extra, pos = _unpack_extra(payload, pos)
    tabs, pos = _unpack_tables(payload, pos)
    (ln,) = struct.unpack_from("<Q", payload, pos); pos += 8
    stream = np.frombuffer(payload, dtype=np.uint8, count=ln, offset=pos)
    desc, cums = _tab_arrays(tabs)
    nvec = m * n // 8
    U = np.empty((m, n), dtype=np.int32)
    rc = _lib.fe8_decode(mode, _ptr(stream), ln, nvec, len(tabs), _ptr(desc), _ptr(cums), P, _ptr(U))
    if rc != 0: raise ValueError(f"decode error {rc}")
    return U, steps, extra


def decode_weights(blob):
    U, steps, extra = decode_matrix(blob)
    return U.astype(np.float32) * steps.astype(np.float32)[:, None] / 2, extra


def estimate_bits(U):
    """Estimated payload bits of both models (no coding): dict(shell=..., prod=...)."""
    sh, ph = _stats(_as_codes(U).reshape(-1, 8))
    (st, sb), (pt, pb) = _shell_model(sh), _prod_model(ph)
    return dict(shell=sb + 8 * len(_pack_tables(st)), prod=pb + 8 * len(_pack_tables(pt)))


def blob_info(blob):
    magic, ver, mode, p, m, n, plen, crc = HDR.unpack_from(memoryview(blob), 0)
    return dict(mode=("shell", "prod")[mode], m=m, n=n, bytes=len(blob))
