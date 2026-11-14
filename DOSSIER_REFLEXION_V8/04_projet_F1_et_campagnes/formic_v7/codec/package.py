"""Package directory of FE8-coded matrices + raw small tensors, with an atomically updated manifest.json.

  write_tensor(pkg, name, U, steps, extra)   -> one <safe>.fe8 file per matrix (fcodec blob)
  read_tensor(pkg, name)                     -> (U, steps, extra)
  write_raw(pkg, name, array)                -> <safe>.raw (FP16 / float32 / any dtype, small tensors)
  read_raw(pkg, name)
  package_bytes(pkg)                         -> total bytes of every file in the directory (manifest included)
  encode_worker_loop(in_dir, pkg)            -> streaming encoder (claims <name>.npz by atomic rename)
CLI:  python package.py verify <pkg_dir>      |   python package.py worker <in_dir> <pkg_dir>
"""
import fcntl, hashlib, json, os, re, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcodec

MANIFEST = "manifest.json"
RAW_MAGIC = b"FE8RAW01"


def sanitize(name):
    s = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if s != name or len(s) > 180:      # keep names unique after sanitising
        s = s[:160] + "-" + hashlib.sha1(name.encode()).hexdigest()[:10]
    return s


def _atomic_write(path, data):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _update_manifest(pkg_dir, name, entry):
    """read-modify-write under an flock, then atomic replace (safe for parallel workers)."""
    os.makedirs(pkg_dir, exist_ok=True)
    with open(os.path.join(pkg_dir, ".manifest.lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            man = load_manifest(pkg_dir)
            man[name] = entry
            _atomic_write(os.path.join(pkg_dir, MANIFEST), json.dumps(man, indent=1, sort_keys=True).encode())
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def load_manifest(pkg_dir):
    p = os.path.join(pkg_dir, MANIFEST)
    return json.load(open(p)) if os.path.exists(p) else {}


def write_tensor(pkg_dir, name, U, steps, extra=None):
    blob = fcodec.encode_matrix(U, steps, extra)
    fn = sanitize(name) + ".fe8"
    os.makedirs(pkg_dir, exist_ok=True)
    _atomic_write(os.path.join(pkg_dir, fn), blob)
    info = fcodec.blob_info(blob)
    _update_manifest(pkg_dir, name, dict(file=fn, bytes=len(blob), shape=[info["m"], info["n"]], kind="e8",
                                         mode=info["mode"], sha256=hashlib.sha256(blob).hexdigest()))
    return len(blob)


def _read_checked(pkg_dir, name):
    e = load_manifest(pkg_dir)[name]
    data = open(os.path.join(pkg_dir, e["file"]), "rb").read()
    if hashlib.sha256(data).hexdigest() != e["sha256"]: raise ValueError(f"sha256 mismatch for {name}")
    return e, data


def read_tensor(pkg_dir, name):
    e, blob = _read_checked(pkg_dir, name)
    return fcodec.decode_matrix(blob)


def write_raw(pkg_dir, name, array):
    a = np.ascontiguousarray(array)
    if a.dtype == np.float64: a = a.astype(np.float32)
    dt = a.dtype.str.encode()
    body = struct.pack("<B", len(dt)) + dt + struct.pack("<B", a.ndim) + struct.pack(f"<{a.ndim}Q", *a.shape) + a.tobytes()
    data = RAW_MAGIC + struct.pack("<I", __import__("zlib").crc32(body)) + body
    fn = sanitize(name) + ".raw"
    os.makedirs(pkg_dir, exist_ok=True)
    _atomic_write(os.path.join(pkg_dir, fn), data)
    _update_manifest(pkg_dir, name, dict(file=fn, bytes=len(data), shape=list(a.shape), kind="raw",
                                         dtype=a.dtype.str, sha256=hashlib.sha256(data).hexdigest()))
    return len(data)


def _parse_raw(data):
    import zlib
    if data[:8] != RAW_MAGIC: raise ValueError("not a raw tensor file")
    (crc,) = struct.unpack_from("<I", data, 8); body = data[12:]
    if zlib.crc32(body) != crc: raise ValueError("raw tensor checksum mismatch")
    ld = body[0]; dt = np.dtype(body[1:1 + ld].decode()); p = 1 + ld
    nd = body[p]; p += 1; shape = struct.unpack_from(f"<{nd}Q", body, p); p += 8 * nd
    return np.frombuffer(body[p:], dtype=dt).reshape(shape).copy()


def read_raw(pkg_dir, name):
    e, data = _read_checked(pkg_dir, name)
    return _parse_raw(data)


def package_bytes(pkg_dir):
    tot = 0
    for root, _, files in os.walk(pkg_dir):
        for f in files: tot += os.path.getsize(os.path.join(root, f))
    return tot


def verify(pkg_dir):
    man = load_manifest(pkg_dir); n_ok = 0; weights = 0
    for name, e in sorted(man.items()):
        _, data = _read_checked(pkg_dir, name)
        if len(data) != e["bytes"]: raise ValueError(f"size mismatch for {name}")
        if e["kind"] == "e8":
            U, steps, _ = fcodec.decode_matrix(data)
            if list(U.shape) != e["shape"]: raise ValueError(f"shape mismatch for {name}")
            weights += U.size
        else:
            _parse_raw(data)
        n_ok += 1
    tot = package_bytes(pkg_dir)
    print(f"verified {n_ok} entries, {weights} coded weights, package_bytes={tot} "
          f"({8 * tot / max(weights, 1):.4f} bits/coded weight incl. everything)")
    return tot


def encode_worker_loop(in_dir, pkg_dir, poll=0.5, verbose=True):
    """Encode <in_dir>/<name>.npz (U int8/16/32 [m,n], steps fp16 [m], optional extra arrays) into pkg_dir.
    Claims a file by atomic rename to <name>.npz.claimed.<pid>; deletes it when done. Exits when in_dir/_END
    exists and no unclaimed .npz remain. Name inside the package = file stem (or the 'name' array if present)."""
    pid = os.getpid(); done = 0
    while True:
        cands = sorted(f for f in os.listdir(in_dir) if f.endswith(".npz"))
        if not cands:
            if os.path.exists(os.path.join(in_dir, "_END")):
                if verbose: print(f"[worker {pid}] END, encoded {done}", flush=True)
                return done
            time.sleep(poll); continue
        for f in cands:
            src = os.path.join(in_dir, f); claimed = f"{src}.claimed.{pid}"
            try:
                os.rename(src, claimed)            # atomic claim; loser gets FileNotFoundError
            except FileNotFoundError:
                continue
            t0 = time.time()
            with np.load(claimed, allow_pickle=False) as z:
                name = str(z["name"]) if "name" in z.files else f[:-4]
                extra = {k: z[k] for k in z.files if k not in ("U", "steps", "name")}
                nb = write_tensor(pkg_dir, name, z["U"], z["steps"], extra or None)
                nw = z["U"].size
            os.remove(claimed); done += 1
            if verbose:
                dt = time.time() - t0
                print(f"[worker {pid}] {name}: {nb} B, {8 * nb / nw:.4f} bpw, {nw / dt / 1e6:.1f} Mw/s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        verify(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "worker":
        encode_worker_loop(sys.argv[2], sys.argv[3])
    else:
        print(__doc__); sys.exit(2)
