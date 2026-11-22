"""Decode a package directory to float32 .npy files, one per tensor, for a consumer that loads and deletes them
(keeps the disk backlog bounded).  CPU only.  Usage: python decode_stream.py <pkg_dir> <out_dir> [workers] [max_backlog_gb]"""
import sys, os, json, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import package
from concurrent.futures import ProcessPoolExecutor

def work(args):
    pkg, name, out = args
    U, steps, extra = package.read_tensor(pkg, name)
    W = U.astype(np.float32) * (steps.astype(np.float32)[:, None] / 2.0)
    if extra and "col_scale" in extra: W = W / extra["col_scale"].astype(np.float32)[None, :]
    tmp = os.path.join(out, name + ".part")
    with open(tmp, "wb") as f: np.save(f, W)
    os.rename(tmp, os.path.join(out, name + ".npy"))
    return name

if __name__ == "__main__":
    pkg, out = sys.argv[1], sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    limit = float(sys.argv[4]) * 1e9 if len(sys.argv) > 4 else 10e9
    os.makedirs(out, exist_ok=True); man = package.load_manifest(pkg)
    ent = man["tensors"] if "tensors" in man else man
    names = [k for k, v in ent.items() if isinstance(v, dict) and v.get("kind", "tensor") in ("tensor", "e8", "matrix")]
    def backlog():
        tot = 0
        for f_ in glob.glob(out + "/*.npy"):
            try: tot += os.path.getsize(f_)
            except FileNotFoundError: pass          # consommé entre glob et getsize (formic_inkling)
        return tot
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for b in range(0, len(names), workers):
            while backlog() > limit: time.sleep(0.3)
            list(ex.map(work, [(pkg, n, out) for n in names[b:b + workers]]))
    open(os.path.join(out, "_DONE"), "w").write(str(len(names)))
    print("decoded %d tensors in %.0fs" % (len(names), time.time() - t0))
