"""Dequantize the QUANTIZED weight tensors of a llama.cpp GGUF file to float16 .npy files (CPU process pool).
A consumer (lab server job) loads and deletes them as they appear, so the disk backlog stays small.
Only data is read from the GGUF (gguf-py reader + its reference dequantizers); nothing from it is executed.
Usage: python -I gguf_dequant.py <file.gguf> <out_dir> [workers] [max_backlog_gb]"""
import sys, os, json, time, glob
import numpy as np
import gguf
from concurrent.futures import ProcessPoolExecutor

def work(args):
    path, idx, out = args
    r = gguf.GGUFReader(path); t = r.tensors[idx]
    a = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float16)
    tmp = os.path.join(out, t.name + ".part")                       # consumer only looks at *.npy
    with open(tmp, "wb") as f: np.save(f, a)
    os.replace(tmp, os.path.join(out, t.name + ".npy"))
    return t.name, t.tensor_type.name, list(a.shape), int(t.n_bytes)

if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    limit = float(sys.argv[4]) * 1e9 if len(sys.argv) > 4 else 8e9
    os.makedirs(out, exist_ok=True)
    r = gguf.GGUFReader(path)
    idxs = [i for i, t in enumerate(r.tensors) if t.tensor_type.name not in ("F32", "F16", "BF16", "F64")]
    kept = {t.name: [t.tensor_type.name, int(t.n_bytes)] for t in r.tensors if t.tensor_type.name in ("F32", "F16", "BF16", "F64")}
    idxs.sort(key=lambda i: -int(r.tensors[i].n_elements)); t0 = time.time(); meta = {}
    with ProcessPoolExecutor(workers) as ex:
        for b in range(0, len(idxs), workers):
            while sum(os.path.getsize(f) for f in glob.glob(out + "/*.npy")) > limit: time.sleep(0.5)   # wait for the consumer
            for res in ex.map(work, [(path, i, out) for i in idxs[b:b + workers]]):
                meta[res[0]] = dict(type=res[1], shape=res[2], bytes=res[3])
    json.dump(dict(file=os.path.basename(path), file_bytes=os.path.getsize(path), tensors=meta, kept_unquantized=kept), open(os.path.join(out, "_manifest.json"), "w"))
    print("done %d tensors in %.0fs" % (len(idxs), time.time() - t0))
