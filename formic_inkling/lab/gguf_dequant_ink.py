"""Déquantifie les tenseurs quantifiés (et BF16) d'un GGUF Inkling 2 Mini d'Unsloth en .npy float32 (pool de processus CPU).
Dérivé de formic_v7/lab/gguf_dequant.py. Lecture des en-têtes par gguf_hdr.py (les types 40–43 sont inconnus de gguf 0.19) ;
types K/IQ/Q8_0/BF16 : déquantifieurs de référence de gguf-py 0.19 (pip) ; NVFP4 (40), Q2_0 (42), F8_E4M3 (43) :
implémentation maison écrite d'après gguf-py de llama.cpp master (lu, non exécuté). Rien du GGUF n'est exécuté.
Usage : python -I gguf_dequant_ink.py <fichier.gguf> <dossier> [workers] [backlog_go]"""
import sys, os, json, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_hdr as G
from concurrent.futures import ProcessPoolExecutor

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)

def ue4m3_half(x):
    x = x.astype(np.int32); e = (x >> 3) & 0xF; m = (x & 7).astype(np.float32)
    raw = np.where(e == 0, m * 2.0 ** -9, (1.0 + m / 8.0) * np.exp2(e.astype(np.float32) - 7))
    return np.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5).astype(np.float32)

def deq_nvfp4(b):                       # b [nb, 36]
    d = ue4m3_half(b[:, :4]).reshape(-1, 4, 1)
    qs = b[:, 4:].reshape(-1, 4, 8)
    v = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)            # (nb,4,16) : quartets bas puis hauts
    return (d * KV[v]).reshape(-1, 64)

def deq_q2_0(b):                        # b [nb, 18]
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:].reshape(-1, 16, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8)
    q = (qs & 3).reshape(-1, 64).astype(np.float32) - 1.0
    return d * q

def deq_f8(b):
    b = b.astype(np.uint8); s = np.where(b & 0x80, -1.0, 1.0).astype(np.float32); mag = b & 0x7F
    e = (mag >> 3).astype(np.int32); m = (mag & 7).astype(np.float32)
    v = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1.0 + m / 8.0, e - 7)).astype(np.float32)
    return np.where(mag == 0x7F, np.nan, s * v).astype(np.float32)

OWN = {40: (64, 36, deq_nvfp4), 42: (64, 18, deq_q2_0)}

def work(args):
    path, t, out, r0, r1 = args
    dims = t["dims"]; ne0 = dims[0]; rows = int(np.prod(dims[1:]))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    typ = t["type"]
    if typ == 43:
        a = deq_f8(np.asarray(mm[t["abs"] + r0 * ne0: t["abs"] + r1 * ne0])).reshape(r1 - r0, ne0)
    elif typ in OWN:
        bs, bb, f = OWN[typ]; rb = ne0 // bs * bb
        raw = np.asarray(mm[t["abs"] + r0 * rb: t["abs"] + r1 * rb]).reshape(-1, bb)
        a = f(raw).reshape(r1 - r0, ne0)
    else:
        import gguf
        qt = gguf.GGMLQuantizationType(typ)
        bs, bb = gguf.GGML_QUANT_SIZES[qt]; rb = ne0 // bs * bb
        raw = np.asarray(mm[t["abs"] + r0 * rb: t["abs"] + r1 * rb]).reshape(r1 - r0, rb)
        a = gguf.quants.dequantize(raw, qt).astype(np.float32).reshape(r1 - r0, ne0)
    nm = f"{t['name']}@{r0}"
    tmp = os.path.join(out, nm + ".part")
    with open(tmp, "wb") as f_: np.save(f_, a)
    os.replace(tmp, os.path.join(out, nm + ".npy"))
    return nm

if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    limit = float(sys.argv[4]) * 1e9 if len(sys.argv) > 4 else 30e9
    os.makedirs(out, exist_ok=True)
    kv, T, base = G.read(path)
    sel = [t for t in T if t["tname"] not in ("F32", "F16")]
    jobs = []
    for t in sel:
        rows = int(np.prod(t["dims"][1:])); step = 2304 * 8 if rows > 2304 * 8 else rows   # morceaux de 8 experts
        for r0 in range(0, rows, step): jobs.append((path, t, out, r0, min(rows, r0 + step)))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for b in range(0, len(jobs), workers):
            while sum(os.path.getsize(f) for f in glob.glob(out + "/*.npy")) > limit: time.sleep(0.3)
            list(ex.map(work, jobs[b:b + workers]))
    meta = {t["name"]: dict(type=t["tname"], dims=t["dims"], bytes=t["span"]) for t in T}
    json.dump(dict(file=os.path.basename(path), file_bytes=os.path.getsize(path), tensor_bytes=sum(t["span"] for t in T),
                   tensors=meta, chunks={t["name"]: [j[3] for j in jobs if j[1]["name"] == t["name"]] for t in sel}),
              open(os.path.join(out, "_manifest.json"), "w"))
    print("done %d chunks in %.0fs" % (len(jobs), time.time() - t0))
