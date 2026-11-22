"""Lecteur minimal d'en-têtes GGUF (v3) — formic_inkling. Lit les métadonnées et la table des tenseurs sans dépendre
de l'énumération des types de gguf-py (les fichiers Unsloth utilisent des types récents)."""
import struct, numpy as np

# taille de bloc / octets par bloc (ggml), types utiles ici
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
         16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
         20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
         29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
         41: ("Q1_0", None, None), 42: ("Q2_0", None, None), 43: ("F8_E4M3", 1, 1)}

def _rs(f):
    n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode("utf-8", "replace")

def _rv(f, t):
    if t == 8: return _rs(f)
    if t == 9:
        it, n = struct.unpack("<IQ", f.read(12))
        if it in (8,):
            return [_rs(f) for _ in range(n)]
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[it]
        sz = struct.calcsize(fmt); return list(struct.unpack("<%d%s" % (n, fmt), f.read(sz * n)))
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

def read(path, keep_arrays=False):
    with open(path, "rb") as f:
        magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24)); assert magic == b"GGUF", magic
        kv = {}
        for _ in range(nkv):
            k = _rs(f); t = struct.unpack("<I", f.read(4))[0]; v = _rv(f, t)
            kv[k] = v if (keep_arrays or not isinstance(v, list) or len(v) < 64) else ("array", len(v))
        tens = []
        for _ in range(nt):
            name = _rs(f); nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd))); typ, off = struct.unpack("<IQ", f.read(12))
            tens.append(dict(name=name, dims=dims, type=typ, offset=off))
        align = kv.get("general.alignment", 32); pos = f.tell(); base = (pos + align - 1) // align * align
    import os
    fs = os.path.getsize(path)
    order = sorted(range(len(tens)), key=lambda i: tens[i]["offset"])
    for j, i in enumerate(order):
        t = tens[i]; t["abs"] = base + t["offset"]
        nxt = tens[order[j + 1]]["abs"] if False else (base + tens[order[j + 1]]["offset"] if j + 1 < len(order) else fs)
        t["span"] = nxt - t["abs"]                     # octets jusqu'au tenseur suivant (padding d'alignement inclus)
        t["n"] = int(np.prod(t["dims"]))
        name, bs, bb = TYPES.get(t["type"], ("T%d" % t["type"], None, None))
        t["tname"] = name
        t["bytes"] = (t["n"] // bs * bb) if bs else None
    return kv, tens, base
