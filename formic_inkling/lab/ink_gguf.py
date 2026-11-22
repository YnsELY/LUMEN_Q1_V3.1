"""Chargement d'un GGUF concurrent dans le graphe Transformers d'Inkling (dérivé de formic_v7/lab/v7.py load_gguf).
Le rangement est vérifié tenseur par tenseur (cosinus avec le professeur, et écart relatif) AVANT toute mesure."""
import os, sys, json, time, glob, shutil, subprocess
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_hdr as G
import ink_quant as Q

LAB = os.path.dirname(os.path.abspath(__file__))


def target(name):
    """nom GGUF -> (paramètre, transformation) ; None = conservé du professeur."""
    lm = Q.S["lm"]; model = Q.S["model"]
    if name == "token_embd.weight": return lm.embed_tokens.weight
    if name == "output.weight": return model.lm_head.weight
    if not name.startswith("blk."): return None
    p = name.split("."); i = int(p[1]); k = p[2]; L = lm.layers[i]
    m = {"attn_q": L.self_attn.q_proj.linear, "attn_k": L.self_attn.k_proj.linear, "attn_v": L.self_attn.v_proj.linear,
         "attn_rel": L.self_attn.relative_position_bias.linear, "attn_output": L.self_attn.o_proj.linear,
         "attn_gate": L.self_attn.o_proj.gate_proj, "attn_sink_proj": L.self_attn.sink_proj}
    if k in m: return m[k].weight
    if k == "ffn_gate_up_exps": return L.mlp.experts.gate_up_proj
    if k == "ffn_down_exps": return L.mlp.experts.down_proj
    return None


@torch.no_grad()
def load_gguf(path, workers=48, tmp="/dev/shm/ink_deq"):
    Q.restore(); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    kv, T, base = G.read(path); byname = {t["name"]: t for t in T}
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    def f32(name):
        t = byname[name]; return torch.from_numpy(np.frombuffer(mm[t["abs"]:t["abs"] + 4 * t["n"]].tobytes(), dtype=np.float32).copy())
    proc = subprocess.Popen([sys.executable, "-I", "-B", f"{LAB}/gguf_dequant_ink.py", path, tmp, str(workers), "30"],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=open(tmp + "_stdout.log", "w"), stderr=subprocess.STDOUT)
    acc = {}; t0 = time.time(); skipped = set()
    while True:
        files = sorted(glob.glob(tmp + "/*.npy")); done = os.path.exists(tmp + "/_manifest.json")
        if not files:
            if done: break
            if proc.poll() is not None and not done: raise RuntimeError("déquantification échouée : " + open(tmp + "_stdout.log").read()[-1500:])
            time.sleep(0.2); continue
        for f in files:
            nm = os.path.basename(f)[:-4]; name, r0 = nm.rsplit("@", 1); r0 = int(r0)
            a = torch.from_numpy(np.load(f)); os.remove(f)
            W = target(name)
            if W is None: skipped.add(name); continue
            t = byname[name]; ne0 = t["dims"][0]
            if W.dim() == 3:                                        # experts [E, rows, ne0]
                rows_e = W.shape[1]; e0 = r0 // rows_e; ne = a.shape[0] // rows_e
                q = a.view(ne, rows_e, ne0).cuda()
                sc = f"{name.rsplit('.', 1)[0]}.scale"
                if sc in byname: q = q * f32(sc)[e0:e0 + ne].cuda()[:, None, None]
                w = W[e0:e0 + ne].float()
                st = acc.setdefault(name, dict(dot=0.0, qq=0.0, ww=0.0, dd=0.0, params=0, scaled=sc in byname, exact=True))
                st["dot"] += float((q * w).sum()); st["qq"] += float((q * q).sum()); st["ww"] += float((w * w).sum()); st["dd"] += float(((q - w) ** 2).sum())
                st["exact"] = st["exact"] and bool(torch.equal(q.to(W.dtype), W[e0:e0 + ne])); st["params"] += q.numel()
                W[e0:e0 + ne].copy_(q.to(W.dtype))
            else:
                q = a.cuda()
                if name == "output.weight": q = q[:W.shape[0]] if r0 == 0 else None
                if q is None: continue
                q = q[: W.shape[0] - r0]
                w = W[r0:r0 + q.shape[0]].float()
                if q.shape[1] != w.shape[1]: raise RuntimeError(f"forme {name} {tuple(q.shape)} contre {tuple(w.shape)}")
                st = acc.setdefault(name, dict(dot=0.0, qq=0.0, ww=0.0, dd=0.0, params=0, scaled=False, exact=True))
                st["dot"] += float((q * w).sum()); st["qq"] += float((q * q).sum()); st["ww"] += float((w * w).sum()); st["dd"] += float(((q - w) ** 2).sum())
                st["exact"] = st["exact"] and bool(torch.equal(q.to(W.dtype), W[r0:r0 + q.shape[0]])); st["params"] += q.numel()
                W[r0:r0 + q.shape[0]].copy_(q.to(W.dtype))
    man = json.load(open(tmp + "/_manifest.json"))
    rep = {}
    for k, s in acc.items():
        rep[k] = dict(type=man["tensors"][k]["type"], bytes=man["tensors"][k]["bytes"], params=s["params"], scaled=s["scaled"],
                      cos=s["dot"] / (s["qq"] ** .5 * s["ww"] ** .5 + 1e-30), rel_err=(s["dd"] / (s["ww"] + 1e-30)) ** .5,
                      norm_ratio=(s["qq"] / (s["ww"] + 1e-30)) ** .5, bit_exact_vs_teacher=s["exact"])
    shutil.rmtree(tmp, ignore_errors=True)
    bad = {k: round(v["cos"], 4) for k, v in rep.items() if v["cos"] < 0.80 or abs(v["norm_ratio"] - 1) > 0.2}
    if bad: raise RuntimeError("correspondance des tenseurs NON validée : %s" % dict(list(bad.items())[:8]))
    return dict(file=os.path.basename(path), file_bytes=os.path.getsize(path), tensor_bytes=man["tensor_bytes"], loaded=len(rep),
                loaded_bytes=sum(v["bytes"] for v in rep.values()), skipped_quantized=sorted(skipped), kept_from_teacher_types=sorted(set(t["tname"] for t in T if t["name"] not in rep)),
                min_cos=min(v["cos"] for v in rep.values()), tensors=rep, seconds=time.time() - t0)
