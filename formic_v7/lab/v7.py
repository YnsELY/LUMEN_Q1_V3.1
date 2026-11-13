"""V7 campaign helpers (run inside the lab server): competitor GGUF -> our graph, and the public Arki05 protocol.

Protocol P-A108 = the 108 redistributed windows (of 128) of Arki05/Qwen3.8-27B-GGUF/eval (rev cdcbea9996):
4096-token windows read from an empty context, positions 2048..4094 scored where scored.u8 == 1 (191 486 positions),
KL(BF16 || candidate) over the full vocabulary in nats, token-level mean, 99th percentile, same-top-token share.
It is NOT numerically comparable with the published 128-window table (20 windows missing: science, part of code)."""
import os, json, time, math
import numpy as np, torch, torch.nn.functional as F
import harness as H

ARKI = "/workspace/work/competitors/arki05/eval"


@torch.no_grad()
def arki_teacher():
    """Cache BF16 teacher hidden states at the scored positions (call with BF16 weights loaded)."""
    S = H.S; lm = S["lm"]
    tok = torch.from_numpy(np.fromfile(ARKI + "/tokens.i32", "<i4").reshape(-1, 4096).astype(np.int64))
    sc = torch.from_numpy(np.fromfile(ARKI + "/scored.u8", np.uint8).reshape(-1, 4096).astype(bool))
    ws = json.load(open(ARKI + "/windows.json"))["windows"]
    dom = {w["row"]: w["source"] for w in ws if w.get("row") is not None}
    hs = []
    for i in range(tok.shape[0]):
        h = lm(input_ids=tok[i:i + 1].cuda()).last_hidden_state[0]
        hs.append(h[sc[i].cuda()].cpu())
    doms = [dom[i] for i in range(tok.shape[0])]
    # development / confirmation halves, stratified by domain (alternating rows inside each domain)
    seen = {}; part = []
    for d in doms:
        part.append("dev" if seen.get(d, 0) % 2 == 0 else "conf"); seen[d] = seen.get(d, 0) + 1
    S["arki"] = dict(tok=tok, sc=sc, teach=hs, dom=doms, part=part)
    return tok.shape, int(sc.sum())


@torch.no_grad()
def arki_eval(chunk=1024):
    S = H.S; lm = S["lm"]; model = S["model"]; A = S["arki"]
    kls = []; agree = []; perw = []
    for i in range(A["tok"].shape[0]):
        h = lm(input_ids=A["tok"][i:i + 1].cuda()).last_hidden_state[0][A["sc"][i].cuda()]
        ht = A["teach"][i]; kw = []; aw = []
        for j in range(0, h.shape[0], chunk):
            lps = F.log_softmax(model.lm_head(h[j:j + chunk]).float(), -1)
            lpt = F.log_softmax(F.linear(ht[j:j + chunk].cuda(), S["T_head"]).float(), -1)
            kw.append((lpt.exp() * (lpt - lps)).sum(-1).cpu()); aw.append((lps.argmax(-1) == lpt.argmax(-1)).cpu())
            del lps, lpt
        kw = torch.cat(kw); aw = torch.cat(aw); kls.append(kw); agree.append(aw)
        perw.append(dict(row=i, domain=A["dom"][i], n=int(kw.numel()), kl=float(kw.mean()) if kw.numel() else 0.0, agree=float(aw.float().mean()) if aw.numel() else 0.0))
    if "part" not in A:
        seen = {}; A["part"] = []
        for d in A["dom"]:
            A["part"].append("dev" if seen.get(d, 0) % 2 == 0 else "conf"); seen[d] = seen.get(d, 0) + 1
    for w in perw: w["part"] = A["part"][w["row"]]
    def summ(rows, name):
        allk = torch.cat([kls[i] for i in rows]); alla = torch.cat([agree[i] for i in rows]); bydom = {}
        for d in sorted(set(A["dom"])):
            rr = [i for i in rows if A["dom"][i] == d]
            if not rr: continue
            k = torch.cat([kls[i] for i in rr]); a = torch.cat([agree[i] for i in rr])
            bydom[d] = dict(kl=float(k.mean()), agree=float(a.float().mean()), n=int(k.numel()))
        g = torch.Generator().manual_seed(0); sub = allk[torch.randperm(allk.numel(), generator=g)[:150000]]
        return dict(protocol=name, positions=int(allk.numel()), kl=float(allk.mean()), kl_p99=float(torch.quantile(sub.double(), 0.99)),
                    agree=float(alla.float().mean()), by_domain=bydom, per_window=[perw[i] for i in rows])
    n = len(kls)
    full = summ(list(range(n)), "P-A108")
    full["dev"] = summ([i for i in range(n) if A["part"][i] == "dev"], "P-A108-dev")
    full["conf"] = summ([i for i in range(n) if A["part"][i] == "conf"], "P-A108-conf")
    return full


def arki_short(r):
    return "%s KL=%.4f p99=%.3f top1=%.2f%% | " % (r["protocol"], r["kl"], r["kl_p99"], 100 * r["agree"]) + " ".join("%s:%.3f" % (k[:6], v["kl"]) for k, v in r["by_domain"].items())


# ----------------------------------------------------------------------------- competitor weights
def _target(name):
    """GGUF tensor name -> (module weight parameter) of our HF graph, or None (kept from BF16)."""
    lm = H.S["lm"]; model = H.S["model"]
    if name == "token_embd.weight": return lm.embed_tokens.weight
    if name == "output.weight": return model.lm_head.weight
    if not name.startswith("blk."): return None
    p = name.split("."); i = int(p[1]); k = p[2]; L = lm.layers[i]
    if k in ("ffn_gate", "ffn_up", "ffn_down"): return getattr(L.mlp, {"ffn_gate": "gate_proj", "ffn_up": "up_proj", "ffn_down": "down_proj"}[k]).weight
    if hasattr(L, "linear_attn"):
        m = {"attn_qkv": "in_proj_qkv", "attn_gate": "in_proj_z", "ssm_out": "out_proj", "ssm_alpha": "in_proj_a", "ssm_beta": "in_proj_b"}
        return getattr(L.linear_attn, m[k]).weight if k in m else None
    m = {"attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj", "attn_output": "o_proj"}
    return getattr(L.self_attn, m[k]).weight if k in m else None


# llama.cpp stores the 48 value heads of a Gated DeltaNet layer in "tiled" order (q/k heads are tiled, where HF uses
# repeat_interleave): GGUF head i = HF head PERM[i], PERM = [0,3,...,45, 1,4,...,46, 2,5,...,47]. Measured on the real
# file: cosine 0.94-1.00 after matching, 0.04 before. This affects the v rows of attn_qkv, the rows of attn_gate,
# ssm_alpha, ssm_beta, and the columns of ssm_out.
_PERM = [3 * j + r for r in range(3) for j in range(16)]
_INV = [0] * 48
for _i, _h in enumerate(_PERM): _INV[_h] = _i


def _to_hf_order(name, q):
    """q: dequantized GGUF tensor (torch, on GPU) -> same tensor in HF head order."""
    k = name.split(".")[2] if name.startswith("blk.") else ""
    if k not in ("attn_qkv", "attn_gate", "ssm_out", "ssm_alpha", "ssm_beta"): return q
    inv = torch.tensor(_INV, device=q.device)
    rows128 = (inv[:, None] * 128 + torch.arange(128, device=q.device)[None, :]).reshape(-1)     # HF row -> GGUF row
    if k == "attn_qkv":
        v = q[4096:]; return torch.cat([q[:4096], v[rows128]])
    if k == "attn_gate": return q[rows128]
    if k == "ssm_out": return q[:, rows128]
    return q[inv]


@torch.no_grad()
def load_gguf(path, workers=16, tmp="/workspace/work/competitors/_deq"):
    """Replace every quantized weight matrix of the text path by the competitor's dequantized values.
    A separate CPU process (lab/gguf_dequant.py) dequantizes to .npy files; they are consumed and deleted here.
    Returns a per-tensor report (type, bytes, cosine with BF16, relative error); unquantized tensors keep BF16."""
    import subprocess, glob, shutil
    H.restore(); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.Popen(["python3", "-I", "/workspace/LUMEN_Q1_V3.1/formic_v7/lab/gguf_dequant.py", path, tmp, str(workers), "8"], env=env,
                            stdout=open(tmp + "/_stdout.log", "w"), stderr=subprocess.STDOUT)
    rep = {}; skipped = {}; t0 = time.time()
    while True:
        files = sorted(glob.glob(tmp + "/*.npy")); done = os.path.exists(tmp + "/_manifest.json")
        if not files:
            if done: break
            if proc.poll() is not None and not os.path.exists(tmp + "/_manifest.json"):
                raise RuntimeError("dequantizer failed: " + open(tmp + "/_stdout.log").read()[-800:])
            time.sleep(0.3); continue
        for f in files:
            name = os.path.basename(f)[:-4]; a = np.load(f); os.remove(f)
            W = _target(name)
            if W is None: skipped[name] = "no target"; continue
            q = _to_hf_order(name, torch.from_numpy(a).cuda().float()); w = W.float()
            if q.shape != w.shape: raise RuntimeError(f"shape mismatch {name} {tuple(q.shape)} vs {tuple(w.shape)}")
            rep[name] = dict(params=int(w.numel()), cos=float((q * w).sum() / (q.norm() * w.norm())), rel_err=float((q - w).norm() / w.norm()))
            W.copy_(q.to(W.dtype)); del q, w
    man = json.load(open(tmp + "/_manifest.json"))
    for k, v in rep.items(): v.update(type=man["tensors"][k]["type"], bytes=man["tensors"][k]["bytes"])
    shutil.rmtree(tmp, ignore_errors=True)
    bad = {k: v["cos"] for k, v in rep.items() if v["cos"] < 0.80}
    if bad: raise RuntimeError("tensor correspondence not validated (cosine < 0.80): %s" % dict(list(bad.items())[:6]))
    tot_b = sum(v["bytes"] for v in rep.values()); tot_p = sum(v["params"] for v in rep.values())
    return dict(file=os.path.basename(path), file_bytes=os.path.getsize(path), loaded=len(rep), loaded_bytes=tot_b, loaded_params=tot_p, bpw_loaded=8 * tot_b / tot_p,
                kept_bf16=man["kept_unquantized"], skipped=skipped, low_cosine=bad, tensors=rep, seconds=time.time() - t0)


# ----------------------------------------------------------------------------- V7 variant runner
OUT7 = "/workspace/LUMEN_Q1_V3.1/formic_v7/results"


def run(vid, spec, mult, method="", prepare=None):
    """Full crash-safe variant: resumable E8-LDLQ pass, then every development protocol:
    select / confirm (both DEVELOPMENT data in V7), big-600, long-8192 and the public P-A108 windows."""
    import seq as SQ
    H.OUT = OUT7
    info = SQ.run_ldlq_variant(vid, spec, mult, method=method, prepare=prepare)
    ra = arki_eval(); meta = dict(bpw=info["bpw"], bits=info["bits"], bytes=info["bits"] / 8)
    # development half is printed and saved in the repo; the confirmation half (and the full set) go to the sealed
    # directory and are only read after the finalists are frozen
    dev = dict(ra["dev"], info=meta); json.dump(dev, open(OUT7 + f"/arkidev_{vid}.json", "w"))
    json.dump(dict(conf=ra["conf"], full={k: v for k, v in ra.items() if k not in ("dev", "conf")}, info=meta), open(H.SEALED + f"/arki_{vid}.json", "w"))
    print("[%s] %s  (confirmation half sealed)" % (vid, arki_short(dev)), flush=True)
    return info, dev


def arki_alloc(path="/workspace/work/competitors/arki05/Qwen3.8-27B-2.82bpw.gguf"):
    """Per-matrix bits/weight actually spent by a GGUF file (bytes*8/params from its header), keyed by OUR names."""
    import gguf
    r = gguf.GGUFReader(path); out = {}
    hf = {"ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "attn_qkv": "linear_attn.in_proj_qkv",
          "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj", "ssm_alpha": "linear_attn.in_proj_a", "ssm_beta": "linear_attn.in_proj_b",
          "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj"}
    for t in r.tensors:
        if t.tensor_type.name in ("F32", "F16", "BF16"): continue
        b = 8.0 * int(t.n_bytes) / int(t.n_elements)
        if t.name == "token_embd.weight": out["embed_tokens"] = b
        elif t.name == "output.weight": out["lm_head"] = b
        else:
            p = t.name.split(".")
            if p[2] in hf: out[f"layers.{p[1]}.{hf[p[2]]}"] = b
    return out
