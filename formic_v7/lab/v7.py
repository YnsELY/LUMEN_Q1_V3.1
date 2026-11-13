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
    S["arki"] = dict(tok=tok, sc=sc, teach=hs, dom=[dom[i] for i in range(tok.shape[0])])
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
    allk = torch.cat(kls); alla = torch.cat(agree)
    bydom = {}
    for d in sorted(set(A["dom"])):
        k = torch.cat([kls[i] for i in range(len(kls)) if A["dom"][i] == d]); a = torch.cat([agree[i] for i in range(len(kls)) if A["dom"][i] == d])
        bydom[d] = dict(kl=float(k.mean()), agree=float(a.float().mean()), n=int(k.numel()))
    return dict(protocol="P-A108", positions=int(allk.numel()), kl=float(allk.mean()), kl_p99=float(torch.quantile(allk[torch.randperm(allk.numel())[:150000]].double(), 0.99)) if allk.numel() > 150000 else float(torch.quantile(allk.double(), 0.99)),
                agree=float(alla.float().mean()), by_domain=bydom, per_window=perw)


def arki_short(r):
    return "P-A108 KL=%.4f p99=%.3f top1=%.2f%% | " % (r["kl"], r["kl_p99"], 100 * r["agree"]) + " ".join("%s:%.3f" % (k[:6], v["kl"]) for k, v in r["by_domain"].items())


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
            q = torch.from_numpy(a).cuda().float(); w = W.float()
            if q.shape != w.shape: raise RuntimeError(f"shape mismatch {name} {tuple(q.shape)} vs {tuple(w.shape)}")
            rep[name] = dict(params=int(w.numel()), cos=float((q * w).sum() / (q.norm() * w.norm())), rel_err=float((q - w).norm() / w.norm()))
            W.copy_(q.to(W.dtype)); del q, w
    man = json.load(open(tmp + "/_manifest.json"))
    for k, v in rep.items(): v.update(type=man["tensors"][k]["type"], bytes=man["tensors"][k]["bytes"])
    shutil.rmtree(tmp, ignore_errors=True)
    bad = {k: v["cos"] for k, v in rep.items() if v["cos"] < 0.80}
    tot_b = sum(v["bytes"] for v in rep.values()); tot_p = sum(v["params"] for v in rep.values())
    return dict(file=os.path.basename(path), file_bytes=os.path.getsize(path), loaded=len(rep), loaded_bytes=tot_b, loaded_params=tot_p, bpw_loaded=8 * tot_b / tot_p,
                kept_bf16=man["kept_unquantized"], skipped=skipped, low_cosine=bad, tensors=rep, seconds=time.time() - t0)
