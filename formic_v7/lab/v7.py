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


# ----------------------------------------------------------------------------- real package (formic_v7/codec)
CODEC = "/workspace/LUMEN_Q1_V3.1/formic_v7/codec"
PKG_ROOT = "/workspace/work/packages"


def export_begin(vid, workers=6):
    """Start CPU encoder workers; from now on every quantized matrix is also written to the package <vid>."""
    import subprocess, shutil
    ind = f"/workspace/work/export_in/{vid}"; pkg = f"{PKG_ROOT}/{vid}"
    os.makedirs(ind, exist_ok=True); os.makedirs(pkg, exist_ok=True)
    if os.path.exists(ind + "/_END"): os.remove(ind + "/_END")
    for f in os.listdir(ind):                               # files claimed by workers of a crashed run
        if ".claimed." in f: os.rename(os.path.join(ind, f), os.path.join(ind, f.split(".claimed.")[0]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen(["python3", "-I", CODEC + "/package.py", "worker", ind, pkg], env=env,
                              stdout=open(f"{ind}/_worker{i}.log", "a"), stderr=subprocess.STDOUT) for i in range(workers)]
    H.S["export"] = {"in": ind, "pkg": pkg, "procs": procs}
    return pkg


def export_end():
    """Stop the workers after they have encoded everything; add the small unquantized text-path tensors (FP16)."""
    import sys
    ex = H.S.pop("export"); open(ex["in"] + "/_END", "w").write("end")
    for p in ex["procs"]: p.wait()
    left = [f for f in os.listdir(ex["in"]) if f.endswith(".npz") or ".claimed." in f]
    if left: raise RuntimeError("export incomplete: %s" % left[:5])
    if CODEC not in sys.path: sys.path.insert(0, CODEC)
    import package
    qnames = set(id(m["mod"].weight) for m in H.qmats()); nraw = 0
    for name, p in H.S["lm"].named_parameters():
        if id(p) in qnames: continue
        package.write_raw(ex["pkg"], "raw." + name, p.detach().float().cpu().numpy().astype("float16")); nraw += p.numel()
    man = package.load_manifest(ex["pkg"])
    e8 = {k: v for k, v in man.items() if v.get("kind") == "e8"}
    return dict(pkg=ex["pkg"], package_bytes=package.package_bytes(ex["pkg"]), e8_files=len(e8), e8_bytes=sum(v["bytes"] for v in e8.values()),
                e8_params=sum(v["shape"][0] * v["shape"][1] for v in e8.values()), raw_params_fp16=nraw)


@torch.no_grad()
def load_package(pkg, tmp="/workspace/work/_pkgdec"):
    """Rebuild the text path FROM THE FILES of a package (independent CPU decoder process), on top of BF16 weights
    for everything that is not in the package.  Returns the number of matrices loaded."""
    import subprocess, glob, shutil
    H.restore(); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    proc = subprocess.Popen(["python3", "-I", CODEC + "/decode_stream.py", pkg, tmp, "12", "5"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                            stdout=open(tmp + "/_stdout.log", "w"), stderr=subprocess.STDOUT)
    by = {m["name"]: m for m in H.qmats()}; n = 0
    while True:
        files = sorted(glob.glob(tmp + "/*.npy")); done = os.path.exists(tmp + "/_DONE")
        if not files:
            if done: break
            if proc.poll() is not None and not os.path.exists(tmp + "/_DONE"): raise RuntimeError("decoder failed: " + open(tmp + "/_stdout.log").read()[-800:])
            time.sleep(0.3); continue
        for f in files:
            name = os.path.basename(f)[:-4]; a = np.load(f); os.remove(f)
            if name.startswith("mtp."): continue                       # not loadable in this runtime (not evaluated)
            if name.startswith("visual."):
                P = dict(H.S["model"].model.visual.named_parameters())[name[7:]]
                P.copy_(torch.from_numpy(a).cuda().reshape(P.shape).to(P.dtype)); n += 1; continue
            W = by[name]["mod"].weight
            for j in range(0, a.shape[0], 65536): W[j:j + 65536] = torch.from_numpy(a[j:j + 65536]).cuda().to(W.dtype)
            n += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return n


def run_exported(vid, spec, mult, method="", prepare=None, workers=6):
    """run() + real package: codes are encoded to files during the pass; the files are then decoded by a separate
    process and the model REBUILT FROM THEM is evaluated again (must reproduce the in-memory numbers)."""
    pkg = export_begin(vid, workers)
    try:
        info, dev = run(vid, spec, mult, method=method, prepare=prepare)
    except BaseException:
        ex = H.S.pop("export", None)
        if ex:
            for p in ex["procs"]: p.terminate()
        raise
    import shutil
    shutil.rmtree(f"/workspace/work/ckpt/{vid}", ignore_errors=True)       # results are written; free 52 GB before decoding
    rep = export_end()
    print("[%s] package: %d E8 files, %.6f GB for %.3f G weights = %.4f bpw REAL (ideal accounting said %.4f); whole text package %.6f GB" % (
        vid, rep["e8_files"], rep["e8_bytes"] / 1e9, rep["e8_params"] / 1e9, 8 * rep["e8_bytes"] / rep["e8_params"], info["bpw"], rep["package_bytes"] / 1e9), flush=True)
    t = time.time(); n = load_package(pkg)
    rb = H.evaluate("big"); ra = arki_eval()
    mem = json.load(open(OUT7 + f"/big_{vid}.json"))["big"]["kl"]; memd = json.load(open(OUT7 + f"/arkidev_{vid}.json"))["kl"]
    rep.update(decoded_matrices=n, decode_eval_seconds=time.time() - t, big_kl_from_files=rb["kl"], big_kl_in_memory=mem, arkidev_kl_from_files=ra["dev"]["kl"], arkidev_kl_in_memory=memd)
    json.dump(rep, open(OUT7 + f"/package_{vid}.json", "w"), indent=1)
    print("[%s] REBUILT FROM FILES (%d matrices): big-600 KL %.6f (in memory %.6f) ; P-A108-dev KL %.6f (in memory %.6f)" % (vid, n, rb["kl"], mem, ra["dev"]["kl"], memd), flush=True)
    return info, dev, rep


# ----------------------------------------------------------------------------- Vision tower and MTP block in the package
@torch.no_grad()
def _e8_rtn_to_pkg(W2, bpw, pkg, name):
    """No-feedback E8 at a requested ideal rate, written to the package with the real codec. Returns (Q, bytes)."""
    import sys, formic_q as fq
    if CODEC not in sys.path: sys.path.insert(0, CODEC)
    import package
    W2 = W2.float(); m, n = W2.shape; lo, hi = 0.002, 3.0
    for _ in range(20):
        mid = math.sqrt(lo * hi); r = fq.row_steps(W2, mid); Q, hist = fq.quantize_rtn(W2, r)
        if fq.matrix_bits(hist, m, n) > bpw * m * n: lo = mid
        else: hi = mid
    r = fq.row_steps(W2, math.sqrt(lo * hi)); Q, hist = fq.quantize_rtn(W2, r)
    U = torch.round(2.0 * Q / r[:, None])
    assert torch.equal(U * (r[:, None] / 2.0), Q)
    amax = float(U.abs().max()); dt = np.int8 if amax <= 127 else (np.int16 if amax <= 32767 else np.int32)
    nb = package.write_tensor(pkg, name, U.cpu().numpy().astype(dt), r.cpu().numpy().astype(np.float16), None)
    return Q, nb


@torch.no_grad()
def package_vision(pkg, bpw):
    """Quantize the vision tower (all weight tensors with >= 65536 elements and a multiple of 8 columns) into the
    package and write the decoded values into the model; small tensors are stored FP16."""
    import sys
    if CODEC not in sys.path: sys.path.insert(0, CODEC)
    import package
    vis = H.S["model"].model.visual; nb = 0; nq = 0; nraw = 0
    for name, p in vis.named_parameters():
        W2 = p.reshape(p.shape[0], -1) if p.dim() >= 2 else None
        if W2 is not None and W2.numel() >= 65536 and W2.shape[1] % 8 == 0:
            Q, b = _e8_rtn_to_pkg(W2, bpw, pkg, "visual." + name); p.copy_(Q.reshape(p.shape).to(p.dtype)); nb += b; nq += p.numel()
        else:
            package.write_raw(pkg, "raw.visual." + name, p.detach().float().cpu().numpy().astype("float16")); nraw += p.numel()
    return dict(e8_bytes=nb, e8_params=nq, raw_params=nraw, bpw=8 * nb / nq)


@torch.no_grad()
def package_mtp(pkg, bpw):
    """MTP block: NOT loaded by this runtime, so it is quantized straight from the checkpoint and NOT evaluated."""
    import sys
    from safetensors import safe_open
    if CODEC not in sys.path: sys.path.insert(0, CODEC)
    import package
    idx = json.load(open(os.path.join(H.MODEL_DIR, "model.safetensors.index.json")))["weight_map"]; nb = 0; nq = 0; nraw = 0
    for k in sorted(k for k in idx if k.startswith("mtp.")):
        with safe_open(os.path.join(H.MODEL_DIR, idx[k]), "pt", device="cuda") as f: p = f.get_tensor(k)
        W2 = p.reshape(p.shape[0], -1) if p.dim() >= 2 else None
        if W2 is not None and W2.numel() >= 65536 and W2.shape[1] % 8 == 0:
            Q, b = _e8_rtn_to_pkg(W2, bpw, pkg, k); nb += b; nq += p.numel()
        else:
            package.write_raw(pkg, "raw." + k, p.float().cpu().numpy().astype("float16")); nraw += p.numel()
        del p
    return dict(e8_bytes=nb, e8_params=nq, raw_params=nraw, bpw=8 * nb / nq)


def package_aux_files(pkg):
    """Tokenizer, chat template, configs: copied verbatim; they are part of the product bytes."""
    import shutil
    n = 0
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
              "preprocessor_config.json", "video_preprocessor_config.json"):
        src = os.path.join(H.MODEL_DIR, f)
        if os.path.exists(src): shutil.copy(src, os.path.join(pkg, "aux." + f)); n += os.path.getsize(src)
    return n


@torch.no_grad()
def vision_teacher(path="/workspace/work/data/vision/vision_inputs.pt", max_new=64):
    """BF16 teacher continuations for the image prompts (greedy), cached with their log-probabilities on CPU."""
    data = torch.load(path); model = H.S["model"]; dt = model.dtype; T = []
    for s in data:
        inp = {k: v.cuda() for k, v in s["inputs"].items()}; inp["pixel_values"] = inp["pixel_values"].to(dt); L = inp["input_ids"].shape[1]
        full = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        lg = model(**_vis_inputs(s, full, dt)).logits[0, L - 1:-1].float()
        T.append(dict(full=full.cpu(), L=L, lp=F.log_softmax(lg, -1).to(torch.float16).cpu(), kind=s["kind"]))
    H.S["vis"] = dict(data=data, T=T)
    return len(T)


def _vis_inputs(s, ids, dt):
    inp = {k: v.cuda() for k, v in s["inputs"].items()}; inp["pixel_values"] = inp["pixel_values"].to(dt)
    extra = ids.shape[1] - inp["input_ids"].shape[1]
    if "mm_token_type_ids" in inp and extra > 0:
        tt = inp["mm_token_type_ids"]; inp["mm_token_type_ids"] = torch.cat([tt, torch.zeros(1, extra, dtype=tt.dtype, device=tt.device)], 1)
    inp["input_ids"] = ids.cuda(); inp["attention_mask"] = torch.ones_like(inp["input_ids"])
    return inp


@torch.no_grad()
def vision_eval():
    V_ = H.S["vis"]; model = H.S["model"]; out = {}
    for s, tt in zip(V_["data"], V_["T"]):
        lg = model(**_vis_inputs(s, tt["full"], model.dtype)).logits[0, tt["L"] - 1:-1].float()
        lps = F.log_softmax(lg, -1); lpt = tt["lp"].cuda().float()
        out.setdefault(tt["kind"], []).append(((lpt.exp() * (lpt - lps)).sum(-1).mean().item(), (lps.argmax(-1) == lpt.argmax(-1)).float().mean().item()))
    allv = [v for vs in out.values() for v in vs]
    return dict(kl=sum(v[0] for v in allv) / len(allv), agree=sum(v[1] for v in allv) / len(allv), n=len(allv),
                by_kind={k: dict(kl=sum(v[0] for v in vs) / len(vs), agree=sum(v[1] for v in vs) / len(vs)) for k, vs in out.items()})
