"""Experiment harness living inside the lab server namespace (model stays on GPU).

init(model, tok) must be called once.  All evaluation is text-only, teacher = BF16 model.
Metrics: KL(teacher || student) per token (nats, exact over the full vocabulary), top-1
agreement, cross-entropy on the real next token.  None of these is a capability percentage.
"""
import os, json, time, math, glob, contextlib
import torch
import torch.nn.functional as F
from safetensors import safe_open
import formic_q as fq

MODEL_DIR = "/root/models/qwen3.8-27b"
DATA_PT = "/workspace/work/data/corpus_tokens.pt"
OUT = "/workspace/LUMEN_Q1_V3.1/formic_arch/results"
SEALED = "/workspace/work/sealed"          # confirm results are written here, never printed
S = {}                                      # shared state


def init(model, tok):
    S["model"] = model; S["tok"] = tok; S["lm"] = model.model.language_model
    d = torch.load(DATA_PT)
    S["data"] = {k: d[k] for k in ("cal", "select", "confirm")}
    S["meta"] = d["meta"]
    os.makedirs(OUT, exist_ok=True); os.makedirs(SEALED, exist_ok=True)


# ----------------------------------------------------------------------------- weights
def restore(names=None):
    """Reload original BF16 weights from the checkpoint shards, in place (all, or given names)."""
    model = S["model"]; sd = dict(model.named_parameters())
    idx = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    want = set(sd) if names is None else set(names)
    byshard = {}
    for k in want:
        byshard.setdefault(idx[k], []).append(k)
    n = 0
    with torch.no_grad():
        for shard, keys in byshard.items():
            with safe_open(os.path.join(MODEL_DIR, shard), "pt", device="cuda") as f:
                for k in keys:
                    sd[k].copy_(f.get_tensor(k)); n += 1
    S["dirty"] = set()
    return n


def qmats(include_embed=True, include_head=True):
    """All quantizable 2-D matrices of the text path: list of dicts(name, mod, kind, layer, ltype)."""
    lm = S["lm"]; out = []
    for i, L in enumerate(lm.layers):
        if hasattr(L, "linear_attn"):
            a = L.linear_attn
            for k in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                out.append(dict(name=f"layers.{i}.linear_attn.{k}", mod=getattr(a, k), kind=k, layer=i, ltype="gdn"))
        else:
            a = L.self_attn
            for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
                out.append(dict(name=f"layers.{i}.self_attn.{k}", mod=getattr(a, k), kind=k, layer=i, ltype="attn"))
        for k in ("gate_proj", "up_proj", "down_proj"):
            out.append(dict(name=f"layers.{i}.mlp.{k}", mod=getattr(L.mlp, k), kind=k, layer=i,
                            ltype="gdn" if hasattr(L, "linear_attn") else "attn"))
    if include_embed:
        out.append(dict(name="embed_tokens", mod=lm.embed_tokens, kind="embed", layer=-1, ltype="io"))
    if include_head:
        out.append(dict(name="lm_head", mod=S["model"].lm_head, kind="lm_head", layer=64, ltype="io"))
    return out


# ----------------------------------------------------------------------------- teacher / eval
@torch.no_grad()
def build_teacher():
    """Cache teacher final hidden states (post final norm) + a frozen BF16 copy of lm_head."""
    lm = S["lm"]; S["T_head"] = S["model"].lm_head.weight.detach().clone()
    S["teach"] = {}
    for split in ("select", "confirm"):
        ids = S["data"][split]; hs = []
        for i in range(0, ids.shape[0], 8):
            hs.append(lm(input_ids=ids[i:i + 8].cuda()).last_hidden_state)
        S["teach"][split] = torch.cat(hs)
    return {k: tuple(v.shape) for k, v in S["teach"].items()}


@torch.no_grad()
def evaluate(split="select", docs=None, bs=8):
    """Exact KL(teacher||student) etc. Returns dict with per-doc lists. docs: optional index list."""
    lm = S["lm"]; model = S["model"]
    ids_all = S["data"][split]; Ht = S["teach"][split]
    idx = list(range(ids_all.shape[0])) if docs is None else list(docs)
    per = dict(kl=[], agree=[], ce_s=[], ce_t=[], klmax=[])
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = ids_all[b].cuda()
        hs = lm(input_ids=x).last_hidden_state
        for k in range(len(b)):
            ls = model.lm_head(hs[k:k + 1]).float()[0]
            lt = F.linear(Ht[b[k]:b[k] + 1], S["T_head"]).float()[0]
            lps = F.log_softmax(ls, -1); del ls
            lpt = F.log_softmax(lt, -1); del lt
            kl = (lpt.exp() * (lpt - lps)).sum(-1)
            per["kl"].append(kl.mean().item()); per["klmax"].append(kl.max().item())
            per["agree"].append((lps.argmax(-1) == lpt.argmax(-1)).float().mean().item())
            tgt = x[k, 1:, None]
            per["ce_s"].append(-lps[:-1].gather(-1, tgt).mean().item())
            per["ce_t"].append(-lpt[:-1].gather(-1, tgt).mean().item())
            del lps, lpt, kl
    meta = [S["meta"][split][j] for j in idx]
    res = dict(split=split, n_docs=len(idx), per_doc=per, doc_ids=[m["id"] for m in meta],
               domains=[m["domain"] for m in meta])
    nan = lambda v: sum(v) / len(v)
    res["kl"] = nan(per["kl"]); res["agree"] = nan(per["agree"])
    res["ce_s"] = nan(per["ce_s"]); res["ce_t"] = nan(per["ce_t"])
    res["ppl_s"] = math.exp(res["ce_s"]); res["ppl_t"] = math.exp(res["ce_t"])
    bydom = {}
    for dname in sorted(set(res["domains"])):
        sel = [j for j, dd in enumerate(res["domains"]) if dd == dname]
        bydom[dname] = dict(kl=nan([per["kl"][j] for j in sel]), agree=nan([per["agree"][j] for j in sel]),
                            ce_s=nan([per["ce_s"][j] for j in sel]), ce_t=nan([per["ce_t"][j] for j in sel]))
    res["by_domain"] = bydom
    return res


def short(res):
    return "KL=%.5f agree=%.4f ppl_s=%.4f ppl_t=%.4f | " % (res["kl"], res["agree"], res["ppl_s"], res["ppl_t"]) + \
        " ".join("%s:%.4f" % (k[:5], v["kl"]) for k, v in res["by_domain"].items())


def run_variant(vid, info=None, select_docs=None):
    """Evaluate current model state on select (printed + saved) and confirm (SEALED, not printed)."""
    t = time.time()
    rs = evaluate("select", docs=select_docs)
    rec = dict(id=vid, info=info or {}, select=rs, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(rec, open(os.path.join(OUT, f"variant_{vid}.json"), "w"))
    rc = evaluate("confirm")
    json.dump(dict(id=vid, info=info or {}, confirm=rc), open(os.path.join(SEALED, f"{vid}.json"), "w"))
    print("[%s] SELECT %s  (confirm sealed; %.0fs)" % (vid, short(rs), time.time() - t), flush=True)
    return rs


@contextlib.contextmanager
def skip_layers(idxs):
    lm = S["lm"]
    try:
        for i in idxs:
            lm.layers[i].forward = (lambda hidden_states, **kw: hidden_states)
        yield
    finally:
        for i in idxs:
            if "forward" in lm.layers[i].__dict__: del lm.layers[i].forward


# ----------------------------------------------------------------------------- statistics
def collect_stats(n_docs=144, T=512, seed=0):
    """One forward+backward pass per calibration doc (BF16 teacher weights):
       X[name] = E[x_j^2] of the linear input, G[name] = E[g_i^2] of d(loss)/d(output),
       loss = CE against labels SAMPLED from the model itself (Fisher), summed over positions.
       For down_proj also the Taylor saliency  sum_t (a_ti * dL/da_ti)^2  per channel."""
    lm = S["lm"]; model = S["model"]
    mats = qmats(include_embed=False)
    X = {}; G = {}; TS = {}; hooks = []; cur = {}
    def fpre(name):
        def h(mod, inp):
            x = inp[0].detach()
            v = x.float().pow(2).sum(dim=tuple(range(x.dim() - 1)))
            X[name] = X.get(name, 0) + v.double()
            if name.endswith("down_proj"): cur[name] = x
        return h
    def bwd(name):
        def h(mod, gin, gout):
            g = gout[0].detach().float()
            G[name] = G.get(name, 0) + g.pow(2).sum(dim=tuple(range(g.dim() - 1))).double()
            if name.endswith("down_proj") and gin[0] is not None:
                a = cur.pop(name).float(); ga = gin[0].detach().float()
                TS[name] = TS.get(name, 0) + (a * ga).pow(2).sum(dim=(0, 1)).double()
        return h
    for m in mats:
        hooks.append(m["mod"].register_forward_pre_hook(fpre(m["name"])))
        hooks.append(m["mod"].register_full_backward_hook(bwd(m["name"])))
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    ids = S["data"]["cal"][:n_docs, :T]; ntok = 0
    try:
        for i in range(ids.shape[0]):
            x = ids[i:i + 1].cuda()
            emb = lm.embed_tokens(x).detach().requires_grad_(True)
            with torch.enable_grad():
                h = lm(inputs_embeds=emb).last_hidden_state
                logits = model.lm_head(h).float()[0]
                y = torch.multinomial(F.softmax(logits.detach(), -1), 1, generator=gen)[:, 0]
                loss = F.cross_entropy(logits, y, reduction="sum")
                loss.backward()
            ntok += x.numel()
            del emb, h, logits, loss
    finally:
        for hk in hooks: hk.remove()
    st = dict(X={k: (v / ntok).float().cpu() for k, v in X.items()},
              G={k: (v / ntok).float().cpu() for k, v in G.items()},
              TS={k: (v / ntok).float().cpu() for k, v in TS.items()}, ntok=ntok, T=T, n_docs=n_docs)
    return st


# ----------------------------------------------------------------------------- quantization of the model
def _scales(m, W, spec, mult, st):
    """Return (r [rows], c [cols] or None) for matrix dict m, weight W (possibly row/col subset)."""
    rule = spec.get("rule", "lumen")
    kind = m["kind"]
    if kind == "embed" or rule == "lumen" or (kind == "lm_head" and rule in ("fisher_global",)):
        if kind == "lm_head" and rule == "fisher_global" and spec.get("head_col", True):
            rule = "col"
        else:
            mm = spec.get("io_mult", mult) if kind in ("embed", "lm_head") else mult
            return fq.row_steps(W, mm), None
    name = m["name"]
    X = st["X"][name].cuda().float()
    if "cols" in m: X = X[m["cols"]]
    ac = spec.get("alpha_col", 1.0)
    c = (X.clamp_min(X.mean() * 1e-6)).pow(ac / 2)
    c = fq.fp16(c / c.log().mean().exp())                       # geometric mean 1, FP16-rounded
    if rule == "col" or kind == "lm_head":
        mm = spec.get("io_mult", mult) if kind == "lm_head" else mult
        return fq.row_steps(W.float() * c, mm), c
    G = st["G"][name].cuda().float()
    if "rows" in m: G = G[m["rows"]]
    ar = spec.get("alpha_row", 1.0)
    G = G.clamp_min(G.median() * spec.get("g_floor", 1e-4))
    if rule == "fisher_local":
        rho = G.pow(-ar / 2)
        Wn = W.float() * c / rho[:, None]
        rho = rho * Wn.pow(2).mean().sqrt()                     # overall RMS of normalized matrix = 1
        return fq.fp16(mult * rho).clamp_min(6e-8), c
    if rule == "fisher_global":
        Xr = X.clamp_min(X.mean() * 1e-6).sqrt()               # true (unnormalised) column scale
        c = fq.fp16(Xr)
        return fq.fp16(mult * G.pow(-0.5)).clamp_min(6e-8), c
    raise ValueError(rule)


@torch.no_grad()
def quantize_matrix(m, spec, mult, st=None, sample=None, write=True, H=None):
    """Quantize one matrix in place (or just measure bits on a row sample). Returns (bits, n_params)."""
    W = m["mod"].weight
    rows = m.get("rows"); cols = m.get("cols")
    Wsub = W
    if rows is not None: Wsub = Wsub[rows]
    if cols is not None: Wsub = Wsub[:, cols]
    r, c = _scales(m, Wsub, spec, mult, st)
    if sample:
        sel = torch.arange(0, Wsub.shape[0], sample, device=W.device)
        _, hist = fq.quantize_rtn(Wsub[sel], r[sel], c)
        bits = fq.bits_from_hist(hist) * (Wsub.shape[0] / len(sel))
        bits += 16 * Wsub.shape[0] + (16 * Wsub.shape[1] if c is not None else 0) + 8 * 656
        return bits, Wsub.numel()
    if H is not None:
        if cols is not None: H = H[cols][:, cols]
        Q, hist, _ = fq.quantize_ldlq(Wsub, H, r, c, damp=spec.get("damp", 0.01))
    else:
        Q, hist = fq.quantize_rtn(Wsub, r, c)
    bits = fq.matrix_bits(hist, Wsub.shape[0], Wsub.shape[1], has_col_scale=c is not None)
    if write:
        Qb = Q.to(W.dtype)
        if rows is None and cols is None:
            W.copy_(Qb)
        elif rows is not None and cols is None:
            W[rows] = Qb
        elif cols is not None and rows is None:
            W[:, cols] = Qb
        else:
            raise NotImplementedError
    return bits, Wsub.numel()


def total_bits(mats, spec, mult, st, sample=16, fixed=None):
    tot = 0.0; npar = 0
    for m in mats:
        if fixed is not None and m["name"] in fixed:
            b, n = fixed[m["name"]]
        else:
            b, n = quantize_matrix(m, spec, mult, st, sample=sample, write=False)
        tot += b; npar += n
    return tot, npar


def calibrate_mult(mats, spec, target_bits, st, lo=0.2, hi=3.0, iters=14, sample=16, fixed=None):
    """Bisection (log domain) of the global multiplier so that estimated total bits == target."""
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        b, _ = total_bits(mats, spec, mid, st, sample=sample, fixed=fixed)
        if b > target_bits: lo = mid
        else: hi = mid
    return math.sqrt(lo * hi)


def quantize_all(mats, spec, mult, st, verbose=False):
    tot = 0.0; npar = 0; bykind = {}
    for m in mats:
        b, n = quantize_matrix(m, spec, mult, st)
        tot += b; npar += n
        k = bykind.setdefault(m["kind"], [0.0, 0]); k[0] += b; k[1] += n
    return dict(bits=tot, params=npar, bpw=tot / npar, bykind={k: v[0] / v[1] for k, v in bykind.items()})
