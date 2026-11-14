# Structural diagnostics on the BF16 weights (no model modification).
import json, math, time
H.restore()
lm = H.S["lm"]; out = {}
E = lm.embed_tokens.weight; Wh = model.lm_head.weight
def cgram(A, B=None, scale=None, chunk=16384):
    """sum over row chunks of A_c^T B_c in float64 (A, B [V, n])."""
    out = 0
    for i in range(0, A.shape[0], chunk):
        a = A[i:i+chunk].double(); b = a if B is None else B[i:i+chunk].double()
        if scale is not None: a = a * scale; b = a
        out = out + a.T @ b
    return out
# (1) is lm_head a linear function of embed_tokens?  min_M ||Wh - E M||
EtE = cgram(E); EtW = cgram(E, Wh)
M = torch.linalg.solve(EtE + 1e-8*EtE.diagonal().mean()*torch.eye(5120, device=E.device, dtype=torch.float64), EtW)
ss_res = 0.0; ss_tot = 0.0; cs = 0.0
for i in range(0, E.shape[0], 16384):
    e = E[i:i+16384].double(); w = Wh[i:i+16384].double()
    ss_res += float((w - e @ M).pow(2).sum()); ss_tot += float(w.pow(2).sum())
    cs += float(torch.nn.functional.cosine_similarity(e, w, dim=1).sum())
out["head_from_embed_R2"] = 1 - ss_res/ss_tot; out["cos_rowwise_mean"] = cs/E.shape[0]
print("lm_head ~ embed*M  R2=%.4f ; mean row cosine(embed, head)=%.4f" % (out["head_from_embed_R2"], out["cos_rowwise_mean"]))
del M, EtE, EtW, e, w
# (2) spectra: energy captured by top-r singular directions (plain and activation-weighted)
def spec_energy(W, X=None):
    if W.shape[0] > 100000:
        G = cgram(W, scale=None if X is None else X.double().sqrt().cuda()[None, :])
    else:
        W = W.double()
        if X is not None: W = W * X.double().sqrt().cuda()[None, :]
        G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)
    cs = ev.cumsum(0)/ev.sum(); n = len(ev)
    p = ev/ev.sum(); erank = float(torch.exp(-(p[p>0]*p[p>0].log()).sum()))
    return {"n": n, "erank": erank, **{"top%d%%"%int(100*f): float(cs[int(n*f)-1]) for f in (0.125, 0.25, 0.5)}}
st = H.S["st"]
sp = {"embed": spec_energy(E), "lm_head": spec_energy(Wh), "lm_head_actw": spec_energy(Wh, st["X"]["lm_head"])}
for i in (1, 15, 31, 32, 47, 62):
    L = lm.layers[i]
    for k in ("gate_proj", "up_proj", "down_proj"):
        nm = f"layers.{i}.mlp.{k}"
        sp[nm] = spec_energy(getattr(L.mlp, k).weight); sp[nm+"_actw"] = spec_energy(getattr(L.mlp, k).weight, st["X"][nm])
    if hasattr(L, "linear_attn"):
        for k in ("in_proj_qkv", "in_proj_z", "out_proj"):
            nm = f"layers.{i}.linear_attn.{k}"; sp[nm] = spec_energy(getattr(L.linear_attn, k).weight); sp[nm+"_actw"] = spec_energy(getattr(L.linear_attn, k).weight, st["X"][nm])
    else:
        for k in ("q_proj", "o_proj"):
            nm = f"layers.{i}.self_attn.{k}"; sp[nm] = spec_energy(getattr(L.self_attn, k).weight); sp[nm+"_actw"] = spec_energy(getattr(L.self_attn, k).weight, st["X"][nm])
for k, v in sp.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
out["spectra"] = sp
# (3) weight statistics relevant to an E8 + per-row-step codec: kurtosis, row-RMS spread, massive input dims
ws = {}
for m in H.qmats():
    if m["layer"] not in (-1, 1, 31, 32, 62, 64): continue
    W = m["mod"].weight.float(); rms = W.pow(2).mean(1).sqrt()
    z = W / rms.clamp_min(1e-12)[:, None]
    d = dict(kurt_rownorm=float(z.pow(4).mean()), rowrms_cv=float(rms.std()/rms.mean()),
             colrms_cv=float(W.pow(2).mean(0).sqrt().std()/W.pow(2).mean(0).sqrt().mean()))
    if m["name"] in st["X"]:
        X = st["X"][m["name"]]; d["X_max_over_median"] = float(X.max()/X.median()); d["X_top8_share"] = float(X.topk(8).values.sum()/X.sum())
        G = st["G"][m["name"]]; d["G_p99_over_p01"] = float(torch.quantile(G[:100000], .99)/torch.quantile(G[:100000], .01).clamp_min(1e-30))
    ws[m["name"]] = d; print(m["name"], {a: round(b, 3) for a, b in d.items()})
out["weight_stats"] = ws
# (4) SwiGLU channel saliency concentration (Taylor): share of saliency in the weakest channels
ts = {}
for i in range(64):
    s = st["TS"][f"layers.{i}.mlp.down_proj"].double().sort().values; c = s.cumsum(0)/s.sum()
    ts[i] = [float(c[int(len(s)*f)-1]) for f in (0.125, 0.25, 0.5)]
print("Taylor saliency share of weakest 12.5/25/50% channels, layers 0,8,..:", {i: [round(a, 4) for a in ts[i]] for i in range(0, 64, 8)})
out["taylor_share_weakest"] = ts
json.dump(out, open(H.OUT+"/diag_structure.json", "w"))

for k in ("E", "Wh", "W", "z", "rms", "sp", "ws"):
    if k in globals(): del globals()[k]
import gc; gc.collect(); torch.cuda.empty_cache(); print("alloc GiB", torch.cuda.memory_allocated()/2**30)
