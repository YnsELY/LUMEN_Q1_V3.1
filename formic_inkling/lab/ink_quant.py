"""FORMIC-Ink — passe E8-LDLQ (recette F1 transposée à Inkling 2 Mini). Dérivé de formic_arch/lab/seq.py (ldlq_pass),
formic_arch/lab/harness.py (quantize_matrix/_group) et formic_v7/lab/v7.py (export, load_package).

Différences avec F1 imposées par l'architecture (mandat § 4) :
  * Gram par expert (non pondéré, L0), compté en tokens × applications (×3 dans le cœur) ; seuil de couverture
    n_min = 4·2304 (w13) / 4·1152 (w2) : sous le seuil -> E8 au plus proche, même multiplicateur.
  * Capture par crochets sur le VRAI graphe Transformers (forward complet de l'élève courant, arrêt anticipé dès la dernière
    application de la dernière couche du groupe) ; Gram d'une matrice du cœur = somme des 3 passes.
  * Ordre : phase A (entrées de l'attention, wo, w13 de toutes les couches du groupe), phase B (w2 du groupe, Gram
    recalculé après quantification de l'attention et des w13 : prend en compte les softcaps du MLP).
Écart budgétaire déclaré (non imposé par l'architecture) : capture PAR GROUPES de couches au lieu de couche par couche
(le forward mesuré, ~2 200 tokens/s, rend 30 à 48 forwards de la calibration hors budget).
Non quantifiés (copie bit à bit dans le type du checkpoint, comptés) : routeurs, contrôleurs de boucle, requêtes inter-passes
et attn_res, normes, puits, rel_logits_proj, gate_attn (o_proj.gate_proj, 7,08 M < 10 M), sconv, clips.
"""
import os, sys, json, time, math, glob, shutil, subprocess
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formic_q as fq

S = globals().get("S") or {}          # persistant à travers importlib.reload (copie du professeur)
CODEC = "/workspace/LUMEN_Q1_V3.1/formic_inkling/codec"
PKG_ROOT = "/workspace/work_ink/packages"
GROUPS = [list(range(0, 6)), list(range(6, 10)), list(range(10, 14)), list(range(14, 18)), list(range(18, 24))]
ATT_IN = ("self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.relative_position_bias.linear")
ATT_OUT = ("self_attn.o_proj.linear",)
NE = 128


class _Stop(Exception):
    pass


def init(model):
    S["model"] = model; S["lm"] = model.model.language_model
    if "orig" not in S:                       # copie CPU des poids du professeur (restauration)
        S["orig"] = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if not n.startswith("model.vision_tower")}
    return S


@torch.no_grad()
def restore():
    for n, p in S["model"].named_parameters():
        if n in S["orig"]: p.copy_(S["orig"][n].to(p.device))


def mod(path):
    m = S["lm"]
    for a in path.split("."): m = getattr(m, a)
    return m


def napps(i): return 3 if 6 <= i < 18 else 1


# ------------------------------------------------------------------------------------------------ capture
class Capture:
    """Crochets de capture des Gram (FP32, GPU) pour un ensemble de couches.
    phase 'A' : entrées des linéaires d'attention + entrées des experts (w13) ; phase 'B' : entrées de w2 (calculées avec
    les w13 courants, déjà quantifiés), plus 'final' : états finaux normalisés (unembed)."""
    def __init__(self, layers, phase, final=False, stop=None):
        self.G = {}; self.N = {}; self.h = []; self.calls = {}
        lm = S["lm"]; self.stop = stop
        for i in layers:
            L = lm.layers[i]
            if phase == "A":
                for k in ATT_IN + ATT_OUT:
                    self.h.append(mod(f"layers.{i}.{k}").register_forward_pre_hook(self._lin(f"layers.{i}.{k}")))
                self.G[f"{i}.w13"] = torch.zeros(NE, 2304, 2304, device="cuda"); self.N[f"{i}.w13"] = torch.zeros(NE, dtype=torch.long, device="cuda")
            else:
                self.G[f"{i}.w2"] = torch.zeros(NE, 1152, 1152, device="cuda"); self.N[f"{i}.w2"] = torch.zeros(NE, dtype=torch.long, device="cuda")
            self.h.append(L.mlp.experts.register_forward_pre_hook(self._exp(i, phase)))
        if stop is not None:
            self.h.append(lm.layers[stop].register_forward_hook(self._stop(stop)))
        if final:
            self.G["final"] = torch.zeros(2304, 2304, device="cuda"); self.N["final"] = 0

    def _lin(self, key):
        def hook(m, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            if key not in self.G: self.G[key] = torch.zeros(x.shape[1], x.shape[1], device="cuda"); self.N[key] = 0
            self.G[key].addmm_(x.T, x); self.N[key] += x.shape[0]
        return hook

    def _exp(self, i, phase):
        def hook(m, args, kwargs=None):
            x, idx = args[0], args[1]
            T, k = idx.shape
            flat = idx.reshape(-1); tid = torch.arange(T, device=x.device).repeat_interleave(k)
            keep = flat < NE; flat = flat[keep]; tid = tid[keep]
            order = torch.argsort(flat, stable=True); flat = flat[order]; tid = tid[order]
            cnt = torch.bincount(flat, minlength=NE); cl = cnt.tolist(); s = 0
            key = f"{i}.w13" if phase == "A" else f"{i}.w2"
            G = self.G[key]; self.N[key] += cnt
            for e in range(NE):
                n = cl[e]
                if n == 0: continue
                xe = x[tid[s:s + n]]; s += n
                if phase == "A":
                    xf = xe.float(); G[e].addmm_(xf.T, xf)
                else:
                    gu = F.linear(xe, m.gate_up_proj[e]); g, u = gu.chunk(2, dim=-1)
                    hh = m.act_fn(g, u).float(); G[e].addmm_(hh.T, hh)
        return hook

    def _stop(self, i):
        need = napps(i)
        def hook(m, args, out):
            self.calls[i] = self.calls.get(i, 0) + 1
            if self.calls[i] >= need: raise _Stop()
        return hook

    def remove(self):
        for h in self.h: h.remove()


@torch.no_grad()
def run_cal(cap, X, bs=8):
    lm = S["lm"]; t0 = time.time()
    for b in range(0, X.shape[0], bs):
        cap.calls = {}
        try:
            out = lm(input_ids=X[b:b + bs].cuda().long())
            if "final" in cap.G:                                    # Gram des états finaux normalisés (entrée de la tête)
                x = out.last_hidden_state.reshape(-1, out.last_hidden_state.shape[-1]).float()
                cap.G["final"].addmm_(x.T, x); cap.N["final"] += x.shape[0]
        except _Stop:
            pass
    torch.cuda.synchronize()
    return time.time() - t0


# ------------------------------------------------------------------------------------------------ E8 batché
def ldl_prepare_batched(H, damp=0.01):
    """H [E,n,n] float64 -> U [E,n,n] fp32, inv/metric [E,n/8,8,8] (même algèbre que fq.ldl_prepare, par expert)."""
    E, n, _ = H.shape
    d = torch.diagonal(H, dim1=1, dim2=2)
    H = H + torch.diag_embed(damp * d.mean(1, keepdim=True).expand(-1, n))
    C = torch.linalg.cholesky(H.flip(1, 2)); del H
    sd = torch.diagonal(C, dim1=1, dim2=2).clone()
    C = C / sd[:, None, :]
    U64 = C.flip(1, 2); del C
    d64 = (sd * sd).flip(1)
    nb = n // 8
    blocks = U64.view(E, nb, 8, nb, 8)[:, torch.arange(nb), :, torch.arange(nb), :]          # [nb,E,8,8]
    blocks = blocks.permute(1, 0, 2, 3).contiguous()
    inv = torch.linalg.inv(blocks).float()
    dd = d64.view(E, nb, 8)
    metric = ((blocks * dd[:, :, None, :]) @ blocks.transpose(-1, -2)).float()
    return U64.float(), inv, metric


@torch.no_grad()
def ldlq_rows_batched(Wp, U, INV, MET, superblock=256):
    """Wp [E,m,n] (coordonnées normalisées W/r). Même boucle que fq.ldlq_rows, vectorisée sur les experts."""
    E_, m, n = Wp.shape
    Err = torch.zeros_like(Wp); Qp = torch.empty_like(Wp)
    ar = torch.arange(E_ * m, device=Wp.device)
    for j0 in range(0, n, superblock):
        j1 = min(n, j0 + superblock)
        Fprev = torch.bmm(Err[:, :, :j0], U[:, :j0, j0:j1]) if j0 else None
        for j in range(j0, j1, 8):
            b = j // 8
            center = Wp[:, :, j:j + 8]
            raw = None
            if Fprev is not None: raw = Fprev[:, :, j - j0:j - j0 + 8]
            if j > j0:
                t = torch.bmm(Err[:, :, j0:j], U[:, j0:j, j:j + 8])
                raw = t if raw is None else raw + t
            if raw is not None: center = center + torch.bmm(raw, INV[:, b])
            c2 = center.reshape(-1, 8)
            cand = fq.e8_candidates(c2)                                          # [E*m,7,8]
            err = c2[:, None, :] - 0.5 * cand
            Mb = MET[:, b].repeat_interleave(m, 0)                               # [E*m,8,8]
            val = torch.einsum('xki,xij,xkj->xk', err, Mb, err)
            u = cand[ar, val.argmin(1)].view(E_, m, 8)
            q = 0.5 * u
            Qp[:, :, j:j + 8] = q
            Err[:, :, j:j + 8] = Wp[:, :, j:j + 8] - q
    del Err
    return Qp


# ------------------------------------------------------------------------------------------------ export
def export_begin(vid, workers=48):
    ind = f"/workspace/work_ink/export_in/{vid}"; pkg = f"{PKG_ROOT}/{vid}"
    os.makedirs(ind, exist_ok=True); os.makedirs(pkg, exist_ok=True)
    if os.path.exists(ind + "/_END"): os.remove(ind + "/_END")
    for f in os.listdir(ind):
        if ".claimed." in f: os.rename(os.path.join(ind, f), os.path.join(ind, f.split(".claimed.")[0]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen([sys.executable, "-I", "-B", CODEC + "/package.py", "worker", ind, pkg], env=env,
                              stdout=open(f"{ind}/_worker{i}.log", "a"), stderr=subprocess.STDOUT) for i in range(workers)]
    S["export"] = dict(ind=ind, pkg=pkg, procs=procs, vid=vid)
    return pkg


def emit(name, U, r):
    """U : codes doublés (float, GPU) [m,n] ; r : pas FP16 [m]."""
    ex = S.get("export")
    if ex is None: return
    a = float(U.abs().max()); dt = np.int8 if a <= 127 else np.int16
    tmp = os.path.join(ex["ind"], name + ".tmpnpz")
    with open(tmp, "wb") as f:
        np.savez(f, U=U.cpu().numpy().astype(dt), steps=r.to(torch.float16).cpu().numpy(), name=np.array(name))
    os.replace(tmp, os.path.join(ex["ind"], name + ".npz"))


def export_wait():
    ex = S["export"]
    while any(f.endswith(".npz") or ".claimed." in f for f in os.listdir(ex["ind"])): time.sleep(0.5)


def export_end():
    ex = S.pop("export"); open(ex["ind"] + "/_END", "w").write("end")
    for p in ex["procs"]: p.wait()
    left = [f for f in os.listdir(ex["ind"]) if f.endswith(".npz") or ".claimed." in f]
    if left: raise RuntimeError("export incomplet : %s" % left[:5])
    return ex["pkg"]


# ------------------------------------------------------------------------------------------------ quantification
def acc(name, kind, U, m, n, A):
    h = fq.shell_hist(U.view(m, n // 8, 8))
    b = fq.matrix_bits(h, m, n)
    A["bits"] += b; A["params"] += m * n
    k = A["bykind"].setdefault(kind, [0.0, 0]); k[0] += b; k[1] += m * n
    return b


@torch.no_grad()
def q_dense(name, kind, W, H, mult, A, rtn=False):
    """W : paramètre (modifié en place). LDLQ sur H (FP32 [n,n]) ou E8 au plus proche."""
    Wf = W.float(); m, n = Wf.shape
    r = fq.row_steps(Wf, mult)
    if rtn or H is None:
        Q, _ = fq.quantize_rtn(Wf, r)
    else:
        Q, _, _ = fq.quantize_ldlq(Wf, H.double(), r, damp=0.01)
    U = torch.round(2.0 * Q / r[:, None])
    acc(name, kind, U, m, n, A); emit(name, U, r)
    W.copy_(Q.to(W.dtype))
    return U


@torch.no_grad()
def q_experts(i, which, G, N, mult, A, nmin, chunk=16, rtn_all=False):
    """Quantifie les 128 experts (w13 : gate_up_proj [E,2304,2304] ; w2 : down_proj [E,2304,1152]) d'une couche."""
    ex = S["lm"].layers[i].mlp.experts
    P = ex.gate_up_proj if which == "w13" else ex.down_proj
    E_, m, n = P.shape
    under = [e for e in range(E_) if rtn_all or int(N[e]) < nmin]
    good = [e for e in range(E_) if e not in under]
    R = torch.empty(E_, m, device="cuda")
    for e in range(E_): R[e] = fq.row_steps(P[e].float(), mult)
    for e in under:
        Q, _ = fq.quantize_rtn(P[e].float(), R[e]); U = torch.round(2.0 * Q / R[e][:, None])
        acc(f"llm.layers.{i}.mlp.experts.{which}_weight.{e}", which, U, m, n, A); emit(f"llm.layers.{i}.mlp.experts.{which}_weight.{e}", U, R[e])
        P[e].copy_(Q.to(P.dtype))
    for c in range(0, len(good), chunk):
        es = good[c:c + chunk]
        Hc = G[es].double() / N[es].double()[:, None, None]
        U_, INV, MET = ldl_prepare_batched(Hc); del Hc
        Wp = P[es].float() / R[es][:, :, None]
        Qp = ldlq_rows_batched(Wp, U_, INV, MET); del Wp, U_, INV, MET
        for k, e in enumerate(es):
            U = 2.0 * Qp[k]
            name = f"llm.layers.{i}.mlp.experts.{which}_weight.{e}"
            acc(name, which, U, m, n, A); emit(name, U, R[e])
            P[e].copy_((Qp[k] * R[e][:, None]).to(P.dtype))
        del Qp
    return under


def matrix_list():
    """Toutes les matrices E8 du chemin texte : (nom paquet, chemin module/param, genre)."""
    out = [("llm.embed", "embed_tokens.weight", "embed")]
    for i in range(24):
        for k in ATT_IN + ATT_OUT: out.append((f"llm.layers.{i}.{k}", f"layers.{i}.{k}.weight", "attn"))
        for e in range(NE):
            out.append((f"llm.layers.{i}.mlp.experts.w13_weight.{e}", (i, "w13", e), "w13"))
            out.append((f"llm.layers.{i}.mlp.experts.w2_weight.{e}", (i, "w2", e), "w2"))
    out.append(("llm.unembed", "lm_head", "unembed"))
    return out


def weight_of(path):
    if path == "lm_head": return S["model"].lm_head.weight
    if isinstance(path, tuple):
        i, w, e = path; ex = S["lm"].layers[i].mlp.experts
        return (ex.gate_up_proj if w == "w13" else ex.down_proj)[e]
    m = S["lm"]
    for a in path.split("."): m = getattr(m, a)
    return m


@torch.no_grad()
def est_bits_rtn(mult, sample=16, kinds=None, mults=None):
    """Bits idéaux E8 au plus proche (échantillon de lignes), par genre."""
    out = {}
    for name, path, kind in matrix_list():
        if kinds and kind not in kinds: continue
        W = weight_of(path).float(); m, n = W.shape
        mu = (mults or {}).get(kind, mult)
        sel = torch.arange(0, m, sample, device=W.device)
        r = fq.row_steps(W[sel], mu); _, h = fq.quantize_rtn(W[sel], r)
        b = fq.bits_from_hist(h) * (m / len(sel)) + 16 * m + 8 * 656
        o = out.setdefault(kind, [0.0, 0]); o[0] += b; o[1] += m * n
    return out


def small_tensors():
    """Paramètres du chemin texte NON quantifiés : (nom, tenseur, dtype de stockage 'bf16'|'f32')."""
    qids = set()
    for name, path, kind in matrix_list():
        if isinstance(path, tuple):
            ex = S["lm"].layers[path[0]].mlp.experts; qids.add(id(ex.gate_up_proj)); qids.add(id(ex.down_proj))
        else: qids.add(id(weight_of(path)))
    out = []
    for n, p in S["model"].named_parameters():
        if n.startswith("model.vision_tower") or id(p) in qids: continue
        o = S["orig"][n] if n in S["orig"] else p.detach().cpu()
        dt = "bf16" if (o.dtype == torch.bfloat16 or torch.equal(o.float().to(torch.bfloat16).float(), o.float())) else "f32"
        out.append((n, o, dt))
    for n, b in S["model"].named_buffers():
        if n.startswith("model.vision_tower") or "e_score_correction_bias" not in n: continue
        out.append((n, b.detach().cpu(), "f32"))
    return out


def small_bytes():
    return sum(t.numel() * (2 if dt == "bf16" else 4) + 64 for _, t, dt in small_tensors())


# ================================================================================================ passe complète
BUDGET_TEXT = 8474257664          # octets des tenseurs texte de UD-IQ2_XXS (mesuré sur fichier, gguf_hdr)
NMIN = {"w13": 4 * 2304, "w2": 4 * 1152}


def zone(i): return "core" if 6 <= i < 18 else "edge"


def mult_for(kind, i, spec):
    m = spec["mult"]
    if spec.get("delta") and kind in ("w13", "w2", "attn") and i is not None:
        m = spec["mult_core"] if zone(i) == "core" else spec["mult_edge"]
    if spec.get("fine") and kind in ("attn", "unembed"):
        m = spec["mult_fine"]
    return m


@torch.no_grad()
def pilot(X, n_docs=128):
    """Gram du professeur (prélude, couches 0 et 5) -> rapport bits LDLQ / bits au plus proche, par genre, au multiplicateur 0,6."""
    mu = 0.6; out = {}
    cap = Capture([0, 5], "A", stop=5); run_cal(cap, X[:n_docs]); cap.remove()
    capB = Capture([0, 5], "B", stop=5); run_cal(capB, X[:n_docs]); capB.remove()
    for kind, items in (("attn", [(f"layers.{i}.{k}", cap.G[f"layers.{i}.{k}"] / cap.N[f"layers.{i}.{k}"]) for i in (0, 5) for k in ATT_IN + ATT_OUT]),):
        lr = rr = 0.0
        for path, H in items:
            W = mod(path).weight.float(); r = fq.row_steps(W, mu)
            _, hl, _ = fq.quantize_ldlq(W, H.double(), r); _, hr = fq.quantize_rtn(W, r)
            lr += fq.bits_from_hist(hl); rr += fq.bits_from_hist(hr)
        out[kind] = lr / rr
    for which, CP in (("w13", cap), ("w2", capB)):
        lr = rr = 0.0
        for i in (0, 5):
            G = CP.G[f"{i}.{which}"]; N = CP.N[f"{i}.{which}"]
            ex = S["lm"].layers[i].mlp.experts; P = ex.gate_up_proj if which == "w13" else ex.down_proj
            es = torch.argsort(N, descending=True)[:16].tolist()
            R = torch.stack([fq.row_steps(P[e].float(), mu) for e in es])
            U_, INV, MET = ldl_prepare_batched(G[es].double() / N[es].double()[:, None, None])
            Qp = ldlq_rows_batched(P[es].float() / R[:, :, None], U_, INV, MET)
            for k, e in enumerate(es):
                lr += fq.bits_from_hist(fq.shell_hist((2 * Qp[k]).view(P.shape[1], -1, 8)))
                _, hr = fq.quantize_rtn(P[e].float(), R[k]); rr += fq.bits_from_hist(hr)
        out[which] = lr / rr
    out["unembed"] = out["attn"]; out["embed"] = 1.0
    del cap, capB; torch.cuda.empty_cache()
    return out


@torch.no_grad()
def calibrate(ratio, target_bits, spec, iters=16):
    """Multiplicateur global (ou couple cœur/bords pour V1, ou fin pour V4) tel que bits estimés = cible."""
    def tot(mult):
        sp = dict(spec, mult=mult)
        if spec.get("delta"):
            sp["mult_core"] = mult * 2 ** (-spec["delta"]); sp["mult_edge"] = mult * 2 ** (spec["delta_edge"])
        if spec.get("fine"): sp["mult_fine"] = mult * 2 ** (-spec["fine"])
        b = 0.0
        for name, path, kind in matrix_list():
            i = path[0] if isinstance(path, tuple) else (int(path.split(".")[1]) if path.startswith("layers.") else None)
            W = weight_of(path).float(); m, n = W.shape
            sel = torch.arange(0, m, 16, device=W.device)
            r = fq.row_steps(W[sel], mult_for(kind, i, sp)); _, h = fq.quantize_rtn(W[sel], r)
            b += ratio[kind] * fq.bits_from_hist(h) * (m / len(sel)) + 16 * m + 8 * 656
        return b, sp
    lo, hi = 0.2, 3.0
    for _ in range(iters):
        mid = math.sqrt(lo * hi); b, _ = tot(mid)
        if b > target_bits: lo = mid
        else: hi = mid
    b, sp = tot(math.sqrt(lo * hi))
    return sp, b


def write_small(pkg):
    sys.path.insert(0, CODEC); import package
    nb = 0; n = 0
    for name, t, dt in small_tensors():
        if dt == "bf16": a = t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        else: a = t.float().numpy()
        nb += package.write_raw(pkg, f"raw.{dt}.{name}", a); n += t.numel()
    return nb, n


def pkg_bytes(pkg, prefix=None):
    sys.path.insert(0, CODEC); import package
    man = package.load_manifest(pkg)
    return sum(v["bytes"] for k, v in man.items() if prefix is None or k.startswith(prefix))


@torch.no_grad()
def run_pass(vid, spec, X, bs=4, log=print):
    """Passe complète : embed (E8 au plus proche), groupes (phase A / B), unembed (LDLQ sur Gram final, multiplicateur ajusté
    pour égaler le budget en octets réels). Les Gram sont normalisés par le nombre de lignes (comme F1)."""
    t0 = time.time(); A = dict(bits=0.0, params=0, bykind={}); cov = {}; times = {}
    pkg = export_begin(vid)
    sb, sn = write_small(pkg); log(f"[{vid}] petits tenseurs : {sn} poids, {sb} octets")
    q_dense("llm.embed", "embed", S["lm"].embed_tokens.weight, None, mult_for("embed", None, spec), A, rtn=True)
    for g in GROUPS:
        tg = time.time()
        cap = Capture(g, "A", stop=g[-1]); ta = run_cal(cap, X, bs); cap.remove()
        for i in g:
            for k in ATT_IN + ATT_OUT:
                key = f"layers.{i}.{k}"
                q_dense(f"llm.{key}", "attn", mod(key).weight, cap.G[key] / cap.N[key], mult_for("attn", i, spec), A)
            G = cap.G.pop(f"{i}.w13"); N = cap.N[f"{i}.w13"]
            if spec.get("weighted"): pass
            und = q_experts(i, "w13", G, N, mult_for("w13", i, spec), A, NMIN["w13"]); del G
            cov[f"{i}.w13"] = dict(counts=N.tolist(), under=und)
        del cap; torch.cuda.empty_cache()
        capB = Capture(g, "B", stop=g[-1]); tb = run_cal(capB, X, bs); capB.remove()
        for i in g:
            G = capB.G.pop(f"{i}.w2"); N = capB.N[f"{i}.w2"]
            und = q_experts(i, "w2", G, N, mult_for("w2", i, spec), A, NMIN["w2"]); del G
            cov[f"{i}.w2"] = dict(counts=N.tolist(), under=und)
        del capB; torch.cuda.empty_cache()
        times[str(g)] = dict(captureA=ta, captureB=tb, total=time.time() - tg)
        log(f"[{vid}] groupe {g} : capture A {ta:.0f}s B {tb:.0f}s total {time.time() - tg:.0f}s ; bpw idéal jusqu'ici {A['bits'] / A['params']:.4f} ; "
            f"pic {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB ; sous le seuil : " + str({k: len(v['under']) for k, v in cov.items() if int(k.split('.')[0]) in g}))
        json.dump(dict(done=[str(x) for x in GROUPS[:GROUPS.index(g) + 1]], A=A, cov=cov, times=times), open(f"{PKG_ROOT}/{vid}.state.json", "w"))
    # unembed : Gram des états finaux normalisés de l'élève complet
    tf = time.time()
    capF = Capture([], "A", final=True); run_cal(capF, X, bs); capF.remove()
    Hf = capF.G["final"] / capF.N["final"]; del capF
    export_wait()
    used = pkg_bytes(pkg)
    rem = BUDGET_TEXT - used
    W = S["model"].lm_head.weight; m, n = W.shape
    lo, hi = 0.05, 5.0; Wf = W.float()
    for _ in range(18):            # ajustement : octets idéaux LDLQ de la tête = reste du budget (le codec réel est ~idéal)
        mid = math.sqrt(lo * hi); r = fq.row_steps(Wf, mid)
        sel = torch.arange(0, m, 8, device=W.device)
        _, h = fq.quantize_rtn(Wf[sel], r[sel])
        b = spec["_ratio"]["unembed"] * fq.bits_from_hist(h) * 8 + 16 * m + 8 * 656
        if b / 8 > rem: lo = mid
        else: hi = mid
    mu_u = math.sqrt(lo * hi)
    q_dense("llm.unembed", "unembed", W, Hf, mu_u, A)
    times["final+unembed"] = time.time() - tf
    export_end()
    tot = pkg_bytes(pkg)
    info = dict(vid=vid, spec={k: v for k, v in spec.items()}, mult_unembed=mu_u, ideal_bpw=A["bits"] / A["params"],
                ideal_bykind={k: v[0] / v[1] for k, v in A["bykind"].items()}, params_e8=A["params"], small_bytes=sb, small_params=sn,
                text_bytes_real=tot, budget=BUDGET_TEXT, rel_dev=tot / BUDGET_TEXT - 1, seconds=time.time() - t0, times=times,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                under_threshold={k: v["under"] for k, v in cov.items() if v["under"]})
    json.dump(dict(info=info, coverage=cov), open(f"{PKG_ROOT}/{vid}.pass.json", "w"))
    log(f"[{vid}] passe terminée en {info['seconds']:.0f}s ; octets texte réels {tot} (budget {BUDGET_TEXT}, écart {100 * info['rel_dev']:+.3f} %) ; bpw idéal {info['ideal_bpw']:.4f}")
    return info, cov
