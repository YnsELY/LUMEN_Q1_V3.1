"""Layer-by-layer (sequential) machinery on real contextual activations of the calibration split:
Gram capture at the true input of each linear, structural FFN-width pruning with least-squares
compensation, and a full sequential E8-LDLQ pass (GPTQ-style symmetric objective on student inputs).
"""
import time, math, torch
import formic_q as fq
import harness as H


class _Stop(Exception):
    pass


class Seq:
    def __init__(self, n_docs=144, T=1024, bs=8):
        S = H.S; lm = S["lm"]; self.lm = lm; self.bs = bs
        ids = S["data"]["cal"][:n_docs, :T]
        self.n = ids.shape[0]
        assert self.n % bs == 0
        # capture the exact kwargs the text model passes to each layer type
        kw = {}
        def grab(kind):
            def h(mod, args, kwargs):
                kw[kind] = dict(kwargs); return None
            return h
        i_gdn = next(i for i, L in enumerate(lm.layers) if hasattr(L, "linear_attn"))
        i_att = next(i for i, L in enumerate(lm.layers) if hasattr(L, "self_attn"))
        h1 = lm.layers[i_gdn].register_forward_pre_hook(grab("gdn"), with_kwargs=True)
        h2 = lm.layers[i_att].register_forward_pre_hook(grab("attn"), with_kwargs=True)
        with torch.no_grad():
            lm(input_ids=ids[:bs].cuda())
        h1.remove(); h2.remove()
        self.kw = kw
        with torch.no_grad():
            self.h = torch.cat([lm.embed_tokens(ids[i:i + bs].cuda()) for i in range(0, self.n, bs)])
        self.ntok = self.n * ids.shape[1]

    def _kw(self, i):
        return self.kw["gdn" if hasattr(self.lm.layers[i], "linear_attn") else "attn"]

    @torch.no_grad()
    def gram(self, i, module, also=None):
        """Gram (sum x x^T / ntok, fp64) of the input of `module` when layer i runs on current hidden."""
        L = self.lm.layers[i]; acc = {}
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            g = x.T @ x
            acc["g"] = g if "g" not in acc else acc["g"].add_(g)
            raise _Stop()
        hk = module.register_forward_pre_hook(hook)
        try:
            for b in range(0, self.n, self.bs):
                try:
                    L(self.h[b:b + self.bs], **self._kw(i))
                except _Stop:
                    pass
        finally:
            hk.remove()
        return (acc["g"].double() / self.ntok)

    @torch.no_grad()
    def advance(self, i):
        L = self.lm.layers[i]
        for b in range(0, self.n, self.bs):
            self.h[b:b + self.bs] = L(self.h[b:b + self.bs], **self._kw(i))

    @torch.no_grad()
    def final_gram(self):
        acc = None
        for b in range(0, self.n, self.bs):
            x = self.lm.norm(self.h[b:b + self.bs]).reshape(-1, self.h.shape[-1]).float()
            g = x.T @ x
            acc = g if acc is None else acc.add_(g)
        return acc.double() / self.ntok


# ----------------------------------------------------------------------------- structural width pruning
@torch.no_grad()
def prune_width(frac, st, score="taylor", ridge=1e-4, compensate=True, layers=None, log=True):
    """A1: remove a fraction of SwiGLU channels in every layer (kept count multiple of 8),
    ranked by Taylor saliency sum_t (a_t * dL/da_t)^2 measured on the BF16 model, then refit the kept
    columns of down_proj by ridge least squares on the layer's real inputs (sequential, pruned prefix):
        min_B || B a_K - W a ||^2  ->  B = W A[:,K] (A[K,K] + ridge*mean(diag) I)^-1
    Modifies the model in place (zeros = removed structure). Returns dict(keep={layer: idx}, rel_err=...)."""
    lm = H.S["lm"]; seq = Seq(); keep = {}; rel = {}
    nI = lm.layers[0].mlp.down_proj.weight.shape[1]
    nk = int(round(nI * (1 - frac) / 8)) * 8
    for i, L in enumerate(lm.layers):
        t = time.time()
        if layers is None or i in layers:
            sc = st["TS"][f"layers.{i}.mlp.down_proj"].cuda() if score == "taylor" else st["X"][f"layers.{i}.mlp.down_proj"].cuda()
            K = torch.topk(sc, nk).indices.sort().values
            mask = torch.ones(nI, dtype=torch.bool, device=sc.device); mask[K] = False
            W = L.mlp.down_proj.weight
            if compensate:
                A = seq.gram(i, L.mlp.down_proj)                       # [nI,nI] fp64, full channels
                AKK = A[K][:, K].clone()
                AKK.diagonal().add_(ridge * AKK.diagonal().mean())
                rhs = W.double() @ A[:, K]                              # [out, nk]
                B = torch.linalg.solve(AKK, rhs.T).T                    # B AKK = rhs
                # relative output error of the refit vs zeroing only (calibration, in-sample)
                Wd = W.double()
                tot = ((Wd @ A) * Wd).sum()
                D = Wd.clone(); D[:, K] -= B; e_fit = ((D @ A) * D).sum() / tot
                D = torch.zeros_like(Wd); D[:, mask] = Wd[:, mask]; e_zero = ((D @ A) * D).sum() / tot
                rel[i] = (float(e_zero), float(e_fit))
                W[:, K] = B.to(W.dtype)
                del A, AKK, rhs, B, Wd, D
            W[:, mask] = 0
            L.mlp.gate_proj.weight[mask] = 0
            L.mlp.up_proj.weight[mask] = 0
            keep[i] = K.cpu()
            if log and (i % 8 == 0 or i == 63):
                print("prune L%d keep %d/%d rel_err zero=%.4f fit=%.4f (%.1fs)" % (i, nk, nI, *rel.get(i, (float('nan'),) * 2), time.time() - t), flush=True)
        seq.advance(i)
    del seq
    return dict(keep=keep, rel_err=rel, frac=frac, nk=nk)


def plan_mats(keep=None, drop=()):
    """Matrix list for a structure: FFN channel subsets (keep[layer] = kept indices) and dropped layers."""
    out = []
    for m in H.qmats():
        if m["layer"] in drop: continue
        if keep is not None and m["layer"] in keep and m["kind"] in ("gate_proj", "up_proj", "down_proj"):
            m = dict(m); K = keep[m["layer"]].cuda()
            if m["kind"] == "down_proj": m["cols"] = K
            else: m["rows"] = K
        out.append(m)
    return out


@torch.no_grad()
def drop_layers(drop):
    """A2: remove whole layers = zero their residual-writing matrices (function becomes identity)."""
    lm = H.S["lm"]
    for i in drop:
        L = lm.layers[i]
        (L.linear_attn.out_proj if hasattr(L, "linear_attn") else L.self_attn.o_proj).weight.zero_()
        L.mlp.down_proj.weight.zero_()


# ----------------------------------------------------------------------------- sequential LDLQ pass
@torch.no_grad()
def ldlq_pass(mats, spec, mult, st, drop=(), log_every=8):
    """Sequential E8-LDLQ over the whole text model. `mats` from plan_mats (may carry rows/cols subsets).
    Embedding: no-feedback E8 rounding (lookup table, no input Gram). lm_head: LDLQ on final hidden Gram."""
    lm = H.S["lm"]; t0 = time.time()
    by = {}
    for m in mats: by.setdefault(m["layer"], {})[m["kind"]] = m
    tot = 0.0; npar = 0; bykind = {}
    def acc(m, res):
        nonlocal tot, npar
        b, n = res; tot += b; npar += n
        k = bykind.setdefault(m["kind"], [0.0, 0]); k[0] += b; k[1] += n
    if -1 in by:
        acc(by[-1]["embed"], H.quantize_matrix(by[-1]["embed"], spec, mult, st))
    seq = Seq()
    for i, L in enumerate(lm.layers):
        if i in drop or i not in by:
            seq.advance(i); continue
        d = by[i]
        if hasattr(L, "linear_attn"):
            first, ins, outk = L.linear_attn.in_proj_qkv, ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"), "out_proj"
        else:
            first, ins, outk = L.self_attn.q_proj, ("q_proj", "k_proj", "v_proj"), "o_proj"
        Hm = seq.gram(i, first)
        for k in ins: acc(d[k], H.quantize_matrix(d[k], spec, mult, st, H=Hm))
        del Hm
        Hm = seq.gram(i, d[outk]["mod"]); acc(d[outk], H.quantize_matrix(d[outk], spec, mult, st, H=Hm)); del Hm
        Hm = seq.gram(i, L.mlp.gate_proj)
        for k in ("gate_proj", "up_proj"): acc(d[k], H.quantize_matrix(d[k], spec, mult, st, H=Hm))
        del Hm
        Hm = seq.gram(i, L.mlp.down_proj); acc(d["down_proj"], H.quantize_matrix(d["down_proj"], spec, mult, st, H=Hm)); del Hm
        seq.advance(i)
        if i % log_every == 0 or i == 63:
            print("  ldlq layer %d done, %.0fs elapsed, bpw so far %.4f" % (i, time.time() - t0, tot / npar), flush=True)
    if 64 in by:
        Hm = seq.final_gram(); acc(by[64]["lm_head"], H.quantize_matrix(by[64]["lm_head"], spec, mult, st, H=Hm)); del Hm
    del seq
    return dict(bits=tot, params=npar, bpw=tot / npar, bykind={k: v[0] / v[1] for k, v in bykind.items()}, seconds=time.time() - t0)
