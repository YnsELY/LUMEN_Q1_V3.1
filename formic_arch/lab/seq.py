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
        S = H.S; lm = S["lm"]; self.lm = lm
        cfg = S.get("cal_cfg")                    # V7: alternative calibration set (split name, docs, length, batch)
        if cfg and n_docs == 144:
            ids = S["data"][cfg["split"]][:cfg["n_docs"], :cfg["T"]]; bs = cfg["bs"]
        else:
            ids = S["data"]["cal"][:n_docs, :T]
        self.bs = bs
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
            lm(input_ids=ids[:bs].cuda(), use_cache=False)
        h1.remove(); h2.remove()
        # No cache object may be shared between calls: a DynamicCache captured here would carry attention
        # keys/values and GDN recurrent/conv states from one batch (and layer call) to the next.
        for k in kw:
            assert kw[k].get("past_key_values") is None, "cache leaked into captured layer kwargs"
            kw[k]["past_key_values"] = None; kw[k]["use_cache"] = False
        self.kw = kw
        with torch.no_grad():
            self.h = torch.cat([lm.embed_tokens(ids[i:i + bs].cuda()) for i in range(0, self.n, bs)])
        self.ntok = self.n * ids.shape[1]

    def _kw(self, i):
        return self.kw["gdn" if hasattr(self.lm.layers[i], "linear_attn") else "attn"]

    @torch.no_grad()
    def gram(self, i, module, tf32=True, center=False):
        """Gram (sum x x^T / ntok, fp64) of the input of `module` when layer i runs on current hidden."""
        L = self.lm.layers[i]; acc = {}
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if center:
                # centre on a provisional mean BEFORE accumulating: the covariance is then obtained without the
                # catastrophic cancellation of  E[xx^T] - mu mu^T  on the massive (near-constant) dimensions
                if "mu0" not in acc: acc["mu0"] = x.mean(0)
                x = x - acc["mu0"]
            g = x.T @ x
            acc["g"] = g if "g" not in acc else acc["g"].add_(g)
            acc["m"] = acc.get("m", 0) + x.sum(0).double()
            raise _Stop()
        hk = module.register_forward_pre_hook(hook)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)     # TF32 only where flagged (inputs are BF16 anyway)
        try:
            for b in range(0, self.n, self.bs):
                try:
                    L(self.h[b:b + self.bs], **self._kw(i))
                except _Stop:
                    pass
        finally:
            hk.remove(); torch.backends.cuda.matmul.allow_tf32 = old_tf32
        m = acc["m"] / self.ntok
        Hd = acc["g"].double() / self.ntok
        if center:
            Hd.sub_(torch.outer(m, m)); self.last_mu = (acc["mu0"].double() + m).float()
        else:
            self.last_mu = m.float()
        return Hd

    @torch.no_grad()
    def advance(self, i):
        L = self.lm.layers[i]
        for b in range(0, self.n, self.bs):
            for attempt in range(4):              # per-batch OOM retry: a batch is only written once computed
                try:
                    out = L(self.h[b:b + self.bs], **self._kw(i)); break
                except torch.OutOfMemoryError:
                    if attempt == 3: raise
                    import gc; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
            self.h[b:b + self.bs] = out; del out

    @torch.no_grad()
    def final_gram(self, center=False):
        acc = None; msum = 0; mu0 = None
        for b in range(0, self.n, self.bs):
            x = self.lm.norm(self.h[b:b + self.bs]).reshape(-1, self.h.shape[-1]).float()
            if center:
                if mu0 is None: mu0 = x.mean(0)
                x = x - mu0
            g = x.T @ x
            acc = g if acc is None else acc.add_(g)
            msum = msum + x.sum(0).double()
        m = msum / self.ntok; Hd = acc.double() / self.ntok
        if center:
            Hd.sub_(torch.outer(m, m)); self.last_mu = (mu0.double() + m).float()
        else:
            self.last_mu = m.float()
        return Hd


# ----------------------------------------------------------------------------- structural width pruning
@torch.no_grad()
def prune_width(frac, st, score="taylor", ridge=1e-4, compensate=True, layers=None, log=True):
    """A1: remove a fraction of SwiGLU channels in every layer (kept count multiple of 8),
    ranked by Taylor saliency sum_t (a_t * dL/da_t)^2 measured on the BF16 model, then refit the kept
    columns of down_proj by ridge least squares on the layer's real inputs (sequential, pruned prefix):
        min_B || B a_K - W a ||^2  ->  B = W A[:,K] (A[K,K] + ridge*mean(diag) I)^-1
    Modifies the model in place (zeros = removed structure). Returns dict(keep={layer: idx}, rel_err=...)."""
    lm = H.S["lm"]; seq = Seq(); keep = {}; rel = {}; down = {}
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
                A = _retry(lambda: seq.gram(i, L.mlp.down_proj, tf32=True), f"prune L{i} gram")   # [nI,nI] fp64
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
            keep[i] = K.cpu(); down[i] = W[:, K].clone().cpu()
            if log and (i % 8 == 0 or i == 63):
                print("prune L%d keep %d/%d rel_err zero=%.4f fit=%.4f (%.1fs)" % (i, nk, nI, *rel.get(i, (float('nan'),) * 2), time.time() - t), flush=True)
        seq.advance(i)
    del seq
    return dict(keep=keep, rel_err=rel, frac=frac, nk=nk, down=down)


@torch.no_grad()
def apply_prune(pr):
    """Re-apply a stored width-pruned structure (after H.restore()) without redoing the pass."""
    lm = H.S["lm"]
    for i, K in pr["keep"].items():
        L = lm.layers[i]; K = K.cuda(); W = L.mlp.down_proj.weight
        mask = torch.ones(W.shape[1], dtype=torch.bool, device=W.device); mask[K] = False
        W[:, K] = pr["down"][i].cuda(); W[:, mask] = 0
        L.mlp.gate_proj.weight[mask] = 0; L.mlp.up_proj.weight[mask] = 0


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
import os, gc, shutil


def _retry(fn, what=""):
    """Run fn(); on CUDA OOM free the allocator cache and retry (the step is restartable: weights are only
    written at the end of a matrix)."""
    for attempt in range(4):
        try:
            return fn()
        except torch.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
            print("  [OOM on %s, retry %d, allocated %.1f GiB]" % (what, attempt + 1, torch.cuda.memory_allocated() / 2**30), flush=True)
    return fn()


def _ckpt_save(ckpt, i, layers, by, seq, state):
    """Checkpoint: decoded weights (+ biases) of the given layers, then (atomically) hidden state + accounting."""
    os.makedirs(ckpt, exist_ok=True)
    for j in layers:
        if j not in by: continue
        blob = {m["name"]: dict(w=m["mod"].weight.detach().cpu(),
                                b=None if getattr(m["mod"], "bias", None) is None else m["mod"].bias.detach().cpu())
                for m in by[j].values()}
        torch.save(blob, f"{ckpt}/layer_{j}.pt")
    torch.save(dict(i=i, h=seq.h.cpu(), **state), f"{ckpt}/state.tmp")
    os.replace(f"{ckpt}/state.tmp", f"{ckpt}/state.pt")


def _ckpt_load(ckpt, by, seq):
    stt = torch.load(f"{ckpt}/state.pt")
    for j in [-1] + list(range(stt["i"] + 1)) + [64]:
        f = f"{ckpt}/layer_{j}.pt"
        if j not in by or not os.path.exists(f): continue
        blob = torch.load(f)
        for m in by[j].values():
            e = blob[m["name"]]
            m["mod"].weight.copy_(e["w"].cuda())
            if e["b"] is not None: m["mod"].bias = torch.nn.Parameter(e["b"].cuda(), requires_grad=False)
    seq.h.copy_(stt["h"].cuda())
    return stt


def _rot_matrix(n, seed=0):
    """Fixed Haar-random orthogonal matrix (float64) for the residual-stream basis change A_rot."""
    key = f"rotR_{n}_{seed}"
    if key not in H.S:
        g = torch.Generator(device="cuda"); g.manual_seed(seed)
        A = torch.randn(n, n, dtype=torch.float64, device="cuda", generator=g)
        Q, Rr = torch.linalg.qr(A)
        H.S[key] = Q * torch.sign(torch.diagonal(Rr))[None, :]
    return H.S[key]


@torch.no_grad()
def ldlq_pass(mats, spec, mult, st, drop=(), log_every=8, only=None, ckpt=None, ckpt_every=4):
    """Sequential E8-LDLQ over the whole text model. `mats` from plan_mats (may carry rows/cols subsets).
    Embedding: no-feedback E8 rounding (lookup table, no input Gram). lm_head: LDLQ on final hidden Gram.
    ckpt: directory for resumable checkpoints (every `ckpt_every` layers); an existing checkpoint is resumed."""
    lm = H.S["lm"]; t0 = time.time()
    by = {}
    for m in mats: by.setdefault(m["layer"], {})[m["kind"]] = m
    A = dict(tot=0.0, npar=0, bykind={})
    def acc(m, res):
        b, n = res; A["tot"] += b; A["npar"] += n
        k = A["bykind"].setdefault(m["kind"], [0.0, 0]); k[0] += b; k[1] += n
    resume = ckpt is not None and os.path.exists(f"{ckpt}/state.pt")
    # A_rot: exact change of basis of the residual stream h' = R h (RMSNorm gains folded into the readers).
    # Emulated in the ORIGINAL graph: each matrix is quantized in the rotated coordinates
    #   reader  W' = W diag(g) R^T  (inputs R N(h), Gram R diag(1/g) H diag(1/g) R^T)  -> decoded  Q' R diag(1/g)
    #   writer  W' = R W                                                             -> decoded  R^T Q'
    #   embed   E' = E R^T                                                           -> decoded  Q' R
    # which is function-identical to running the rotated graph with the quantized rotated weights.
    rot = bool(spec.get("rot", False))
    if rot:
        R64 = _rot_matrix(lm.embed_tokens.weight.shape[1], spec.get("rot_seed", 0)); R32 = R64.float()
        assert not spec.get("bias_corr", False)
    def rows_mm(A, B, pre=None, chunk=32768):          # (A * pre) @ B by row chunks (the 248320-row tables stay lean)
        out = torch.empty(A.shape[0], B.shape[1], dtype=torch.float32, device=A.device)
        for j in range(0, A.shape[0], chunk):
            a = A[j:j + chunk].float()
            if pre is not None: a = a * pre
            out[j:j + chunk] = a @ B
        return out
    def back_inplace(Q, g=None, chunk=32768):          # Q <- (Q @ R) / g , in place by row chunks
        for j in range(0, Q.shape[0], chunk):
            q = Q[j:j + chunk] @ R32
            Q[j:j + chunk] = q if g is None else q / g
        return Q
    def as_reader(m, gam):
        # decoded weight = Q' R acting on the gain-less normalised input: the RMSNorm gain of this reader group is
        # set to 1 (weight 0) once the group is quantized (see unit_gain) -- no division by gains (some are 0).
        m["W_override"] = rows_mm(m["mod"].weight, R32.T, pre=gam)
        m["post"] = lambda Q: back_inplace(Q)
    def unit_gain(norm):
        norm.weight.zero_()
    def as_writer(m):
        m["W_override"] = R32 @ m["mod"].weight.float()
        m["post"] = lambda Q: R32.T @ Q
    def clean(ms):
        for m in ms: m.pop("W_override", None); m.pop("post", None)
    def reader_H(Hm):
        return R64 @ Hm @ R64.T
    if -1 in by and only is None and not resume:
        me = by[-1]["embed"]
        if rot:
            me["W_override"] = rows_mm(me["mod"].weight, R32.T); me["post"] = lambda Q: back_inplace(Q)
        acc(me, H.quantize_matrix(me, spec, mult, st)); clean([me]); torch.cuda.empty_cache()
    seq = Seq(); start = 0; pending = [-1]
    if resume:
        stt = _ckpt_load(ckpt, by, seq)
        A.update(tot=stt["tot"], npar=stt["npar"], bykind=stt["bykind"]); start = stt["i"] + 1; pending = []
        if rot:
            for j in range(start):
                if j in by and j not in drop: lm.layers[j].input_layernorm.weight.zero_(); lm.layers[j].post_attention_layernorm.weight.zero_()
            if os.path.exists(f"{ckpt}/layer_64.pt"): lm.norm.weight.zero_()
        print("  resumed from checkpoint: next layer %d, bpw so far %.4f" % (start, A["tot"] / max(1, A["npar"])), flush=True)
    bc = spec.get("bias_corr", False)
    for i, L in enumerate(lm.layers):
        if i < start: continue
        if only is not None and i > max(only): break
        if i in drop or i not in by or (only is not None and i not in only):
            seq.advance(i); pending.append(i)
        else:
            d = by[i]
            if hasattr(L, "linear_attn"):
                first, ins, outk = L.linear_attn.in_proj_qkv, ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"), "out_proj"
            else:
                first, ins, outk = L.self_attn.q_proj, ("q_proj", "k_proj", "v_proj"), "o_proj"
            def G(mod):
                Hm = seq.gram(i, mod, center=bool(bc)); mu = seq.last_mu   # centred covariance when the bias carries the mean
                return [Hm], (mu if bc else None)
            def group(mod, kinds, gam, norm=None):
                def run():
                    if rot:
                        # Gram of the gain-less normalised input N(h): measured with the gain temporarily set to 1
                        saved = norm.weight.clone(); norm.weight.zero_()
                        try: hh, mu = G(mod)
                        finally: norm.weight.copy_(saved)
                        hh[0] = reader_H(hh[0])
                        for k in kinds: as_reader(d[k], gam)
                    else:
                        hh, mu = G(mod)
                    try:
                        out_ = H.quantize_group([d[k] for k in kinds], spec, mult, st, hh, mu_override=mu)
                    finally:
                        clean([d[k] for k in kinds])
                    if rot: unit_gain(norm)                       # readers now hold Q' R ; gain folded
                    return out_
                for k, res in zip(kinds, _retry(run, f"L{i} {kinds[0]}")): acc(d[k], res)
            def single(k):
                def run():
                    hh, mu = G(d[k]["mod"])
                    if rot: as_writer(d[k])
                    try:
                        return H.quantize_matrix(d[k], spec, mult, st, H=hh, mu_override=mu)
                    finally:
                        clean([d[k]])
                acc(d[k], _retry(run, f"L{i} {k}"))
            g_in = (1.0 + L.input_layernorm.weight.float()) if rot else None
            g_mlp = (1.0 + L.post_attention_layernorm.weight.float()) if rot else None
            group(first, ins, g_in, L.input_layernorm); single(outk)
            group(L.mlp.gate_proj, ("gate_proj", "up_proj"), g_mlp, L.post_attention_layernorm); single("down_proj")
            torch.cuda.empty_cache()
            seq.advance(i); pending.append(i)
        if ckpt is not None and only is None and (i % ckpt_every == ckpt_every - 1 or i == 63):
            _ckpt_save(ckpt, i, pending, by, seq, A); pending = []
        if i % log_every == 0 or i == 63:
            print("  ldlq layer %d done, %.0fs elapsed, bpw so far %.4f, peak %.1f GiB" % (i, time.time() - t0, A["tot"] / max(1, A["npar"]), torch.cuda.max_memory_allocated() / 2**30), flush=True)
    if 64 in by and only is None and not (resume and os.path.exists(f"{ckpt}/layer_64.pt")):
        def run():
            Hm = seq.final_gram(center=bool(bc)); mu = seq.last_mu if bc else None
            if rot:
                gf = 1.0 + lm.norm.weight.float()
                saved = lm.norm.weight.clone(); lm.norm.weight.zero_()
                try: Hm = seq.final_gram()
                finally: lm.norm.weight.copy_(saved)
                Hm = reader_H(Hm); as_reader(by[64]["lm_head"], gf)
            try:
                out_ = H.quantize_matrix(by[64]["lm_head"], spec, mult, st, H=[Hm], mu_override=mu)
            finally:
                clean([by[64]["lm_head"]])
            if rot: unit_gain(lm.norm)
            return out_
        acc(by[64]["lm_head"], _retry(run, "lm_head"))
        if ckpt is not None:
            _ckpt_save(ckpt, 63, [64], by, seq, A)
    del seq
    return dict(bits=A["tot"], params=A["npar"], bpw=A["tot"] / A["npar"], bykind={k: v[0] / v[1] for k, v in A["bykind"].items()}, seconds=time.time() - t0)


@torch.no_grad()
def rotate_model(seed=0, chunk=32768):
    """A_rot, native form: rewrite the whole text path in a rotated residual basis h' = R h (R Haar orthogonal).
    embed <- E R^T ; readers <- W diag(1+w) R^T and their RMSNorm gain <- 1 ; writers <- R W ;
    lm_head <- W diag(1+w_final) R^T and final norm gain <- 1.  Exact in real arithmetic (N(Rh) = R N(h));
    stored back in the model dtype (BF16), so the equivalence residual must be measured."""
    lm = H.S["lm"]; R = _rot_matrix(lm.embed_tokens.weight.shape[1], seed).float(); Rt = R.T.contiguous()
    def big(W, g=None):
        for j in range(0, W.shape[0], chunk):
            a = W[j:j + chunk].float()
            if g is not None: a = a * g
            W[j:j + chunk] = (a @ Rt).to(W.dtype)
    big(lm.embed_tokens.weight)
    for L in lm.layers:
        gd = hasattr(L, "linear_attn")
        ins = ([L.linear_attn.in_proj_qkv, L.linear_attn.in_proj_z, L.linear_attn.in_proj_a, L.linear_attn.in_proj_b] if gd
               else [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj])
        for norm, mods in ((L.input_layernorm, ins), (L.post_attention_layernorm, [L.mlp.gate_proj, L.mlp.up_proj])):
            g = 1.0 + norm.weight.float()
            for mod in mods: mod.weight.copy_(((mod.weight.float() * g) @ Rt).to(mod.weight.dtype))
            norm.weight.zero_()
        for mod in ((L.linear_attn.out_proj if gd else L.self_attn.o_proj), L.mlp.down_proj):
            mod.weight.copy_((R @ mod.weight.float()).to(mod.weight.dtype))
    big(H.S["model"].lm_head.weight, 1.0 + lm.norm.weight.float()); lm.norm.weight.zero_()


def run_ldlq_variant(vid, spec, mult, drop=(), keep=None, method="", prepare=None):
    """Crash-safe full variant: (resumable) LDLQ pass -> selection + sealed confirmation -> long-context check.
    The checkpoint directory is removed only after every result file has been written."""
    import json
    st = H.S["st"]; target = H.S["target_bits"]; ckpt = f"/workspace/work/ckpt/{vid}"
    os.makedirs("/workspace/work/ckpt", exist_ok=True)
    for other in os.listdir("/workspace/work/ckpt"):            # free disk from variants that are finished
        if other != vid and os.path.exists(H.OUT + f"/variant_{other}.json"):
            shutil.rmtree(f"/workspace/work/ckpt/{other}", ignore_errors=True)
    free = shutil.disk_usage("/workspace/work").free / 2**30
    if free < 62 and not os.path.exists(ckpt): raise RuntimeError("not enough disk for a checkpointed pass: %.0f GiB free" % free)
    H.restore()
    if prepare is not None: prepare()           # exact re-parametrisation applied to the BF16 model before the pass
    mats = plan_mats(keep=keep, drop=drop)
    torch.cuda.reset_peak_memory_stats()
    info = ldlq_pass(mats, spec, mult, st, drop=drop, ckpt=ckpt)
    info.update(spec=dict(spec, method=method), mult=mult, target_bits=target, bits_over_target=info["bits"] / target - 1, bytes=info["bits"] / 8)
    print("%s bpw=%.4f bits/target-1=%+.4f%% pass %.0fs peak %.1f GiB" % (vid, info["bpw"], 100 * info["bits_over_target"], info["seconds"], torch.cuda.max_memory_allocated() / 2**30),
          {k: round(v, 3) for k, v in info["bykind"].items()}, flush=True)
    H.run_variant(vid, info)
    if "long_teach" in H.S:
        rl = H.evaluate_long(); json.dump(rl, open(H.OUT + f"/long_{vid}.json", "w"))
        print("%s LONG (8192 tok) KL by 1024-bucket:" % vid, [round(v, 4) for v in rl["kl_by_bucket"]], "mean %.4f" % rl["kl"], flush=True)
    return info


@torch.no_grad()
def seq_parity(n_docs=16):
    """Sanity: propagate BF16 hidden states layer by layer with Seq and compare with the model's own forward."""
    lm = H.S["lm"]; seq = Seq(n_docs=n_docs)
    for i in range(len(lm.layers)): seq.advance(i)
    ids = H.S["data"]["cal"][:n_docs]
    ref = torch.cat([lm(input_ids=ids[b:b + seq.bs].cuda()).last_hidden_state for b in range(0, n_docs, seq.bs)])
    got = torch.cat([lm.norm(seq.h[b:b + seq.bs]) for b in range(0, n_docs, seq.bs)])
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    return dict(rel_err=rel, max_abs=float((got.float() - ref.float()).abs().max()), identical=bool(torch.equal(got, ref)))
