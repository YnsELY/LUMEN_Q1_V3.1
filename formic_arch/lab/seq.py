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
    def gram(self, i, module, tf32=True):
        """Gram (sum x x^T / ntok, fp64) of the input of `module` when layer i runs on current hidden."""
        L = self.lm.layers[i]; acc = {}
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
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
        self.last_mu = (acc["m"] / self.ntok).float()
        return (acc["g"].double() / self.ntok)

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
    def final_gram(self):
        acc = None; msum = 0
        for b in range(0, self.n, self.bs):
            x = self.lm.norm(self.h[b:b + self.bs]).reshape(-1, self.h.shape[-1]).float()
            g = x.T @ x
            acc = g if acc is None else acc.add_(g)
            msum = msum + x.sum(0).double()
        self.last_mu = (msum / self.ntok).float()
        return acc.double() / self.ntok


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
    if -1 in by and only is None and not resume:
        acc(by[-1]["embed"], H.quantize_matrix(by[-1]["embed"], spec, mult, st))
    seq = Seq(); start = 0; pending = [-1]
    if resume:
        stt = _ckpt_load(ckpt, by, seq)
        A.update(tot=stt["tot"], npar=stt["npar"], bykind=stt["bykind"]); start = stt["i"] + 1; pending = []
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
                Hm = seq.gram(i, mod); mu = seq.last_mu
                if bc: Hm.sub_(torch.outer(mu.double(), mu.double()))      # the mean direction is carried by the bias
                return [Hm], (mu if bc else None)
            def group(mod, kinds):
                def run():
                    hh, mu = G(mod)
                    return H.quantize_group([d[k] for k in kinds], spec, mult, st, hh, mu_override=mu)
                for k, res in zip(kinds, _retry(run, f"L{i} {kinds[0]}")): acc(d[k], res)
            def single(k):
                def run():
                    hh, mu = G(d[k]["mod"])
                    return H.quantize_matrix(d[k], spec, mult, st, H=hh, mu_override=mu)
                acc(d[k], _retry(run, f"L{i} {k}"))
            group(first, ins); single(outk); group(L.mlp.gate_proj, ("gate_proj", "up_proj")); single("down_proj")
            torch.cuda.empty_cache()
            seq.advance(i); pending.append(i)
        if ckpt is not None and only is None and (i % ckpt_every == ckpt_every - 1 or i == 63):
            _ckpt_save(ckpt, i, pending, by, seq, A); pending = []
        if i % log_every == 0 or i == 63:
            print("  ldlq layer %d done, %.0fs elapsed, bpw so far %.4f, peak %.1f GiB" % (i, time.time() - t0, A["tot"] / max(1, A["npar"]), torch.cuda.max_memory_allocated() / 2**30), flush=True)
    if 64 in by and only is None and not (resume and os.path.exists(f"{ckpt}/layer_64.pt")):
        def run():
            Hm = seq.final_gram(); mu = seq.last_mu if bc else None
            if mu is not None: Hm.sub_(torch.outer(mu.double(), mu.double()))
            return H.quantize_matrix(by[64]["lm_head"], spec, mult, st, H=[Hm], mu_override=mu)
        acc(by[64]["lm_head"], _retry(run, "lm_head"))
        if ckpt is not None:
            _ckpt_save(ckpt, 63, [64], by, seq, A)
    del seq
    return dict(bits=A["tot"], params=A["npar"], bpw=A["tot"] / A["npar"], bykind={k: v[0] / v[1] for k, v in A["bykind"].items()}, seconds=time.time() - t0)


def run_ldlq_variant(vid, spec, mult, drop=(), keep=None, method=""):
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
    H.restore(); mats = plan_mats(keep=keep, drop=drop)
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
