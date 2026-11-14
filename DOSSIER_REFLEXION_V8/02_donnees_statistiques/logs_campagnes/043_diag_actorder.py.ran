# LDLQ processing order: does sorting the input columns by decreasing second moment (exact permutation, GPTQ act_order)
# reduce the sum of LDL pivots, and the real activation-weighted error at matched bits?  BF16 prefix, real inputs.
import importlib, harness, seq, json, math, formic_q
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; fq = formic_q
H.restore(); lm = H.S["lm"]; out = {}
sq = SQ.Seq(n_docs=144); want = {3, 16, 31, 32, 62}
def amgm(Hm): d = fq.ldl_pivots(Hm, 0.01); return float(d.mean() / d.log().mean().exp()), float(d.sum())
def run(W, Hm, target_bpw=2.82):
    m_, n_ = W.shape; lo, hi = 0.3, 1.2
    prep = fq.ldl_prepare(Hm.clone(), None, 0.01)            # factor once per ordering
    for _ in range(6):
        mid = (lo * hi) ** 0.5; r = fq.row_steps(W, mid)
        Q, hist, _ = fq.quantize_ldlq(W, None, r, None, prep=prep)
        bpw = fq.matrix_bits(hist, m_, n_) / (m_ * n_)
        if bpw > target_bpw: lo = mid
        else: hi = mid
    E = (W - Q).double(); Wd = W.double()
    return bpw, float(((E @ Hm) * E).sum() / ((Wd @ Hm) * Wd).sum())
for i, L in enumerate(lm.layers):
    if i in want:
        gd = hasattr(L, "linear_attn")
        for nm, mod in (("mixer_in", (L.linear_attn.in_proj_z if gd else L.self_attn.k_proj)), ("mixer_out", (L.linear_attn.out_proj if gd else L.self_attn.o_proj)),
                        ("mlp_gate", L.mlp.gate_proj), ("mlp_down", L.mlp.down_proj)):
            Hm = sq.gram(i, mod); W = mod.weight.float()[:512]
            rec = {}
            for tag, perm in (("native", None), ("desc", torch.argsort(torch.diagonal(Hm), descending=True)), ("asc", torch.argsort(torch.diagonal(Hm), descending=False))):
                Hp = Hm if perm is None else Hm[perm][:, perm]; Wp = W if perm is None else W[:, perm]
                a, tr = amgm(Hp); b, e = run(Wp, Hp)
                rec[tag] = dict(am_over_gm=a, sum_pivots=tr, bpw=b, rel_proxy_err=e); del Hp
            out[f"L{i}.{nm}"] = rec
            print("L%-2d %-9s" % (i, nm), " | ".join("%s: AM/GM %.2f err %.5f (bpw %.3f)" % (k, v["am_over_gm"], v["rel_proxy_err"], v["bpw"]) for k, v in rec.items()), flush=True)
            del Hm; torch.cuda.empty_cache()
    sq.advance(i)
    if i >= max(want): break
json.dump(out, open(H.OUT + "/diag_actorder.json", "w"), indent=1); del sq
