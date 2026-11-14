# Why did LDL-pivot column scaling not help in KL?  Local check at matched bits (BF16 prefix, real inputs):
# activation-weighted output error of E8-LDLQ with the reference steps vs with column scales (clip 4 and 2).
import importlib, seq, json
import formic_q as fq
importlib.reload(seq); SQ = seq
H.restore(); lm = H.S["lm"]; out = {}
sq = SQ.Seq(n_docs=144); want = {3, 16, 31, 47, 62}
def run(W, Hm, c, target_bpw):
    m_, n_ = W.shape; lo, hi = 0.2, 2.5; Wc = W if c is None else W * c
    for _ in range(9):                                  # bisection on the real LDLQ bits of this matrix
        mid = (lo * hi) ** 0.5; r = fq.row_steps(Wc, mid)
        Q, hist, _ = fq.quantize_ldlq(W, Hm.clone(), r, c, damp=0.01)
        bpw = fq.matrix_bits(hist, m_, n_, has_col_scale=c is not None) / (m_ * n_)
        if bpw > target_bpw: lo = mid
        else: hi = mid
    E = (W - Q).double(); Wd = W.double()
    return bpw, float(((E @ Hm) * E).sum() / ((Wd @ Hm) * Wd).sum()), float(hist[-1] / hist.sum())
for i, L in enumerate(lm.layers):
    if i in want:
        gd = hasattr(L, "linear_attn")
        for nm, mod in (("mixer_in", (L.linear_attn.in_proj_z if gd else L.self_attn.k_proj)), ("mixer_out", (L.linear_attn.out_proj if gd else L.self_attn.o_proj)), ("mlp_down", L.mlp.down_proj)):
            Hm = sq.gram(i, mod); W = mod.weight.float()
            if nm == "mlp_down": W = W[:1024]          # rows are independent: a row subset is enough for the proxy
            if W.shape[0] > 2048: W = W[:2048]
            res = {"lumen": run(W, Hm, None, 2.82)}
            for clip in (2.0, 4.0, 64.0):
                c = H._ldlcol(Hm, dict(col_clip=clip)); res[f"ldlcol_clip{clip:g}"] = run(W, Hm, c, 2.82)
            out[f"L{i}.{nm}"] = res
            print("L%-2d %-9s" % (i, nm), " | ".join("%s: bpw %.3f err %.5f esc %.1e" % (k, *v) for k, v in res.items()), flush=True)
            del Hm
    sq.advance(i)
    if i >= max(want): break
json.dump(out, open(H.OUT + "/diag_ldlcol_proxy.json", "w"), indent=1); del sq
