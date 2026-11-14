# How much can ANY exact diagonal / orthogonal change of the input basis still give to E8-LDLQ?
# LDLQ proxy loss at fixed rate ~ sum of LDL pivots D_j (uniform lattice noise per column); a diagonal rescaling
# c_j = sqrt(D_j) flattens the pivots, so the classical transform-coding headroom is AM(D)/GM(D).
# Also: is the Gram dominated by a few token positions?   BF16 weights, calibration split, real layer inputs.
import importlib, seq, json, math, time
importlib.reload(seq); SQ = seq
import formic_q as fq
H.restore(); lm = H.S["lm"]; out = {}
sq = SQ.Seq(n_docs=144)
want = {1, 2, 3, 16, 31, 32, 47, 62, 63}
def pivots(Hm, damp=0.01):
    Hm = Hm.clone(); Hm.diagonal().add_(damp * Hm.diagonal().mean())
    C = torch.linalg.cholesky(Hm.flip(0, 1)); d = (torch.diagonal(C) ** 2).flip(0)
    return d
def stats(d):
    am = float(d.mean()); gm = float(d.log().mean().exp())
    return dict(am_over_gm=am/gm, headroom_bits=0.5*math.log2(am/gm), top8_share=float(d.topk(8).values.sum()/d.sum()))
g = torch.Generator(device="cuda"); g.manual_seed(0)
for i, L in enumerate(lm.layers):
    if i in want:
        mods = {"mixer_in": (L.linear_attn.in_proj_qkv if hasattr(L, "linear_attn") else L.self_attn.q_proj),
                "mixer_out": (L.linear_attn.out_proj if hasattr(L, "linear_attn") else L.self_attn.o_proj),
                "mlp_in": L.mlp.gate_proj, "mlp_down": L.mlp.down_proj}
        for nm, mod in mods.items():
            Hm = sq.gram(i, mod); n = Hm.shape[0]; r = {}
            r["orig"] = stats(pivots(Hm))
            Hc = sq.gram(i, mod, center=True); r["centred"] = stats(pivots(Hc)); del Hc
            if n <= 6144:      # random orthogonal rotation of the input basis (incoherence), cheap sizes only
                Qr = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device="cuda", generator=g))[0]
                r["rotated"] = stats(pivots(Qr @ Hm @ Qr.T)); del Qr
            dg = torch.diagonal(Hm); r["diag_am_over_gm"] = float(dg.mean()/dg.log().mean().exp())
            r["trace"] = float(dg.sum())
            # per-position share of the input energy (is the Gram dominated by a few positions?)
            acc = {}
            def hook(m_, inp):
                x = inp[0].float(); acc["e"] = acc.get("e", 0) + x.pow(2).sum(-1).sum(0).double(); raise SQ._Stop()
            hk = mod.register_forward_pre_hook(hook)
            for b in range(0, sq.n, sq.bs):
                try: L(sq.h[b:b + sq.bs], **sq._kw(i))
                except SQ._Stop: pass
            hk.remove(); e = acc["e"]
            r["pos0_share"] = float(e[0]/e.sum()); r["pos_first8_share"] = float(e[:8].sum()/e.sum()); r["pos_max_over_median"] = float(e.max()/e.median())
            out[f"L{i}.{nm}"] = r; del Hm
            print("L%-2d %-9s n=%5d | AM/GM pivots: orig %.2f (%.2f bit)  centred %.2f (%.2f bit)%s | diag AM/GM %.1f | pos0 share %.4f, first8 %.4f, max/med %.1f" % (
                i, nm, n, r["orig"]["am_over_gm"], r["orig"]["headroom_bits"], r["centred"]["am_over_gm"], r["centred"]["headroom_bits"],
                ("  rotated %.2f (%.2f bit)" % (r["rotated"]["am_over_gm"], r["rotated"]["headroom_bits"])) if "rotated" in r else "",
                r["diag_am_over_gm"], r["pos0_share"], r["pos_first8_share"], r["pos_max_over_median"]), flush=True)
    sq.advance(i)
    if i >= max(want): break
json.dump(out, open(H.OUT + "/diag_ldl_headroom.json", "w"), indent=1)
del sq; torch.cuda.empty_cache()
