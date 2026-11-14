# Structure of what remains AFTER the real E8-LDLQ (BF16 prefix, real inputs), to decide three families of ideas:
#  (a) low-rank protection / correction (QERA, SRR): share of the activation-weighted error in its top-r directions,
#      against the same bytes spent on E8 precision;
#  (b) exceptions / channel splitting (SpQR, OCS): concentration of the error on few entries / columns;
#  (c) factorise-then-quantise: truncation loss (activation metric) of the rank affordable at equal bytes.
import importlib, seq, json, math, time
import formic_q as fq
importlib.reload(seq); SQ = seq
H.restore(); lm = H.S["lm"]; out = {}; mult = H.S["mult_Q0ldlq"]
sq = SQ.Seq(n_docs=144); want = {1, 16, 31, 32, 47, 62}
def top_share(ev, rs=(16, 64, 256)):
    ev = ev.clamp_min(0).flip(0); tot = ev.sum()
    return {r: float(ev[:r].sum() / tot) for r in rs}
for i, L in enumerate(lm.layers):
    if i in want:
        gd = hasattr(L, "linear_attn")
        todo = [("mixer_in", (L.linear_attn.in_proj_qkv if gd else L.self_attn.q_proj)), ("mixer_out", (L.linear_attn.out_proj if gd else L.self_attn.o_proj)),
                ("mlp_gate", L.mlp.gate_proj), ("mlp_down", L.mlp.down_proj)]
        for nm, mod in todo:
            Hm = sq.gram(i, mod); W = mod.weight.float(); m_, n_ = W.shape
            r_ = fq.row_steps(W, mult)
            Q, hist, _ = fq.quantize_ldlq(W, Hm.clone(), r_, None, damp=0.01)
            E = (W - Q).double(); Wd = W.double()
            sig = float(((Wd @ Hm) * Wd).sum()); err = float(((E @ Hm) * E).sum())
            rec = dict(shape=[m_, n_], bpw=fq.matrix_bits(hist, m_, n_) / (m_ * n_), rel_proxy_err=err / sig, rel_frob_err=float(E.norm() / Wd.norm()))
            # (a) spectrum of the activation-weighted error and signal on the smaller side
            if m_ <= n_:
                ev_e = torch.linalg.eigvalsh((E @ Hm) @ E.T); ev_w = torch.linalg.eigvalsh((Wd @ Hm) @ Wd.T)
            else:
                lam, V = torch.linalg.eigh(Hm); Hh = (V * lam.clamp_min(0).sqrt()) @ V.T; del V
                A = E @ Hh; ev_e = torch.linalg.eigvalsh(A.T @ A); A = Wd @ Hh; ev_w = torch.linalg.eigvalsh(A.T @ A); del A, Hh
            rec["err_top_share"] = top_share(ev_e); rec["sig_top_share"] = top_share(ev_w)
            # same bytes in E8 precision: rank r FP16 factors cost 16 r (m+n) bits -> db bits/weight -> error x 4^-db
            rec["lowrank_vs_bits"] = {}
            for r in (16, 64, 256):
                db = 16 * r * (m_ + n_) / (m_ * n_)
                rec["lowrank_vs_bits"][r] = dict(cost_bpw=db, err_removed_by_ideal_rank_r=rec["err_top_share"][r], err_removed_by_same_bits=1 - 4 ** (-db))
            # (b) concentration: top 0.1% entries share of squared error; top-8 columns share of weighted error
            e2 = E.pow(2); k = max(1, e2.numel() // 1000)
            rec["top0.1pct_entries_err_share"] = float(e2.flatten().float().topk(k).values.sum() / e2.sum())
            colw = (e2.sum(0) * torch.diagonal(Hm)); rec["top8_cols_weighted_err_share"] = float(colw.topk(8).values.sum() / colw.sum())
            # (c) factorisation at equal bytes: rank r = bpw*m*n / ((m+n)*b) ; loss = signal energy outside top-r
            evs = ev_w.clamp_min(0).flip(0); tot = evs.sum(); rec["factor_equal_bytes"] = {}
            for b in (3.0, 4.0, 6.0):
                r = int(rec["bpw"] * m_ * n_ / ((m_ + n_) * b)); r = min(r, len(evs))
                rec["factor_equal_bytes"][b] = dict(rank=r, of=len(evs), truncation_energy_loss=float(1 - evs[:r].sum() / tot))
            out[f"L{i}.{nm}"] = rec
            print("L%-2d %-9s %5dx%-5d LDLQ rel proxy err %.4f (frob %.3f) | err in top16/64/256 dirs: %.3f/%.3f/%.3f ; rank-64 FP16 costs %.3f bpw -> same bits remove %.3f | top0.1%% entries %.4f, top8 cols %.4f | factorise@4b: rank %d/%d loses %.3f of signal" % (
                i, nm, m_, n_, rec["rel_proxy_err"], rec["rel_frob_err"], *[rec["err_top_share"][r] for r in (16, 64, 256)], rec["lowrank_vs_bits"][64]["cost_bpw"], rec["lowrank_vs_bits"][64]["err_removed_by_same_bits"],
                rec["top0.1pct_entries_err_share"], rec["top8_cols_weighted_err_share"], rec["factor_equal_bytes"][4.0]["rank"], rec["factor_equal_bytes"][4.0]["of"], rec["factor_equal_bytes"][4.0]["truncation_energy_loss"]), flush=True)
            del Hm, E, Wd, Q, W, e2; torch.cuda.empty_cache()
    sq.advance(i)
    if i >= max(want): break
# cross-layer shared basis: is the joint row space of adjacent layers lower-dimensional than each alone?
def erank(G):
    ev = torch.linalg.eigvalsh(G).clamp_min(0); p = ev / ev.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))
sh = {}
for a, b in ((16, 17), (30, 34), (40, 41)):
    for k in ("gate_proj", "down_proj"):
        Wa = getattr(lm.layers[a].mlp, k).weight.double(); Wb = getattr(lm.layers[b].mlp, k).weight.double()
        if k == "down_proj": Wa = Wa.T; Wb = Wb.T          # compare column spaces in the residual dimension
        Ga = Wa.T @ Wa; Gb = Wb.T @ Wb
        sh[f"{k} L{a}+L{b}"] = dict(erank_a=erank(Ga), erank_b=erank(Gb), erank_joint=erank(Ga + Gb))
        print("shared basis %s L%d+L%d: effective rank %.0f and %.0f alone, %.0f jointly (of 5120)" % (k, a, b, erank(Ga), erank(Gb), erank(Ga + Gb)))
out["shared_basis"] = sh
json.dump(out, open(H.OUT + "/diag_error_structure.json", "w"), indent=1)
del sq; torch.cuda.empty_cache()
