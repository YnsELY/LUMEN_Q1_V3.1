import json, time, importlib, seq
importlib.reload(seq); SQ = seq
st = H.S["st"]
H.S.setdefault("prunes", {})
for frac in (0.125, 0.25):
    t = time.time(); H.restore()
    pr = SQ.prune_width(frac, st)
    H.S["prunes"][frac] = pr
    re = pr["rel_err"]
    print("frac", frac, "nk", pr["nk"], "prune pass %.0fs" % (time.time()-t), "mean rel err zero=%.4f fit=%.4f" % (sum(v[0] for v in re.values())/64, sum(v[1] for v in re.values())/64))
    json.dump(dict(frac=frac, nk=pr["nk"], rel_err=re), open(H.OUT+f"/prune_width_{frac}.json", "w"))
    H.run_variant(f"A1_w{frac}_bf16", dict(frac=frac, nk=pr["nk"], note="BF16, SwiGLU channels removed by Taylor saliency + ridge LS refit of down_proj (sequential)"))
# no-compensation ablation for the smaller fraction
H.restore()
pr0 = SQ.prune_width(0.125, st, compensate=False, log=False)
H.run_variant("A1_w0.125_bf16_nocomp", dict(frac=0.125, note="BF16, Taylor pruning only, no refit"))
