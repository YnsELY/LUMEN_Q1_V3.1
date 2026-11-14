import time, json, math
st = H.S["st"]; mats = H.qmats()
N = sum(m["mod"].weight.numel() for m in mats)
TARGET_BPW = 2.824          # Lumen "Max" symbol capacity 2.8328 minus measured codec overhead 0.009
target = TARGET_BPW * N
H.S["target_bits"] = target; H.S["N_text"] = N
def run(vid, spec, mats=mats, lo=0.25, hi=2.0):
    t=time.time()
    H.restore()
    mult = H.calibrate_mult(mats, spec, target, st, lo=lo, hi=hi, iters=11)
    info = H.quantize_all(mats, spec, mult, st)
    info.update(spec=spec, mult=mult, target_bits=target, bits_over_target=info["bits"]/target-1, bytes=info["bits"]/8)
    print(vid, "mult=%.4f bpw=%.4f bits/target-1=%+.4f%%" % (mult, info["bpw"], 100*info["bits_over_target"]), {k: round(v,3) for k,v in info["bykind"].items()}, "quant %.0fs"%(time.time()-t), flush=True)
    return H.run_variant(vid, info), mult
r0, m0 = run("A0_Q0rtn", dict(rule="lumen"))
H.S["mult_Q0rtn"] = m0
run("A0_Qcol_rtn", dict(rule="col"))
run("A0_Qfl_rtn", dict(rule="fisher_local"))
run("A0_Qfg_rtn", dict(rule="fisher_global", io_mult=m0), lo=1e-6, hi=1e-1)
