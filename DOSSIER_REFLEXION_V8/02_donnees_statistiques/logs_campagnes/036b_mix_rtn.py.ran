# Recipe-axis check (no-feedback E8): activation column scaling only on the token-mixer input projections
# (the two families where it measured useful in isolation), with and without the bias architecture A_b.
import time, json
st = H.S["st"]; mats = H.qmats(); target = H.S["target_bits"]
ck = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "q_proj", "k_proj", "v_proj")
def run(vid, spec):
    H.restore()
    mult = H.calibrate_mult(mats, spec, target, st, lo=0.25, hi=2.0, iters=11)
    info = H.quantize_all(mats, spec, mult, st)
    info.update(spec=spec, mult=mult, target_bits=target, bits_over_target=info["bits"]/target-1, bytes=info["bits"]/8)
    print(vid, "mult=%.4f bpw=%.4f bits/target-1=%+.4f%%" % (mult, info["bpw"], 100*info["bits_over_target"]), flush=True)
    return H.run_variant(vid, info)
run("A0_Qmix_rtn", dict(rule="mix", col_kinds=ck))
run("Ab_Qmix_rtn", dict(rule="mix", col_kinds=ck, bias_corr=True))
H.restore()
