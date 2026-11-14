# Iso-byte quantized versions of the width-pruned structures (no-feedback E8, control recipe, bytes reinvested).
import time, json
st = H.S["st"]; target = H.S["target_bits"]
for frac, pr in H.S["prunes"].items():
    H.restore(); SQ.apply_prune(pr); mats = SQ.plan_mats(keep=pr["keep"])
    spec = dict(rule="lumen")
    mult = H.calibrate_mult(mats, spec, target, st, lo=0.2, hi=2.0, iters=11)
    info = H.quantize_all(mats, spec, mult, st)
    info.update(spec=spec, frac=frac, nk=pr["nk"], mult=mult, target_bits=target, bits_over_target=info["bits"]/target-1)
    print("A1_w%s_Q0rtn mult=%.4f bpw(on kept)=%.4f bits/target-1=%+.4f%%" % (frac, mult, info["bpw"], 100*info["bits_over_target"]), flush=True)
    H.run_variant(f"A1_w{frac}_Q0rtn", info)
H.restore()
