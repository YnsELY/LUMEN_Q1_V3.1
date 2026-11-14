import importlib, sys, time, json, math
import formic_q, harness, seq
_S = dict(harness.S); bf = _S["_bias_free"]
importlib.reload(formic_q); importlib.reload(harness); harness.S.update(_S); H = harness; importlib.reload(seq); SQ = seq
H.S["_bias_free"] = bf
st = H.S["st"]; mats = H.qmats(); target = H.S["target_bits"]; dbits = H.S["dbits"]
print("restore", H.restore(), "biases left:", sum(m.bias is not None for m in bf))
def run(vid, spec):
    t = time.time(); H.restore()
    mult = H.calibrate_mult(mats, spec, target, st, lo=0.25, hi=2.0, iters=11)
    info = H.quantize_all(mats, spec, mult, st)
    info.update(spec=spec, mult=mult, target_bits=target, bits_over_target=info["bits"]/target-1, bytes=info["bits"]/8)
    print(vid, "mult=%.4f bpw=%.4f bits/target-1=%+.4f%%" % (mult, info["bpw"], 100*info["bits_over_target"]), {k: round(v, 3) for k, v in info["bykind"].items()}, flush=True)
    H.S.setdefault("mults", {})[vid] = mult
    return H.run_variant(vid, info)
H.S["run_rtn"] = run
run("A0_Qalloc_rtn", dict(rule="lumen", kind_dbits=dbits))
run("Ab_Qalloc_rtn", dict(rule="lumen", kind_dbits=dbits, bias_corr=True))
# iso-byte depth data point: 2 cheapest layers removed, saved bytes reinvested (control recipe)
drop = H.S["drop_order"][:2]
H.restore(); SQ.drop_layers(drop); m2 = SQ.plan_mats(drop=drop)
mult = H.calibrate_mult(m2, dict(rule="lumen"), target, st, lo=0.25, hi=2.0, iters=11)
info = H.quantize_all(m2, dict(rule="lumen"), mult, st); info.update(drop=drop, mult=mult, target_bits=target, bits_over_target=info["bits"]/target-1)
print("A2_drop2_Q0rtn mult=%.4f bpw(on kept)=%.4f bits/target-1=%+.4f%%" % (mult, info["bpw"], 100*info["bits_over_target"]))
H.run_variant("A2_drop2_Q0rtn", info)
H.restore()
