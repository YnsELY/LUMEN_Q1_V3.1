# A_b2 x Q0: E8-LDLQ with centred Gram; FP16 bias anchors each linear's mean output to the BF16 teacher's
# mean output (b = W mu_teacher - Q mu_student): removes the accumulated drift of the mean activation.
import time, importlib, seq, harness, json
_S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness; importlib.reload(seq); SQ = seq
st = H.S["st"]; target = H.S["target_bits"]
mats = SQ.plan_mats(); rows = sum(m["mod"].weight.shape[0] for m in mats if m["kind"] != "embed")
m_b = H.S["mult_Q0ldlq"] * 2 ** (16 * rows / H.S["N_text"])
H.restore()
spec = dict(rule="lumen", bias_corr="teacher")
info = SQ.ldlq_pass(mats, spec, m_b, st)
info.update(spec=dict(spec, method="E8-LDLQ sequential, centred Gram + FP16 bias anchored on teacher mean output"), mult=m_b, target_bits=target, bits_over_target=info["bits"]/target-1, bytes=info["bits"]/8)
print("Ab2_Q0ldlq bpw=%.4f bits/target-1=%+.4f%% pass %.0fs" % (info["bpw"], 100*info["bits_over_target"], info["seconds"]), flush=True)
H.run_variant("Ab2_Q0ldlq", info)
if "long_teach" in H.S:
    rl = H.evaluate_long(); json.dump(rl, open(H.OUT+"/long_Ab2_Q0ldlq.json", "w"))
    print("Ab2_Q0ldlq LONG KL by 1024-bucket:", [round(v, 4) for v in rl["kl_by_bucket"]], "mean %.4f" % rl["kl"])
