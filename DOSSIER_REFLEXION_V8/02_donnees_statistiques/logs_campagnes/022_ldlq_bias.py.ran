# A_b x Q0: same E8-LDLQ pass, but every linear gets an FP16 bias b=(W-Q)mu (student mean, sequential) and the
# LDLQ Gram is the centred covariance (the mean direction is carried exactly by the bias).
import time, importlib, seq
importlib.reload(seq); SQ = seq
st = H.S["st"]; target = H.S["target_bits"]
# bias vectors cost 16 bits per output row -> remove them from the symbol budget via the multiplier
mats = SQ.plan_mats(); rows = sum(m["mod"].weight.shape[0] for m in mats if m["kind"] != "embed")
m_b = H.S["mult_Q0ldlq"] * 2 ** (16 * rows / H.S["N_text"])
H.restore(); torch.cuda.reset_peak_memory_stats()
import os, json
if os.path.exists("/workspace/work/data/long_tokens.pt") and "long_teach" not in H.S:
    print("long teacher cache:", H.build_teacher_long(), flush=True)
spec = dict(rule="lumen", bias_corr=True)
info = SQ.ldlq_pass(mats, spec, m_b, st)
info.update(spec=dict(spec, method="E8-LDLQ sequential, centred Gram + FP16 bias"), mult=m_b, target_bits=target, bits_over_target=info["bits"]/target-1, bytes=info["bits"]/8)
print("Ab_Q0ldlq bpw=%.4f bits/target-1=%+.4f%% pass %.0fs" % (info["bpw"], 100*info["bits_over_target"], info["seconds"]), {k: round(v, 3) for k, v in info["bykind"].items()}, flush=True)
H.run_variant("Ab_Q0ldlq", info)

if "long_teach" in H.S:
    rl = H.evaluate_long(); json.dump(rl, open(H.OUT+"/long_Ab_Q0ldlq.json", "w"))
    print("Ab_Q0ldlq LONG (8192 tok) KL by 1024-bucket:", [round(v, 4) for v in rl["kl_by_bucket"]], "mean %.4f" % rl["kl"])
