# A0 x Q0 with the real Lumen algebra: E8 + block-8 LDLQ (sequential, student inputs), reference row-RMS steps.
import time, importlib, seq, json
importlib.reload(seq); SQ = seq
st = H.S["st"]; target = H.S["target_bits"]
m_ldlq = H.S["mult_Q0rtn"] * 2 ** (2.8326 - 2.8247)      # rate correction measured in the 3-layer pilot
H.S["mult_Q0ldlq"] = m_ldlq
H.restore(); mats = SQ.plan_mats()
torch.cuda.reset_peak_memory_stats()
info = SQ.ldlq_pass(mats, dict(rule="lumen"), m_ldlq, st)
info.update(spec=dict(rule="lumen", method="E8-LDLQ sequential"), mult=m_ldlq, target_bits=target, bits_over_target=info["bits"]/target-1, bytes=info["bits"]/8)
print("A0_Q0ldlq bpw=%.4f bits/target-1=%+.4f%% pass %.0fs peak %.1f GiB" % (info["bpw"], 100*info["bits_over_target"], info["seconds"], torch.cuda.max_memory_allocated()/2**30), {k: round(v, 3) for k, v in info["bykind"].items()}, flush=True)
H.run_variant("A0_Q0ldlq", info)
