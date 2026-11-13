# Quantizer varied, allocation fixed: Arki05's own per-tensor bits (read from its GGUF header) applied to E8-LDLQ.
# Every matrix gets the competitor's bytes minus the measured physical overhead of our codec (0.005 bpw).
# Codec extension for the tensors above ~3.6 bpw: shell table limit 16384 (ideal cost; CPU reference stops at 128);
# in_proj_a/b (Q8_0 in the competitor): 8-bit scalar + FP16 row step.
import sys, importlib, json
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
formic_q.set_limit(16384)
al = V.arki_alloc(); json.dump(al, open(V.OUT7 + "/alloc_arki05_2.82bpw.json", "w"), indent=0)
tgt = {k: (v if v >= 6.0 and "in_proj_" in k and k.endswith(("_a", "_b")) else v - 0.005) for k, v in al.items()}
tgt["default"] = 2.808
print("matrices with a transplanted target:", len(al), "range %.2f..%.2f" % (min(al.values()), max(al.values())), flush=True)
spec = dict(rule="lumen", match_bpw=tgt, match_offset=-0.011)
V.run("L1_arkialloc", spec, H.S["mult_Q0ldlq"], method="E8-LDLQ with Arki05-2.82bpw per-tensor bit allocation (bytes matched per matrix)")
torch.save(H.S.get("dstat", {}), "/workspace/work/dstat_L1.pt")
formic_q.set_limit(128)
