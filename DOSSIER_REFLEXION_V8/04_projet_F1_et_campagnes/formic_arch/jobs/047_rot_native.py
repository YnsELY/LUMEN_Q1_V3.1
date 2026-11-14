# A_rot, native: the model is rewritten in the rotated residual basis and runs there (no mapping back).
# 1) equivalence check of the rotated BF16 model against the teacher; 2) the standard E8-LDLQ pass on it.
import importlib, harness, seq, json, formic_q
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq
H.restore(); SQ.rotate_model(0)
H.run_variant("Arot_bf16", dict(note="BF16 model rewritten in the rotated residual basis, no quantization (equivalence residual in BF16 storage)"))
rl = H.evaluate_long(); print("Arot_bf16 LONG KL mean %.2e" % rl["kl"], flush=True)
# second seed: is the residual a property of the transformation or of one particular rotation?
H.restore(); SQ.rotate_model(1); r = H.evaluate("select"); print("rotation seed 1, BF16: KL=%.2e agree=%.4f" % (r["kl"], r["agree"]), flush=True)
spec = dict(rule="lumen", match_bpw=2.8158)
SQ.run_ldlq_variant("ArotN_Q0ldlq", spec, H.S["mult_Q0ldlq"], method="standard E8-LDLQ on the model natively rewritten in a rotated residual basis; per-matrix rate matching", prepare=lambda: SQ.rotate_model(0))
