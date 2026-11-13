# A0 x Q_perm: same E8-LDLQ, but the columns of every matrix are processed in decreasing order of input second
# moment (exact permutation of the input basis of each linear, stored: 16 bits per column, counted).
import json
st = H.S["st"]; CTRL_BPW = json.load(open(H.OUT + "/variant_A0_Q0ldlq.json"))["info"]["bpw"]
docs3 = list(range(0, 72, 3))
H.restore(); mats = SQ.plan_mats()
SQ.ldlq_pass(mats, dict(rule="lumen"), H.S["mult_Q0ldlq"], st, only={1, 2, 3}, log_every=100)
r = H.evaluate("select", docs=docs3); print("reference: layers 1-3 only quantized, control recipe: KL=%.5f" % r["kl"], flush=True)
spec = dict(rule="lumen", perm=True, match_bpw=2.8110)
H.restore(); mats = SQ.plan_mats()
pil = SQ.ldlq_pass(mats, spec, H.S["mult_Q0ldlq"], st, only={1, 2, 3}, log_every=1)
print("pilot (layers 1-3) bpw %.4f with match_bpw %.4f ; control overall %.4f" % (pil["bpw"], spec["match_bpw"], CTRL_BPW), flush=True)
r = H.evaluate("select", docs=list(range(0, 72, 3))); print("layers 1-3 only quantized with column order: KL=%.5f" % r["kl"], flush=True)
spec["match_bpw"] = round(spec["match_bpw"] - (pil["bpw"] - CTRL_BPW) - 0.0005, 4)
print("match_bpw set to", spec["match_bpw"], flush=True)
SQ.run_ldlq_variant("A0_Qperm_ldlq", spec, H.S["mult_Q0ldlq"], method="E8-LDLQ sequential, columns processed by decreasing input second moment (stored permutation), per-matrix rate matching")
