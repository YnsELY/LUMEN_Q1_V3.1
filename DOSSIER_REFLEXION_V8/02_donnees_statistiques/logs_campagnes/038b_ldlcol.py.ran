# A0 x Q_ldlcol: exact diagonal change of each linear's INPUT basis, designed for LDLQ: c_j = sqrt(LDL pivot_j)
# (flattens the pivots; FP16 column scales stored and counted: 16 bits per column). Same E8-LDLQ pass otherwise.
# Every matrix is rate-matched individually to the control's bits per weight (same bytes matrix by matrix).
import importlib, harness, seq, json
_S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
import formic_q; importlib.reload(formic_q); importlib.reload(seq); SQ = seq
st = H.S["st"]; CTRL_BPW = json.load(open(H.OUT + "/variant_A0_Q0ldlq.json"))["info"]["bpw"]
spec = dict(rule="ldlcol", match_bpw=2.811, col_clip=4.0)
H.restore(); mats = SQ.plan_mats()
pil = SQ.ldlq_pass(mats, spec, H.S["mult_Q0ldlq"], st, only={1, 2, 3}, log_every=1)
print("pilot (layers 1-3) bpw %.4f with match_bpw %.4f ; control overall %.4f" % (pil["bpw"], spec["match_bpw"], CTRL_BPW), {k: round(v, 4) for k, v in pil["bykind"].items()}, flush=True)
spec["match_bpw"] = round(spec["match_bpw"] - (pil["bpw"] - CTRL_BPW) - 0.0005, 4)       # land at (just under) the control's bytes
print("match_bpw set to", spec["match_bpw"], flush=True)
SQ.run_ldlq_variant("A0_Qldlcol_ldlq", spec, H.S["mult_Q0ldlq"], method="E8-LDLQ sequential, LDL-pivot column scaling (clip 4x), per-matrix rate matching")
