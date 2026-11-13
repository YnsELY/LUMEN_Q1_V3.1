# A_b x Q0: same E8-LDLQ pass, but every linear gets an FP16 bias b=(W-Q)mu (student mean, sequential) and the
# LDLQ Gram is the centred covariance (the mean direction is carried exactly by the bias).  Resumable.
import importlib, seq
importlib.reload(seq); SQ = seq
rows = sum(m["mod"].weight.shape[0] for m in SQ.plan_mats() if m["kind"] != "embed")
m_b = H.S["mult_Q0ldlq"] * 2 ** (16 * rows / H.S["N_text"])     # bias vectors (16 bits/row) paid from the symbol budget
H.S["mult_Abldlq"] = m_b
SQ.run_ldlq_variant("Ab_Q0ldlq", dict(rule="lumen", bias_corr=True), m_b, method="E8-LDLQ sequential, centred Gram + FP16 bias")
