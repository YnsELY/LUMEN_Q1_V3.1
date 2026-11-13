# Localise the Ab2 failure: sequential pass on layers 0-3 only (rest BF16), bias in A_b form vs teacher-anchored form.
# Prints, per matrix, |W (mu_T - mu_S)| against |(W-Q) mu_S| and the relative mean shift of the input.
import importlib, harness, seq, json
_S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq
st = H.S["st"]; docs = list(range(0, 72, 3)); only = {0, 1, 2, 3}
orig_finish = H._finish
def spy(m, Q, hist, c, spec, st_, write=True, mu_override=None):
    if spec.get("bias_corr") and m["kind"] != "embed" and m["layer"] in only:
        W = m["mod"].weight.float(); mu_s = mu_override.cuda().float(); mu_t = st_["MU"][m["name"]].cuda().float()
        a = ((W - Q) @ mu_s).norm().item(); b = (W @ (mu_t - mu_s)).norm().item(); o = (W @ mu_t).norm().item()
        print("  L%d %-12s |(W-Q)mu_S|=%.4f |W(mu_T-mu_S)|=%.4f |W mu_T|=%.3f  |mu_T-mu_S|/|mu_T|=%.5f" % (m["layer"], m["kind"], a, b, o, ((mu_t - mu_s).norm() / mu_t.norm()).item()), flush=True)
    return orig_finish(m, Q, hist, c, spec, st_, write, mu_override)
H._finish = spy
try:
    for mode in (True, "teacher"):
        H.restore(); mats = SQ.plan_mats()
        info = SQ.ldlq_pass(mats, dict(rule="lumen", bias_corr=mode), H.S["mult_Abldlq"], st, only=only, log_every=100)
        r = H.evaluate("select", docs=docs)
        print("bias mode %s, layers 0-3 quantized only: KL=%.5f agree=%.4f" % (mode, r["kl"], r["agree"]), flush=True)
finally:
    H._finish = orig_finish
H.restore()
