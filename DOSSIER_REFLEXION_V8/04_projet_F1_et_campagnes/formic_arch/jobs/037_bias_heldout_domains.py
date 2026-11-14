# Robustness of A_b to a domain shift: bias vectors computed from calibration documents of 4 domains only
# (en_prose, multiling, math_reasoning, science_tech); chat and code never seen by the mean statistic.
import importlib, harness, time, json
_S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
import seq; importlib.reload(seq); SQ = seq
st = H.S["st"]; mats = H.qmats(); target = H.S["target_bits"]
H.restore()
keep_dom = ("en_prose", "multiling", "math_reasoning", "science_tech")
docs = [i for i, m in enumerate(H.S["meta"]["cal"]) if m["domain"] in keep_dom]
mu_full = st["MU"]; st["MU"] = H.collect_means(docs=docs)
cos = {k: float(torch.nn.functional.cosine_similarity(st["MU"][k], mu_full[k], dim=0)) for k in mu_full}
print("docs used for mu:", len(docs), "| cosine(mu_4dom, mu_full): min %.5f mean %.5f" % (min(cos.values()), sum(cos.values())/len(cos)))
spec = dict(rule="lumen", bias_corr=True)
mult = H.calibrate_mult(mats, spec, target, st, lo=0.25, hi=2.0, iters=11)
info = H.quantize_all(mats, spec, mult, st)
info.update(spec=spec, mult=mult, target_bits=target, bits_over_target=info["bits"]/target-1, mu_domains=keep_dom, note="bias means from 4 domains; chat+code held out of the mean statistic")
H.run_variant("Ab_Q0rtn_mu4dom", info)
st["MU"] = mu_full; H.restore()
