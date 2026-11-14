# Pilot of the asymmetric calibration target (Lumen v3.1 par.4, never executed on the model before):
# first 12 layers quantized (rest BF16), symmetric reference vs asymmetric with two ridges; calibration cal2.
import sys, importlib, json
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
st = H.S["st"]; only = set(range(12)); docs = list(range(72)); out = {}
if "cal2" not in H.S["data"]:
    d = torch.load("/workspace/work/data/cal2/cal2_tokens.pt"); H.S["data"]["cal2"] = d["cal2"]; H.S["meta"]["cal2"] = d["meta"]
H.S["cal_cfg"] = dict(split="cal2", n_docs=128, T=2048, bs=4)
mult = H.S["mult_Q0ldlq"] * 2 ** (2.8212 - 2.8125)
try:
    for tag, spec in (("sym", dict(rule="lumen")), ("asym_0.01", dict(rule="lumen", asym=0.01)), ("asym_0.1", dict(rule="lumen", asym=0.1))):
        H.restore(); mats = SQ.plan_mats()
        info = SQ.ldlq_pass(mats, spec, mult, st, only=only, log_every=3)
        r = H.evaluate("select", docs=docs); ra = V.arki_eval()["dev"]
        out[tag] = dict(bpw=info["bpw"], seconds=info["seconds"], select=r["kl"], arkidev=ra["kl"], arkidev_by_domain={k: v["kl"] for k, v in ra["by_domain"].items()})
        print("[layers 0-11 only] %-10s bpw %.4f pass %.0fs | select KL %.5f | %s" % (tag, info["bpw"], info["seconds"], r["kl"], V.arki_short(ra)), flush=True)
    json.dump(out, open(V.OUT7 + "/pilot_asym_12layers.json", "w"), indent=1)
finally:
    H.S.pop("cal_cfg", None); H.restore()
