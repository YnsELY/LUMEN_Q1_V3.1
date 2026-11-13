# Where does the damage come from?  Quantize ONE family of matrices at a time (rest BF16), 3 scale rules,
# no-feedback E8 at the control multiplier (same sigma/step for every rule via each rule's own calibrated mult).
import json, time
st = H.S["st"]; mats = H.qmats(); docs = list(range(0, 72, 3))
groups = {"mlp_in(gate,up)": ("gate_proj", "up_proj"), "mlp_down": ("down_proj",), "gdn_in(qkv,z,a,b)": ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
          "gdn_out": ("out_proj",), "attn_in(q,k,v)": ("q_proj", "k_proj", "v_proj"), "attn_out": ("o_proj",), "embed": ("embed",), "lm_head": ("lm_head",)}
mults = {"lumen": 0.5917, "col": 0.5722, "fisher_local": 0.5066}
pname = lambda m: "lm_head.weight" if m["kind"] == "lm_head" else "model.language_model." + m["name"] + ".weight"
H.restore(); res = {}
for g, kinds in groups.items():
    sub = [m for m in mats if m["kind"] in kinds]; names = [pname(m) for m in sub]
    for rule, mult in mults.items():
        if g == "embed" and rule != "lumen": continue
        if g == "lm_head" and rule == "fisher_local": continue
        b = 0; n = 0
        for m in sub:
            bb, nn = H.quantize_matrix(m, dict(rule=rule), mult, st); b += bb; n += nn
        r = H.evaluate("select", docs=docs)
        res[f"{g}|{rule}"] = dict(kl=r["kl"], agree=r["agree"], bpw=b/n, params=n)
        print("%-20s %-13s bpw=%.3f params=%.2fB KL=%.5f agree=%.4f" % (g, rule, b/n, n/1e9, r["kl"], r["agree"]), flush=True)
        H.restore(names)
json.dump(res, open(H.OUT+"/diag_bykind_rules.json", "w"))
