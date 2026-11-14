"""Regenerate the numeric tables of the report from the raw result files and substitute the @@MARKERS@@."""
import json, os, sys, importlib.util
B = "/workspace/LUMEN_Q1_V3.1/formic_arch"; OUT = B + "/results"
spec = importlib.util.spec_from_file_location("summ", B + "/lab/summarize.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
ORDER = ["A0_Q0ldlq", "ArotN_Q0ldlq", "Arot_Q0ldlq", "A0_Qperm_ldlq", "Ab_Q0ldlq", "A0_Qldlcol_ldlq", "Ab2_Q0ldlq",
         "A0_Q0rtn", "Ab_Q0rtn", "Ab_Q0rtn_mu4dom", "A0_Qmix_rtn", "Ab_Qmix_rtn", "A0_Qalloc_rtn", "A0_Qcol_rtn", "A0_Qfg_rtn", "A0_Qfl_rtn",
         "A2_drop2_Q0rtn", "A1_w0.125_Q0rtn", "A1_w0.25_Q0rtn",
         "Arot_bf16", "A2_drop2_bf16", "A2_drop4_bf16", "A2_drop8_bf16", "A1_w0.125_bf16", "A1_w0.125_bf16_nocomp", "A1_w0.25_bf16"]
LABEL = {"A0_Q0ldlq": "**témoin LDLQ** : architecture d'origine, E8-LDLQ", "ArotN_Q0ldlq": "rotation du résiduel, native, E8-LDLQ", "Arot_Q0ldlq": "rotation du résiduel, émulée, E8-LDLQ",
         "Ab_Q0ldlq": "biais FP16 + Gram centré, E8-LDLQ", "A0_Qperm_ldlq": "ordre des colonnes (second moment décroissant), E8-LDLQ", "A0_Qldlcol_ldlq": "échelles de colonne (pivots LDL), E8-LDLQ", "Ab2_Q0ldlq": "biais ancré professeur, E8-LDLQ",
         "A0_Q0rtn": "**témoin sans rétroaction** : E8 au plus proche", "Ab_Q0rtn": "biais FP16, sans rétroaction", "Ab_Q0rtn_mu4dom": "biais FP16, moyennes sans code ni chat",
         "A0_Qmix_rtn": "échelles de colonne sur entrées des mélangeurs", "Ab_Qmix_rtn": "biais + échelles de colonne (mélangeurs)", "A0_Qalloc_rtn": "bits répartis par famille",
         "A0_Qcol_rtn": "échelles de colonne partout", "A0_Qfg_rtn": "pas Fisher global", "A0_Qfl_rtn": "pas Fisher local",
         "A2_drop2_Q0rtn": "62 couches, octets réinvestis (2,907 bpw)", "A1_w0.125_Q0rtn": "MLP −12,5 %, octets réinvestis (3,068 bpw)", "A1_w0.25_Q0rtn": "MLP −25 %, octets réinvestis (3,358 bpw)",
         "Arot_bf16": "rotation seule, BF16 (résidu d'équivalence)", "A2_drop2_bf16": "62 couches, BF16", "A2_drop4_bf16": "60 couches, BF16", "A2_drop8_bf16": "56 couches, BF16",
         "A1_w0.125_bf16": "MLP −12,5 % + moindres carrés, BF16", "A1_w0.125_bf16_nocomp": "MLP −12,5 % sans réajustement, BF16", "A1_w0.25_bf16": "MLP −25 % + moindres carrés, BF16"}
def control_of(vid):
    if vid.endswith("_bf16") or vid.endswith("nocomp") or vid in ("A0_Q0ldlq", "A0_Q0rtn"): return None
    return "A0_Q0ldlq" if "ldlq" in vid else "A0_Q0rtn"
def f(x, n=4): return (("%." + str(n) + "f") % x).replace(".", ",")
def fr(s): return s.replace(".", ",")
def table(split):
    rows = {r["id"]: r for r in S.table(split, control_of)}
    doms = ["chat", "code", "en_prose", "math_reasoning", "multiling", "science_tech"]
    out = ["| Variante | Identifiant | Octets / cible | KL | Accord top-1 | ΔKL vs témoin [IC 95 %] | chat | code | prose | math | multil. | science |", "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for vid in ORDER:
        if vid not in rows: continue
        r = rows[vid]
        d = "" if "dkl" not in r else fr("%+.4f [%+.4f ; %+.4f] (%+.1f %%)" % (r["dkl"], r["dkl_lo"], r["dkl_hi"], 100 * r["rel"]))
        out.append("| %s | `%s` | %s | %s | %s %% | %s | %s |" % (LABEL.get(vid, vid), vid, "—" if r["over"] is None else fr("%+.2f %%" % (100 * r["over"])), f(r["kl"]), f(100 * r["agree"], 1), d,
                   " | ".join(f(r["by_domain"][k], 3) for k in doms)))
    return "\n".join(out)
def long_table():
    out = ["| Variante | KL moyenne | par tranche de 1024 tokens (0–8192) | Accord top-1 |", "|---|---:|---|---:|"]
    for vid in ("A0_Q0ldlq", "ArotN_Q0ldlq", "Arot_Q0ldlq", "Ab_Q0ldlq", "A0_Qldlcol_ldlq", "A0_Q0ldlq_big", "ArotN_Q0ldlq_big", "A0_Qperm_ldlq"):
        p = OUT + f"/long_{vid}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p)); out.append("| `%s` | %s | %s | %s %% |" % (vid, f(d["kl"]), " · ".join(f(x, 3) for x in d["kl_by_bucket"]), f(100 * sum(d["agree_by_bucket"]) / len(d["agree_by_bucket"]), 1)))
    return "\n".join(out)
def big_table():
    res = {}
    for vid in ("A0_Q0ldlq_big", "ArotN_Q0ldlq_big", "A0_Qperm_ldlq"):
        p = OUT + f"/big_{vid}.json"
        if os.path.exists(p): res[vid] = json.load(open(p))
    if "A0_Q0ldlq_big" not in res: return None
    c = res["A0_Q0ldlq_big"]["big"]; doms = sorted(set(c["domains"]))
    out = ["| Variante | Octets / cible | KL | Accord top-1 | ΔKL vs témoin [IC 95 %] | docs améliorés | " + " | ".join(d[:6] for d in doms) + " |", "|---|---:|---:|---:|---|---:|" + "---:|" * len(doms)]
    for vid, r in res.items():
        b = r["big"]; d = ""; better = ""
        if vid != "A0_Q0ldlq_big":
            m, lo, hi = S.boot(b["per_doc"]["kl"], c["per_doc"]["kl"]); d = fr("%+.4f [%+.4f ; %+.4f] (%+.1f %%)" % (m, lo, hi, 100 * m / c["kl"]))
            better = "%d / %d" % (sum(x < y for x, y in zip(b["per_doc"]["kl"], c["per_doc"]["kl"])), len(c["per_doc"]["kl"]))
        out.append("| `%s` | %s | %s | %s %% | %s | %s | %s |" % (vid, fr("%+.2f %%" % (100 * r["info"]["bits_over_target"])), f(b["kl"]), f(100 * b["agree"], 1), d, better, " | ".join(f(b["by_domain"][k]["kl"], 3) for k in doms)))
    return "\n".join(out)
if __name__ == "__main__":
    print(table("select")); print(); print(table("confirm")); print(); print(long_table()); print(); print(big_table())
