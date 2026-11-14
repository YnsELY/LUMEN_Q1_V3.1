"""Tables from results/variant_*.json (selection) and, ONLY with --unseal after the freeze commit,
from the sealed confirmation files.  Paired bootstrap over documents (resampling documents, not tokens)."""
import json, glob, os, sys, random, math
OUT = "/workspace/LUMEN_Q1_V3.1/formic_arch/results"; SEALED = "/workspace/work/sealed"

def load(split):
    res = {}
    if split == "select":
        for f in sorted(glob.glob(OUT + "/variant_*.json")):
            d = json.load(open(f)); res[d["id"]] = (d["select"], d.get("info", {}))
    else:
        lock = json.load(open(OUT + "/selection.lock.json"))
        for vid in lock["finalists"]:
            d = json.load(open(f"{SEALED}/{vid}.json")); res[vid] = (d["confirm"], d.get("info", {}))
    return res

def boot(a, b, n=5000, seed=0):
    """95% CI of mean(a-b), documents resampled."""
    rnd = random.Random(seed); d = [x - y for x, y in zip(a, b)]; k = len(d); ms = []
    for _ in range(n):
        ms.append(sum(d[rnd.randrange(k)] for _ in range(k)) / k)
    ms.sort(); return sum(d) / k, ms[int(0.025 * n)], ms[int(0.975 * n)]

def table(split, control_of):
    res = load(split); rows = []
    for vid, (r, info) in res.items():
        dce = r["ce_s"] - r["ce_t"]
        row = dict(id=vid, kl=r["kl"], agree=r["agree"], dce=dce, ppl=r["ppl_s"], bpw=info.get("bpw"), over=info.get("bits_over_target"),
                   by_domain={k: v["kl"] for k, v in r["by_domain"].items()})
        c = control_of(vid)
        if c and c in res and c != vid:
            m, lo, hi = boot(r["per_doc"]["kl"], res[c][0]["per_doc"]["kl"])
            row.update(control=c, dkl=m, dkl_lo=lo, dkl_hi=hi, rel=m / res[c][0]["kl"])
            m2, lo2, hi2 = boot(r["per_doc"]["agree"], res[c][0]["per_doc"]["agree"])
            row.update(dagree=m2, dagree_lo=lo2, dagree_hi=hi2)
        rows.append(row)
    return rows

def fmt(rows):
    doms = sorted(rows[0]["by_domain"])
    out = ["| variante | bpw idéal | octets/cible | KL (nats) | accord top-1 | ΔCE | vs témoin | ΔKL [IC95 %] | " + " | ".join(d[:6] for d in doms) + " |",
           "|---|---:|---:|---:|---:|---:|---|---|" + "---:|" * len(doms)]
    for r in rows:
        d = "" if "dkl" not in r else "%+.4f [%+.4f, %+.4f] (%+.1f %%)" % (r["dkl"], r["dkl_lo"], r["dkl_hi"], 100 * r["rel"])
        out.append("| %s | %s | %s | %.4f | %.4f | %+.4f | %s | %s | %s |" % (
            r["id"], "%.4f" % r["bpw"] if r["bpw"] else "BF16", "%+.3f %%" % (100 * r["over"]) if r["over"] is not None else "—",
            r["kl"], r["agree"], r["dce"], r.get("control", ""), d, " | ".join("%.3f" % r["by_domain"][k] for k in doms)))
    return "\n".join(out)

if __name__ == "__main__":
    split = "confirm" if "--unseal" in sys.argv else "select"
    def control_of(vid):
        if vid.endswith("_bf16") or vid.endswith("nocomp"): return None
        return "A0_Q0ldlq" if "ldlq" in vid else "A0_Q0rtn"
    rows = table(split, control_of)
    json.dump(rows, open(OUT + f"/summary_{split}.json", "w"), indent=1)
    print(fmt(rows))
