"""V7 tables from raw result files. Development data only (select-72, big-600, long, P-A108-dev half);
the P-A108 confirmation half is read only with --unseal, after results/v7_selection.lock.json is committed."""
import json, os, sys, glob, random
R = "/workspace/LUMEN_Q1_V3.1/formic_v7/results"; SEALED = "/workspace/work/sealed"

def split_part(perw):
    """dev/conf halves: alternating rows inside each domain (same rule as lab/v7.py)."""
    seen = {}; out = {}
    for w in sorted(perw, key=lambda w: w["row"]):
        d = w["domain"]; out[w["row"]] = "dev" if seen.get(d, 0) % 2 == 0 else "conf"; seen[d] = seen.get(d, 0) + 1
    return out

def arki(vid, part="dev"):
    """per-window list for a variant on the requested half; None if not available."""
    if os.path.exists(R + f"/competitor_{vid}.json"):
        pw = json.load(open(R + f"/competitor_{vid}.json"))["arki108"]["per_window"]
    elif os.path.exists(R + f"/arki_{vid}.json"):            # early variants: full 108 windows were stored in the repo
        pw = json.load(open(R + f"/arki_{vid}.json"))["per_window"]
    elif part == "dev" and os.path.exists(R + f"/arkidev_{vid}.json"):
        return json.load(open(R + f"/arkidev_{vid}.json"))["per_window"]
    elif part != "dev" and os.path.exists(SEALED + f"/arki_{vid}.json"):
        d = json.load(open(SEALED + f"/arki_{vid}.json")); return d["conf"]["per_window"] if part == "conf" else d["full"]["per_window"]
    else: return None
    if part == "all": return pw
    sp = split_part(pw); return [w for w in pw if sp[w["row"]] == part]

def tokmean(pw, key="kl"): return sum(w[key] * w["n"] for w in pw) / sum(w["n"] for w in pw)

def ratio_ci(a, b, n=4000, seed=0):
    """token-mean KL ratio a/b - 1 with a window bootstrap (same windows resampled for both)."""
    A = {w["row"]: w for w in a}; B = {w["row"]: w for w in b}; rows = [r for r in A if r in B and A[r]["n"] > 0]
    rnd = random.Random(seed); bs = []
    for _ in range(n):
        rr = [rows[rnd.randrange(len(rows))] for _ in rows]
        bs.append(sum(A[r]["kl"] * A[r]["n"] for r in rr) / sum(B[r]["kl"] * B[r]["n"] for r in rr) - 1)
    bs.sort(); pa = [A[r] for r in rows]; pb = [B[r] for r in rows]
    return tokmean(pa) / tokmean(pb) - 1, bs[int(.025 * n)], bs[int(.975 * n)], sum(A[r]["kl"] < B[r]["kl"] for r in rows), len(rows)

def big(vid):
    if os.path.exists(R + f"/competitor_{vid}.json"): return json.load(open(R + f"/competitor_{vid}.json"))["big"]
    p = R + f"/big_{vid}.json"; return json.load(open(p))["big"] if os.path.exists(p) else None

def doc_ci(a, b, n=4000, seed=0):
    rnd = random.Random(seed); k = len(a); bs = []
    for _ in range(n):
        idx = [rnd.randrange(k) for _ in range(k)]; bs.append(sum(a[i] for i in idx) / sum(b[i] for i in idx) - 1)
    bs.sort(); return sum(a) / sum(b) - 1, bs[int(.025 * n)], bs[int(.975 * n)]

def variants():
    v = ["Arki05_2.82bpw", "XQD_Q2_nomtp"]
    for f in sorted(glob.glob(R + "/variant_*.json"), key=os.path.getmtime): v.append(os.path.basename(f)[8:-5])
    return v

if __name__ == "__main__":
    part = "conf" if "--unseal" in sys.argv else "dev"; ref = "Arki05_2.82bpw"; base = "L0_uniform_2.808"
    doms = ["code", "encyclopedic", "instruction", "long", "math", "multilingual", "tools", "web"]
    print("| variant | bpw | P-A108-%s KL | top-1 | vs Arki05 [95%% CI] | vs L0 [95%% CI] | big-600 KL | vs L0 big [95%% CI] | long | %s |" % (part, " | ".join(d[:5] for d in doms)))
    print("|---|---:|---:|---:|---|---|---:|---|---:|" + "---:|" * len(doms))
    for vid in variants():
        pw = arki(vid, part); bg = big(vid)
        if pw is None: continue
        comp = os.path.exists(R + f"/competitor_{vid}.json")
        info = dict(bpw=json.load(open(R + f"/competitor_{vid}.json"))["load"]["bpw_loaded"]) if comp else json.load(open(R + f"/variant_{vid}.json"))["info"]
        c1 = ratio_ci(pw, arki(ref, part)) if vid != ref else None; c2 = ratio_ci(pw, arki(base, part)) if vid != base and not comp and arki(base, part) else None
        cb = doc_ci(bg["per_doc"]["kl"], big(base)["per_doc"]["kl"]) if bg and vid != base and big(base) else None
        lp = R + f"/long_{vid}.json"; lg = json.load(open(lp))["kl"] if os.path.exists(lp) else (json.load(open(R + f"/competitor_{vid}.json"))["long"]["kl"] if comp else float("nan"))
        bd = {}
        for d in doms:
            ws = [w for w in pw if w["domain"] == d]; bd[d] = tokmean(ws) if ws else float("nan")
        f = lambda c: "" if c is None else "%+.1f%% [%+.1f, %+.1f]" % (100 * c[0], 100 * c[1], 100 * c[2])
        print("| %s | %.3f | %.4f | %.2f%% | %s | %s | %s | %s | %.3f | %s |" % (vid, info["bpw"], tokmean(pw), 100 * tokmean(pw, "agree"), f(c1), f(c2),
              "%.4f" % bg["kl"] if bg else "", f(cb), lg, " | ".join("%.3f" % bd[d] for d in doms)))
