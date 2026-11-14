# Cheap screen (no feedback, whole text path, global budget 2.808 bpw): clipped Fisher re-weighting of row steps.
import sys, importlib, json
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
st = H.S["st"]; mats = H.qmats(); target = 2.808 * H.S["N_text"]; docs = list(range(0, 72, 2)); out = {}
body = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
def run(tag, spec):
    H.restore(); mult = H.calibrate_mult(mats, spec, target, st, lo=0.25, hi=2.0, iters=10)
    info = H.quantize_all(mats, spec, mult, st); r = H.evaluate("select", docs=docs); ra = V.arki_eval()["dev"]
    out[tag] = dict(spec=spec, bpw=info["bpw"], select36=r["kl"], arkidev=ra["kl"], arkidev_by_domain={k: v["kl"] for k, v in ra["by_domain"].items()})
    print("%-28s bpw %.4f | select(36 docs) KL %.4f | %s" % (tag, info["bpw"], r["kl"], V.arki_short(ra)), flush=True)
run("rtn_lumen", dict(rule="lumen"))
run("rtn_row0.5_body", dict(rule="lumen", row_clip=0.5, row_kinds=body))
run("rtn_row1.0_body", dict(rule="lumen", row_clip=1.0, row_kinds=body))
run("rtn_row1.0_writers", dict(rule="lumen", row_clip=1.0, row_kinds=("out_proj", "o_proj", "down_proj")))
run("rtn_row1.0_head", dict(rule="lumen", row_clip=1.0, row_kinds=("lm_head",)))
run("rtn_row2.0_head", dict(rule="lumen", row_clip=2.0, row_kinds=("lm_head",)))
json.dump(out, open(V.OUT7 + "/screen_rowfisher_rtn.json", "w"), indent=1); H.restore()
