# Second competitor in OUR graph and on both protocols: Axolotl XQD Q2 without MTP (9.469 GB, rev d7110adadd).
import sys, importlib, json
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
rep = V.load_gguf("/workspace/work/competitors/axolotl/Qwen3.8-27B-XQD-Q2-nomtp.gguf")
print("loaded %d tensors in %.0fs: %.3f bpw on %.2f G weights (%.3f GB of the %.3f GB file); kept BF16: %d; skipped: %s" % (
    rep["loaded"], rep["seconds"], rep["bpw_loaded"], rep["loaded_params"] / 1e9, rep["loaded_bytes"] / 1e9, rep["file_bytes"] / 1e9, len(rep["kept_bf16"]), list(rep["skipped"].items())[:5]), flush=True)
cs = sorted(v["cos"] for v in rep["tensors"].values()); print("cosine with BF16: min %.4f median %.4f" % (cs[0], cs[len(cs) // 2]), flush=True)
byk = {}
for k, v in rep["tensors"].items():
    kk = k.split(".")[2] if k.startswith("blk.") else k; byk.setdefault(kk, [0, 0]); byk[kk][0] += v["bytes"] * 8; byk[kk][1] += v["params"]
print("bits/weight by family:", {k: round(b / n, 3) for k, (b, n) in byk.items()}, flush=True)
res = dict(id="XQD_Q2_nomtp", load=rep)
for split in ("select", "big"):
    r = H.evaluate(split); res[split] = r; print("[XQD_Q2_nomtp] %-7s %s" % (split, H.short(r)), flush=True)
ra = V.arki_eval(); res["arki108"] = ra; print("[XQD_Q2_nomtp]", V.arki_short(ra), "| dev half:", V.arki_short(ra["dev"]), flush=True)
rl = H.evaluate_long(); res["long"] = rl; print("[XQD_Q2_nomtp] LONG mean %.4f" % rl["kl"], flush=True)
json.dump(res, open(V.OUT7 + "/competitor_XQD_Q2_nomtp.json", "w")); H.restore()
