# Final step of V7: confirmation of the selected recipe FROM ITS FILES, competitors on the fresh split, and the
# complete package (text + Vision + MTP + tokenizer/config files) with its real size on disk.
import sys, importlib, json, time, os, subprocess
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
SEL = open("/workspace/work/SELECTED").read().strip(); pkg = f"{V.PKG_ROOT}/{SEL}"; out = dict(selected=SEL)
VIS_BPW, MTP_BPW = 5.0, 4.0
# 1) teachers on BF16: fresh confirmation split conf2 and the image prompts
H.restore(); t = time.time()
d = torch.load("/workspace/work/data/conf2/conf2_tokens.pt"); H.S["data"]["conf2"] = d["conf2"]; H.S["meta"]["conf2"] = d["meta"]
with torch.no_grad():
    H.S["teach"]["conf2"] = torch.cat([H.S["lm"](input_ids=d["conf2"][i:i + 8].cuda()).last_hidden_state.cpu() for i in range(0, d["conf2"].shape[0], 8)])
print("conf2 teacher", tuple(H.S["teach"]["conf2"].shape), "| vision teacher prompts:", V.vision_teacher(), "%.0fs" % (time.time() - t), flush=True)
# 2) selected recipe rebuilt from its files -> conf2 ; then Vision tower + MTP + aux files added to the package
n = V.load_package(pkg)
r = H.evaluate("conf2"); out["conf2_selected"] = r; print("[%s from files, %d matrices] conf2 %s" % (SEL, n, H.short(r)), flush=True)
rv0 = V.vision_eval(); print("vision prompts, text path from package + Vision tower BF16:", {k: round(v["kl"], 4) for k, v in rv0["by_kind"].items()}, "mean %.4f" % rv0["kl"], flush=True)
formic_q.set_limit(16384)            # rates above ~3.6 bpw: ideal cost without the reference's 128-shell cap (the real codec has none)
vis = V.package_vision(pkg, VIS_BPW); rv1 = V.vision_eval()
print("vision prompts, text path + Vision tower from package (%.3f bpw real on %.1f M weights):" % (vis["bpw"], vis["e8_params"] / 1e6), {k: round(v["kl"], 4) for k, v in rv1["by_kind"].items()}, "mean %.4f top1 %.2f%%" % (rv1["kl"], 100 * rv1["agree"]), flush=True)
mtp = V.package_mtp(pkg, MTP_BPW); aux = V.package_aux_files(pkg); formic_q.set_limit(128)
ver = subprocess.run(["python3", "-I", V.CODEC + "/package.py", "verify", pkg], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
du = sum(os.path.getsize(os.path.join(pkg, f)) for f in os.listdir(pkg))
out.update(vision=dict(vis, eval_text_only_quantized=rv0, eval_full_package=rv1), mtp=mtp, aux_bytes=aux, package_total_bytes=du, verify_tail=ver.stdout[-400:], verify_rc=ver.returncode)
print("MTP block: %.3f bpw real on %.1f M weights (NOT evaluated: not loadable in this runtime) | aux files %d bytes" % (mtp["bpw"], mtp["e8_params"] / 1e6, aux), flush=True)
print("COMPLETE PACKAGE ON DISK: %d bytes = %.6f GB (limit 10 000 000 000) | verify rc=%d: %s" % (du, du / 1e9, ver.returncode, ver.stdout.strip().splitlines()[-1] if ver.stdout.strip() else ver.stderr[-200:]), flush=True)
json.dump(out, open(V.OUT7 + "/final_selected.json", "w"))
# 3) competitors on the fresh split (same graph, same teacher)
for cid, path in (("Arki05_2.82bpw", "/workspace/work/competitors/arki05/Qwen3.8-27B-2.82bpw.gguf"), ("XQD_Q2_nomtp", "/workspace/work/competitors/axolotl/Qwen3.8-27B-XQD-Q2-nomtp.gguf")):
    V.load_gguf(path); r = H.evaluate("conf2"); out["conf2_" + cid] = r
    print("[%s] conf2 %s" % (cid, H.short(r)), flush=True)
    json.dump(out, open(V.OUT7 + "/final_selected.json", "w"))
H.restore()
