# Re-initialisation after the server restart (allocator: expandable segments; teacher caches on CPU).
import importlib, sys, json, time
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_arch/lab")
import formic_q, harness, seq
H = harness; SQ = seq; H.init(model, tok)
t = time.time(); print(H.build_teacher(), "%.0fs" % (time.time()-t))
st = torch.load("/workspace/work/stats_fisher.pt"); H.S["st"] = st
st["MU"] = H.collect_means(); torch.save(st, "/workspace/work/stats_fisher.pt")
print("long teacher:", H.build_teacher_long())
mats = H.qmats(); N = sum(m["mod"].weight.numel() for m in mats)
H.S["N_text"] = N; H.S["target_bits"] = 2.824 * N
H.S["mult_Q0rtn"] = json.load(open(H.OUT + "/variant_A0_Q0rtn.json"))["info"]["mult"]
ld = json.load(open(H.OUT + "/diag_layerdrop_single.json")); H.S["drop_order"] = sorted(range(1, 64), key=lambda i: ld[str(i)]["kl"])
H.S["mult_Q0ldlq"] = H.S["mult_Q0rtn"] * 2 ** (2.8326 - 2.8247)      # rate correction measured in the 3-layer pilot
r = H.evaluate("select", docs=list(range(0, 72, 6)))
print("teacher sanity KL=%.2e agree=%.4f" % (r["kl"], r["agree"]), "mult_Q0rtn", H.S["mult_Q0rtn"], "alloc GiB %.1f" % (torch.cuda.memory_allocated()/2**30))
rl = H.evaluate_long(); print("teacher long sanity KL=%.2e" % rl["kl"])
