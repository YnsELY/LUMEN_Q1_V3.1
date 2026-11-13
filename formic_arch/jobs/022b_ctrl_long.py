# Reload the saved LDLQ control weights (cold reload check) and run the 8192-token evaluation.
import glob, json, time
t = time.time(); H.restore()
d = "/workspace/work/ctrl_ldlq"; mats = {m["name"]: m for m in H.qmats()}
with torch.no_grad():
    for f in sorted(glob.glob(d + "/layer_*.pt")):
        for name, w in torch.load(f).items(): mats[name]["mod"].weight.copy_(w.cuda())
r = H.evaluate("select")
ref = json.load(open(H.OUT + "/variant_A0_Q0ldlq.json"))["select"]["kl"]
print("reloaded control: select KL=%.6f (recorded at creation %.6f) reload %.0fs" % (r["kl"], ref, time.time()-t))
rl = H.evaluate_long(); json.dump(rl, open(H.OUT + "/long_A0_Q0ldlq.json", "w"))
print("A0_Q0ldlq LONG (8192 tok) KL by 1024-bucket:", [round(v, 4) for v in rl["kl_by_bucket"]], "mean %.4f" % rl["kl"])
H.restore()
