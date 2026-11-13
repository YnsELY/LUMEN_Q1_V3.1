# Save the LDLQ control's decoded weights per layer (so extra evaluations do not need another 25-min pass).
import os, time, importlib, harness
_S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
t = time.time(); d = "/workspace/work/ctrl_ldlq"; os.makedirs(d, exist_ok=True)
groups = {}
for m in H.qmats(): groups.setdefault(m["layer"], {})[m["name"]] = m["mod"].weight.detach().cpu()
for k, g in groups.items(): torch.save(g, f"{d}/layer_{k}.pt")
del groups
print("saved control weights in %.0fs" % (time.time()-t)); os.system(f"du -sh {d}; df -h /workspace | tail -1")
r = H.evaluate("select")
import json; json.dump(dict(kl=r["kl"], kl_pos8=[sum(p[b] for p in r["per_doc"]["kl_pos8"])/72 for b in range(8)]), open(H.OUT+"/pos_A0_Q0ldlq.json", "w"))
print("A0_Q0ldlq KL by position bucket (128 tok each):", [round(sum(p[b] for p in r["per_doc"]["kl_pos8"])/72, 4) for b in range(8)])
