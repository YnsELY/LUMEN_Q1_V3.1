# Is the Ab2 catastrophe (KL 3.0) only my lm_head teacher-mean bug (mean taken over the first 64 positions)?
# Reload the Ab2 checkpoint, replace ONLY the lm_head bias by the A_b form (W-Q) mu_student, re-evaluate.
import importlib, seq, json
importlib.reload(seq); SQ = seq
H.restore(); mats = SQ.plan_mats(); by = {}
for m in mats: by.setdefault(m["layer"], {})[m["kind"]] = m
sq = SQ.Seq(); stt = SQ._ckpt_load("/workspace/work/ckpt/Ab2_Q0ldlq", by, sq); del sq
docs = list(range(0, 72, 3))
r = H.evaluate("select", docs=docs); print("Ab2 reloaded from checkpoint (24 docs): KL=%.4f agree=%.4f" % (r["kl"], r["agree"]))
lm = H.S["lm"]; ids = H.S["data"]["cal"]; msum = 0
with torch.no_grad():
    for i in range(0, ids.shape[0], 8):
        h = lm(input_ids=ids[i:i+8].cuda()).last_hidden_state
        msum = msum + h.float().sum(dim=(0, 1)).double()
    mu_s = (msum / ids.numel()).float()
    Wt = H.S["T_head"]; Q = model.lm_head.weight
    b = torch.zeros(Q.shape[0], device="cuda")
    for i in range(0, Q.shape[0], 32768):
        b[i:i+32768] = (Wt[i:i+32768].float() - Q[i:i+32768].float()) @ mu_s
    old = model.lm_head.bias.detach().float()
    print("lm_head bias: old |b| rms %.4f, new |b| rms %.4f" % (old.pow(2).mean().sqrt(), b.pow(2).mean().sqrt()))
    model.lm_head.bias = torch.nn.Parameter(b.to(torch.float16).to(Q.dtype), requires_grad=False)
info = json.load(open(H.OUT + "/variant_Ab2_Q0ldlq.json"))["info"]
info["note"] = "Ab2 with lm_head bias in A_b form (student mean); inner linears keep teacher-anchored bias"
H.run_variant("Ab2h_Q0ldlq", info)
rl = H.evaluate_long(); json.dump(rl, open(H.OUT + "/long_Ab2h_Q0ldlq.json", "w"))
print("Ab2h_Q0ldlq LONG KL by 1024-bucket:", [round(v, 4) for v in rl["kl_by_bucket"]], "mean %.4f" % rl["kl"])
H.restore()
