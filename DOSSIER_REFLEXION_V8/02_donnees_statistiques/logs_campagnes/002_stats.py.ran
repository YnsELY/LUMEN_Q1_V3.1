import time
t=time.time()
st = H.collect_stats(n_docs=144, T=512)
torch.save(st, "/workspace/work/stats_fisher.pt")
H.S["st"] = st
print("stats done", time.time()-t, "ntok", st["ntok"], len(st["X"]), len(st["G"]), len(st["TS"]), torch.cuda.max_memory_allocated()/2**30)
import math
for n in ["layers.0.mlp.gate_proj","layers.0.mlp.down_proj","layers.30.mlp.gate_proj","layers.30.mlp.up_proj","layers.30.mlp.down_proj","layers.30.linear_attn.in_proj_qkv","layers.30.linear_attn.out_proj","layers.31.self_attn.q_proj","layers.31.self_attn.o_proj","layers.63.mlp.down_proj","lm_head"]:
    X=st["X"][n]; G=st["G"][n]
    q=lambda v: [float(torch.quantile(v[:200000].float(), p)) for p in (0.01,0.25,0.5,0.75,0.99)]
    print(n, "X q:", ["%.2e"%a for a in q(X)], "max %.2e"%X.max(), "| G q:", ["%.2e"%a for a in q(G)], "max %.2e"%G.max())
