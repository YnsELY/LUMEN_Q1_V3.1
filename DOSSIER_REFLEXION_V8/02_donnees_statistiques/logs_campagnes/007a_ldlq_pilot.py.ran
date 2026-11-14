# Timing + rate pilot of the sequential E8-LDLQ pass on 3 layers (BF16 prefix), control recipe.
import time, importlib, seq
importlib.reload(seq); SQ = seq
st = H.S["st"]; m0 = H.S["mult_Q0rtn"]
H.restore(); mats = SQ.plan_mats()
only = {1, 2, 3}
t = time.time()
info = SQ.ldlq_pass(mats, dict(rule="lumen"), m0, st, only=only, log_every=1)
print("pilot lumen LDLQ: bpw %.4f" % info["bpw"], {k: round(v, 4) for k, v in info["bykind"].items()}, "%.1fs" % (time.time()-t), "peak mem %.1f GiB" % (torch.cuda.max_memory_allocated()/2**30))
sub = [m for m in mats if m["layer"] in only]
H.restore()
b = sum(H.quantize_matrix(m, dict(rule="lumen"), m0, st)[0] for m in sub); n = sum(m["mod"].weight.numel() for m in sub)
print("same layers RTN bpw %.4f" % (b/n))
H.restore()
