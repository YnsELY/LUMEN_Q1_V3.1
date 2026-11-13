# Candidate F1: uniform E8-LDLQ at Arki05 bytes, calibrated on BOTH sets (cal 144x1024 + cal2 128x2048 = 409 600 tokens),
# written as a REAL package by the fast codec during the pass, then rebuilt from the files and re-evaluated.
import sys, importlib
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
import gc
for k in ("info", "r", "ra", "rl", "rep", "res", "mats", "out", "spec", "d", "pil", "cs", "byk"):      # leftovers of earlier jobs
    if k in globals(): del globals()[k]
gc.collect(); torch.cuda.empty_cache()
big_t = sorted(((o.numel() * o.element_size() / 2**30, tuple(o.shape), str(o.dtype)) for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda and o.numel() * o.element_size() > 2**29), reverse=True)
print("GPU allocated at job start: %.1f GiB ; tensors > 0.5 GiB outside the model:" % (torch.cuda.memory_allocated() / 2**30), [t for t in big_t if t[1] not in ((248320, 5120),)][:8], flush=True)
if "cal2" not in H.S["data"]:
    d = torch.load("/workspace/work/data/cal2/cal2_tokens.pt"); H.S["data"]["cal2"] = d["cal2"]; H.S["meta"]["cal2"] = d["meta"]
H.S["cal_cfg"] = dict(groups=[dict(split="cal", n_docs=144, T=1024, bs=8), dict(split="cal2", n_docs=128, T=2048, bs=4)])
try:
    import os
    if not os.path.exists("/workspace/work/ckpt/F1_cal12_uniform/state.pt"):
        H.restore(); print("multi-group propagation parity (BF16):", SQ.seq_parity_groups(), flush=True)
    else:
        print("resuming F1 from its checkpoint (parity already verified bit-identical in the first attempt)", flush=True)
    # real codec: file bytes = ideal cost (measured -0.005 %), so the ideal target equals Arki05's physical 2.813 bpw
    mult = H.S["mult_Q0ldlq"] * 2 ** (2.8212 - 2.8125)
    V.run_exported("F1_cal12_uniform", dict(rule="lumen"), mult, method="uniform E8-LDLQ, calibration cal+cal2 (409.6k tokens), real package")
finally:
    H.S.pop("cal_cfg", None)
