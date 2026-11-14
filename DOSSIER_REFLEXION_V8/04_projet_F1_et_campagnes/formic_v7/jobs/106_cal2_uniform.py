# Calibration data varied, everything else fixed: uniform E8-LDLQ at 2.808 bpw, calibrated on cal2
# (128 x 2048 tokens: prose, web, 8 languages, code, math, science + chat, tool calls and reasoning traces rendered with
# the chat template; half of the raw documents start with <|endoftext|>; no document shared with any evaluation set).
import sys, importlib
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
d = torch.load("/workspace/work/data/cal2/cal2_tokens.pt"); H.S["data"]["cal2"] = d["cal2"]; H.S["meta"]["cal2"] = d["meta"]
H.S["cal_cfg"] = dict(split="cal2", n_docs=128, T=2048, bs=4)
try:
    mult = H.S["mult_Q0ldlq"] * 2 ** (2.8212 - 2.808)
    V.run("L2_cal2_uniform", dict(rule="lumen"), mult, method="uniform E8-LDLQ at Arki05 bytes, calibration set cal2 (128x2048, incl. chat/tools/reasoning)")
finally:
    H.S.pop("cal_cfg", None)
