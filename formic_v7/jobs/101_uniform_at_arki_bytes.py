# V7 control: unchanged Lumen recipe (uniform E8-LDLQ) at the SAME BYTES as Arki05 2.82 bpw for the same 498 matrices:
# 9.456 GB = 2.813 bpw physical -> 2.808 bpw ideal after the measured codec overhead (0.005 bpw).
import sys, importlib
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_v7/lab")
import harness, seq, formic_q, v7
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq; importlib.reload(v7); V = v7
mult = H.S["mult_Q0ldlq"] * 2 ** (2.8212 - 2.808)
V.run("L0_uniform_2.808", dict(rule="lumen"), mult, method="Lumen v3.1 port, uniform E8-LDLQ, at Arki05-2.82bpw bytes")
