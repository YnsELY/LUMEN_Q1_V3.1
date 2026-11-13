import json, time, importlib, seq
importlib.reload(seq); SQ = seq
st = H.S["st"]; target = H.S["target_bits"]
ld = json.load(open(H.OUT+"/diag_layerdrop_single.json"))
order = sorted(range(1, 64), key=lambda i: ld[str(i)]["kl"])
print("layers by increasing single-drop KL:", order[:16])
H.S["drop_order"] = order
for k in (2, 4, 8):
    drop = order[:k]
    H.restore(); SQ.drop_layers(drop)
    H.run_variant(f"A2_drop{k}_bf16", dict(drop=drop, note="BF16, layers removed, no recovery"))
