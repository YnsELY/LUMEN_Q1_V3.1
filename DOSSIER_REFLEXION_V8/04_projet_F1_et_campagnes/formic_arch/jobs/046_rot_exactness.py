# Localise the residual of the "exact" basis change in BF16 arithmetic (no quantization): per layer, per operation.
import json
lm = H.S["lm"]; docs = list(range(0, 72, 6)); R = SQ._rot_matrix(5120).float(); out = {}
def readers(L):
    a = ([L.linear_attn.in_proj_qkv, L.linear_attn.in_proj_z, L.linear_attn.in_proj_a, L.linear_attn.in_proj_b] if hasattr(L, "linear_attn") else [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj])
    return ((L.input_layernorm, a), (L.post_attention_layernorm, [L.mlp.gate_proj, L.mlp.up_proj]))
def writers(L): return ((L.linear_attn.out_proj if hasattr(L, "linear_attn") else L.self_attn.o_proj), L.mlp.down_proj)
@torch.no_grad()
def apply(i, mode, which=(0, 1)):
    L = lm.layers[i]
    for j, (norm, mods) in enumerate(readers(L)):
        if j not in which: continue
        g = 1 + norm.weight.float()
        if mode in ("fold", "fold_rot", "fold32"):
            for mod in mods:
                Wf = mod.weight.float() * g
                if mode == "fold_rot": Wf = (Wf @ R.T) @ R
                mod.weight.copy_(Wf.to(mod.weight.dtype))
            norm.weight.zero_()
    if mode == "rot_writers":
        for mod in writers(L): mod.weight.copy_((R.T @ (R @ mod.weight.float())).to(mod.weight.dtype))
def ev(tag):
    r = H.evaluate("select", docs=docs); out[tag] = dict(kl=r["kl"], agree=r["agree"]); print("%-40s KL=%.2e agree=%.4f" % (tag, r["kl"], r["agree"]), flush=True)
H.restore(); ev("BF16 untouched (noise floor of this doc subset)")
for i in (0, 1, 7, 31):
    for mode, which in (("fold", (0,)), ("fold", (1,)), ("fold_rot", (0, 1)), ("rot_writers", ())):
        H.restore(); apply(i, mode, which); ev(f"L{i} {mode} norms{which}")
# gain statistics of the folded dimension
for i in (0, 7, 31):
    L = lm.layers[i]; H.restore()
    for nm, norm in (("input", L.input_layernorm), ("post_attn", L.post_attention_layernorm)):
        g = 1 + norm.weight.float(); print("L%d %s gain: min %.4f max %.2f  #|g|<0.05: %d  #g<0: %d  #g>4: %d" % (i, nm, g.min(), g.max(), int((g.abs() < 0.05).sum()), int((g < 0).sum()), int((g > 4).sum())))
json.dump(out, open(H.OUT + "/diag_rot_exactness.json", "w"), indent=1); H.restore()
