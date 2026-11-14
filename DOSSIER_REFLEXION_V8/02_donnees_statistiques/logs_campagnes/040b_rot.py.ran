# A_rot x Q0: exact residual-stream rotation (Haar orthogonal, RMSNorm gains folded into readers), emulated in the
# original graph by quantizing every matrix in the rotated coordinates; same E8-LDLQ, reference row-RMS steps,
# every matrix rate-matched individually to the control's bytes.
import importlib, harness, seq, json, formic_q
importlib.reload(formic_q); _S = dict(harness.S); importlib.reload(harness); harness.S.update(_S); H = harness
importlib.reload(seq); SQ = seq
st = H.S["st"]; CTRL_BPW = json.load(open(H.OUT + "/variant_A0_Q0ldlq.json"))["info"]["bpw"]
lm = H.S["lm"]
gmin = min(float((1 + L.input_layernorm.weight.float()).abs().min()) for L in lm.layers), min(float((1 + L.post_attention_layernorm.weight.float()).abs().min()) for L in lm.layers), float((1 + lm.norm.weight.float()).abs().min())
print("min |1+w| of RMSNorm gains (input norms, post-attn norms, final norm):", gmin)
# exactness of the emulated basis change WITHOUT quantization: fold gains + rotate + map back on a few layers,
# then compare the model output with the BF16 teacher (must be at the numerical noise floor)
R = SQ._rot_matrix(5120).float()
with torch.no_grad():
    for i in (0, 7, 31):
        L = lm.layers[i]
        for norm, mods in ((L.input_layernorm, ([L.linear_attn.in_proj_qkv, L.linear_attn.in_proj_z, L.linear_attn.in_proj_a, L.linear_attn.in_proj_b] if hasattr(L, "linear_attn") else [L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj])),
                           (L.post_attention_layernorm, [L.mlp.gate_proj, L.mlp.up_proj])):
            g = 1 + norm.weight.float()
            for mod in mods:
                Wp = (mod.weight.float() * g) @ R.T            # rotated-basis weight
                mod.weight.copy_((Wp @ R).to(mod.weight.dtype))  # mapped back, gain folded
            norm.weight.zero_()
        for mod in ((L.linear_attn.out_proj if hasattr(L, "linear_attn") else L.self_attn.o_proj), L.mlp.down_proj):
            mod.weight.copy_((R.T @ (R @ mod.weight.float())).to(mod.weight.dtype))
r = H.evaluate("select", docs=list(range(0, 72, 6)))
print("exact basis change on layers 0, 7, 31 (no quantization): KL=%.2e agree=%.4f" % (r["kl"], r["agree"]), flush=True)
del R
spec = dict(rule="lumen", rot=True, match_bpw=2.8095)
H.restore(); mats = SQ.plan_mats()
pil = SQ.ldlq_pass(mats, spec, H.S["mult_Q0ldlq"], st, only={1, 2, 3}, log_every=1)
print("pilot (layers 1-3) bpw %.4f with match_bpw %.4f ; control overall %.4f" % (pil["bpw"], spec["match_bpw"], CTRL_BPW), {k: round(v, 4) for k, v in pil["bykind"].items()}, flush=True)
r = H.evaluate("select", docs=list(range(0, 72, 3))); print("layers 1-3 only quantized in rotated basis: KL=%.5f" % r["kl"], flush=True)
spec["match_bpw"] = round(spec["match_bpw"] - (pil["bpw"] - CTRL_BPW) - 0.0005, 4)
print("match_bpw set to", spec["match_bpw"], flush=True)
SQ.run_ldlq_variant("Arot_Q0ldlq", spec, H.S["mult_Q0ldlq"], method="E8-LDLQ sequential in a rotated residual basis (Haar orthogonal, gains folded), per-matrix rate matching")
