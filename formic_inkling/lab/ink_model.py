"""Chargeur du professeur Inkling 2 Mini (W4A16) — formic_inkling.

CORRECTIF (mesuré le 22/11/2026, results/teacher_check.json) : sans quantifieur, le chargeur de Transformers 5.24 convertit
toutes les sources dans le type du paramètre (BF16) AVANT `Inkling2LoopDequantizeExperts`. Le facteur FP32 par expert
`weight_amax / weight_outer_scale_denominator` est alors arrondi en BF16 : chaque expert NVFP4 (couches 0–19) est mis à une
échelle fausse de ±0,4 % environ. Ici, les experts NVFP4 sont redéquantifiés depuis les codes avec le facteur FP32 exact
(identique bit à bit au GGUF UD-Q8_K_XL d'Unsloth), puis arrondis en BF16 (RNE). Les experts MXFP8 (couches 20–23) étaient
déjà exacts (échelle puissance de deux)."""
import os, time, torch
MODEL_DIR = os.environ.get("INK_MODEL_DIR", "/root/models/inkling-2-mini")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def nvfp4_codes(f, pre):
    return {k: f.get_tensor(pre + k) for k in ("", "_scale", "_amax", "_outer_scale_denominator")}


@torch.no_grad()
def deq_nvfp4(d, interleaved, device="cuda", bf16_factor=False):
    """Déquantification NVFP4 en FP32 : e2m1(code) × fp8(échelle du bloc de 16) × (amax[e] / dénominateur)."""
    e2m1 = torch.tensor(E2M1, device=device); byte = torch.arange(256, device=device)
    b2v = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
    wt = d[""].to(device); sc = d["_scale"].to(device).float()
    amax = d["_amax"].to(device).float(); den = d["_outer_scale_denominator"].to(device).float()
    if bf16_factor: amax = amax.bfloat16().float(); den = den.bfloat16().float()
    E_, rows = wt.shape[:2]; cols = wt.shape[2] * 2
    out = torch.empty(E_, rows, cols, dtype=torch.float32, device=device)
    for s in range(0, E_, 16):
        v = b2v[wt[s:s + 16].long()].view(min(16, E_ - s), rows, -1, 16) * sc[s:s + 16][..., None]
        v = v * (amax[s:s + 16] / den)[:, None, None, None]
        out[s:s + 16] = v.reshape(-1, rows, cols)
    if interleaved:   # (gate0, up0, gate1, up1, ...) -> [gate ; up]
        out = out.view(E_, rows // 2, 2, cols).transpose(1, 2).reshape(E_, rows, cols)
    return out


@torch.no_grad()
def fix_nvfp4_experts(m, log=print):
    from safetensors import safe_open
    lm = m.model.language_model; rep = {}
    with safe_open(os.path.join(MODEL_DIR, "model-00001-of-00001.safetensors"), "pt") as f:
        for i, L in enumerate(lm.layers):
            for w, P in (("w13", L.mlp.experts.gate_up_proj), ("w2", L.mlp.experts.down_proj)):
                pre = f"model.llm.layers.{i}.mlp.experts.{w}_weight"
                if pre + "_amax" not in f.keys(): continue
                d = nvfp4_codes(f, pre)
                hyp = torch.equal(deq_nvfp4(d, w == "w13", bf16_factor=True).to(P.dtype), P)   # explication du défaut
                ex = deq_nvfp4(d, w == "w13")
                changed = float((ex.to(P.dtype) != P).float().mean())
                P.copy_(ex.to(P.dtype)); del ex
                rep[f"{i}.{w}"] = dict(transformers_equals_bf16_factor_dequant=hyp, frac_weights_changed=changed)
    log("[ink_model] experts NVFP4 corrigés (facteur FP32 exact) : %d tenseurs ; défaut expliqué par l'arrondi BF16 du facteur : %s ; part des poids modifiés %.3f" % (
        len(rep), all(v["transformers_equals_bf16_factor_dequant"] for v in rep.values()), sum(v["frac_weights_changed"] for v in rep.values()) / len(rep)))
    return rep


def load_teacher(dtype=torch.bfloat16, experts_impl="grouped_mm", device="cuda", fix=True):
    from transformers import Inkling2LoopForConditionalGeneration as C
    t = time.time()
    m = C.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device, experts_implementation=experts_impl)
    m.eval()
    m._formic_fix = fix_nvfp4_experts(m) if fix else None
    print(f"[ink_model] loaded in {time.time()-t:.0f}s", flush=True)
    return m
