# First measurement on the Vision path (diagnostic, not a selection experiment): image-conditioned continuations.
# Teacher (BF16) greedily generates <=64 tokens per image prompt; KL(teacher||student) is measured on those
# positions, teacher-forced.  Variants: vision tower only / text only / both, no-feedback E8 at the control multiplier.
import json, time, torch.nn.functional as F
import formic_q as fq
st = H.S["st"]; m0 = H.S["mult_Q0rtn"]; H.restore()
data = torch.load("/workspace/work/data/vision/vision_inputs.pt")
dt = model.dtype
def prep(s, ids=None):
    inp = {k: v.cuda() for k, v in s["inputs"].items()}
    if "pixel_values" in inp: inp["pixel_values"] = inp["pixel_values"].to(dt)
    if ids is not None:
        extra = ids.shape[1] - inp["input_ids"].shape[1]
        if "mm_token_type_ids" in inp and extra > 0:      # generated tokens are text (type 0)
            tt = inp["mm_token_type_ids"]
            inp["mm_token_type_ids"] = torch.cat([tt, torch.zeros(1, extra, dtype=tt.dtype, device=tt.device)], 1)
        inp["input_ids"] = ids; inp["attention_mask"] = torch.ones_like(ids)
    return inp
T = []; t = time.time()
with torch.no_grad():
    for s in data:
        inp = prep(s); L = inp["input_ids"].shape[1]
        full = model.generate(**inp, max_new_tokens=64, do_sample=False)
        lg = model(**prep(s, full)).logits[0, L - 1:-1].float()
        T.append(dict(full=full.cpu(), L=L, lp=F.log_softmax(lg, -1).to(torch.float16).cpu()))
print("teacher generation+logits %.0fs; continuation lengths:" % (time.time() - t), [x["full"].shape[1] - x["L"] for x in T])
print("example continuation:", repr(tok.decode(T[0]["full"][0, T[0]["L"]:])[:200]))
@torch.no_grad()
def ev():
    out = {}
    for s, tt in zip(data, T):
        lg = model(**prep(s, tt["full"].cuda())).logits[0, tt["L"] - 1:-1].float()
        lps = F.log_softmax(lg, -1); lpt = tt["lp"].cuda().float()
        kl = (lpt.exp() * (lpt - lps)).sum(-1).mean().item(); ag = (lps.argmax(-1) == lpt.argmax(-1)).float().mean().item()
        out.setdefault(s["kind"], []).append((kl, ag))
    allv = [v for vs in out.values() for v in vs]
    r = dict(kl=sum(v[0] for v in allv) / len(allv), agree=sum(v[1] for v in allv) / len(allv),
             by_kind={k: dict(kl=sum(v[0] for v in vs) / len(vs), agree=sum(v[1] for v in vs) / len(vs)) for k, vs in out.items()})
    return r
def quant_vision(mult):
    bits = 0.0; n = 0
    with torch.no_grad():
        for name, mod in model.model.visual.named_modules():
            W = getattr(mod, "weight", None)
            if W is None or W.dim() < 2 or W.numel() < 65536: continue
            W2 = W.reshape(W.shape[0], -1)
            if W2.shape[1] % 8: continue
            Q, hist = fq.quantize_rtn(W2, fq.row_steps(W2, mult))
            W.copy_(Q.reshape(W.shape).to(W.dtype)); bits += fq.matrix_bits(hist, W2.shape[0], W2.shape[1]); n += W2.numel()
    return bits, n
res = {"noise_floor": ev()}
print("BF16 self-check:", res["noise_floor"])
b, n = quant_vision(m0); res["vision_only"] = dict(ev(), bpw=b / n, params=n)
print("vision tower only (%.3f bpw on %.1f M weights):" % (b / n, n / 1e6), res["vision_only"], flush=True)
H.restore(); info = H.quantize_all(H.qmats(), dict(rule="lumen"), m0, st); res["text_only"] = dict(ev(), bpw=info["bpw"])
print("text path only (%.3f bpw):" % info["bpw"], res["text_only"], flush=True)
b, n = quant_vision(m0); res["both"] = dict(ev())
print("both:", res["both"], flush=True)
json.dump(res, open(H.OUT + "/diag_vision_check.json", "w"), indent=1)
H.restore(); del T
