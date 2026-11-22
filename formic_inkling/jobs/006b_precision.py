"""Règle § 4.1.3 isolée : KL(experts FP32 au calcul FP32 || experts ARRONDIS BF16 au calcul FP32) -> effet du seul arrondi des poids.
En complément (information) : effet du chemin de calcul des experts (BF16 grouped_mm, BF16 boucle d'origine, FP32).
Sous-ensemble : 18 fenêtres dev (2 par domaine)."""
import sys, json, time, types, os, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q, ink_model as IM
for m in (E, Q, IM): importlib.reload(m)
from safetensors import safe_open
torch.backends.cuda.matmul.allow_tf32 = False
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model); Q.restore()
lm = E.S['lm']; A = E.S['pi']
rows = []; seen = {}
for i, (d, p) in enumerate(zip(A['dom'], A['part'])):
    if p == 'dev' and seen.get(d, 0) < 2: rows.append(i); seen[d] = seen.get(d, 0) + 1
f = safe_open('/root/models/inkling-2-mini/model-00001-of-00001.safetensors', 'pt')
codes = {}
for i in range(24):
    for w in ('w13', 'w2'):
        pre = f'model.llm.layers.{i}.mlp.experts.{w}_weight'
        if pre + '_amax' in f.keys(): codes[(i, w)] = {k: v.pin_memory() for k, v in IM.nvfp4_codes(f, pre).items()}
def W32(ex, i, w):
    if (i, w) in codes: return IM.deq_nvfp4(codes[(i, w)], w == 'w13')
    return (ex.gate_up_proj if w == 'w13' else ex.down_proj).float()       # MXFP8 : exact en BF16
def make(mode):
    def fwd(self, hs, idx, wts):
        if mode == 'fp32w': W13, W2 = W32(self, self._li, 'w13'), W32(self, self._li, 'w2')
        else: W13, W2 = self.gate_up_proj.float(), self.down_proj.float()
        out = torch.zeros_like(hs, dtype=torch.float32)
        mask = F.one_hot(idx, num_classes=self.num_experts + 1).permute(2, 1, 0)
        for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            e = e[0]
            if e == self.num_experts: continue
            pos, tid = torch.where(mask[e])
            g, u = F.linear(hs[tid].float(), W13[e]).chunk(2, dim=-1)
            out.index_add_(0, tid, F.linear(self.act_fn(g, u), W2[e]) * wts[tid, pos, None])
        return out.to(hs.dtype)
    return fwd
def run(mode):
    saved = {}
    for i, L in enumerate(lm.layers):
        ex = L.mlp.experts; ex._li = i; saved[i] = ex.forward
        if mode in ('fp32w', 'bf16w_fp32c'): ex.forward = types.MethodType(make(mode), ex)
        elif mode == 'eager_bf16':
            ex.forward = types.MethodType(type(ex).forward.__wrapped__ if hasattr(type(ex).forward, '__wrapped__') else saved[i], ex)
    H = []
    try:
        with torch.no_grad():
            if mode == 'eager_bf16':
                cfg = lm.config; old = cfg._experts_implementation; cfg._experts_implementation = 'eager'
            try:
                for r in rows: H.append(lm(input_ids=A['tok'][r:r + 1].cuda().long()).last_hidden_state[0, E.P0:E.P1].cpu())
            finally:
                if mode == 'eager_bf16': cfg._experts_implementation = old
    finally:
        for i, L in enumerate(lm.layers): L.mlp.experts.forward = saved[i]
    return H
def kl(Ha, Hb):
    ks = []; ag = []
    with torch.no_grad():
        for a, b in zip(Ha, Hb):
            la = F.log_softmax(E.softcap(F.linear(a.cuda(), E.S['T_head']).float()), -1); lb = F.log_softmax(E.softcap(F.linear(b.cuda(), E.S['T_head']).float()), -1)
            ks.append((la.exp() * (la - lb)).sum(-1).cpu()); ag.append((la.argmax(-1) == lb.argmax(-1)).float().cpu())
    k = torch.cat(ks); return dict(kl=float(k.mean()), p99=float(torch.quantile(k.double(), .99)), top1=float(torch.cat(ag).mean()))
t0 = time.time()
H32 = run('fp32w'); Hb = run('bf16w_fp32c'); He = run('eager_bf16'); Hg = [E.S['teach'][r] for r in rows]
res = dict(windows=len(rows), positions=len(rows) * 2048,
           rule_weights_only=dict(desc='KL(poids FP32, calcul FP32 || poids BF16, calcul FP32)', **kl(H32, Hb)),
           compute_path=dict(fp32_vs_grouped_mm_bf16=kl(H32, Hg), fp32_vs_eager_bf16=kl(H32, He), eager_bf16_vs_grouped_mm_bf16=kl(He, Hg),
                             bf16w_fp32c_vs_grouped_mm_bf16=kl(Hb, Hg)), seconds=time.time() - t0)
res['accepted_bf16_weights'] = res['rule_weights_only']['kl'] <= 1e-4
tc = json.load(open(E.RES + '/teacher_check.json')); tc['precision_isolated'] = res
json.dump(tc, open(E.RES + '/teacher_check.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
