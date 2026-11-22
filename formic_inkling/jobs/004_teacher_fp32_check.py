"""C1 : KL(professeur_FP32 || professeur_BF16) sur la moitié dev de P-I (règle déclarée : accepté si <= 1e-4).
Professeur FP32 : experts déquantifiés en FP32 à la volée depuis les codes NVFP4/MXFP8 du checkpoint (copiés depuis la RAM à
chaque application), produits experts en FP32 (TF32 désactivé). Le reste du graphe est identique."""
import sys, json, time, types, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E; importlib.reload(E)
from safetensors import safe_open
torch.backends.cuda.matmul.allow_tf32 = False
E.init(model, tok); E.load_protocol(); E.teacher_cache()
lm = E.S['lm']; A = E.S['pi']
dev = [i for i, p in enumerate(A['part']) if p == 'dev']
f = safe_open('/root/models/inkling-2-mini/model-00001-of-00001.safetensors', 'pt')
E2M1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0), device='cuda')
byte = torch.arange(256, device='cuda'); B2V = torch.stack((E2M1[byte & 0xF], E2M1[byte >> 4]), dim=-1)
codes = {}
for i in range(24):
    for w in ('w13', 'w2'):
        pre = f'model.llm.layers.{i}.mlp.experts.{w}_weight'
        d = {k: f.get_tensor(pre + k).pin_memory() for k in ('', '_scale', '_amax', '_outer_scale_denominator') if (pre + k) in f.keys()}
        codes[(i, w)] = d
def deq(i, w):
    d = {k: v.cuda(non_blocking=True) for k, v in codes[(i, w)].items()}
    wt, sc = d[''], d['_scale']
    E_, rows = wt.shape[:2]
    if '_amax' in d:
        cols = wt.shape[2] * 2
        v = B2V[wt.long()].view(E_, rows, -1, 16) * sc.float()[..., None]
        v = v * (d['_amax'].float() / d['_outer_scale_denominator'].float())[:, None, None, None]
    else:
        cols = wt.shape[2]
        v = wt.float().view(E_, rows, -1, 32) * torch.exp2(sc.float() - 127)[..., None]
    v = v.reshape(E_, rows, cols)
    if w == 'w13':  # désentrelacement (gate0, up0, ...) -> [gate ; up]
        v = v.view(E_, rows // 2, 2, cols).transpose(1, 2).reshape(E_, rows, cols)
    return v
# vérification : BF16(deq FP32) == poids chargés par Transformers
for i in (0, 13, 21):
    ex = lm.layers[i].mlp.experts
    print('check layer', i, torch.equal(deq(i, 'w13').bfloat16(), ex.gate_up_proj), torch.equal(deq(i, 'w2').bfloat16(), ex.down_proj))
def fwd32(self, hidden_states, top_k_index, top_k_weights):
    W13 = deq(self._li, 'w13'); W2 = deq(self._li, 'w2')
    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for e in hit:
        e = e[0]
        if e == self.num_experts: continue
        pos, tid = torch.where(mask[e])
        x = hidden_states[tid].float()
        gu = F.linear(x, W13[e]); g, u = gu.chunk(2, dim=-1)
        h = self.act_fn(g, u)
        y = F.linear(h, W2[e]) * top_k_weights[tid, pos, None]
        out.index_add_(0, tid, y)
    return out.to(hidden_states.dtype)
saved = {}
for i, L in enumerate(lm.layers):
    ex = L.mlp.experts; ex._li = i; saved[i] = ex.forward; ex.forward = types.MethodType(fwd32, ex)
t0 = time.time()
try:
    # référence = professeur FP32 : on met en cache ses états, puis le professeur BF16 est le « candidat »
    T32 = []
    with torch.no_grad():
        for i in dev:
            T32.append(lm(input_ids=A['tok'][i:i + 1].cuda().long()).last_hidden_state[0, E.P0:E.P1].cpu())
finally:
    for i, L in enumerate(lm.layers): L.mlp.experts.forward = saved[i]
print('fp32 teacher pass %.0fs' % (time.time() - t0))
bf = E.S['teach']; kls = []; ag = []
with torch.no_grad():
    for k, i in enumerate(dev):
        lpt = F.log_softmax(E.softcap(F.linear(T32[k].cuda(), E.S['T_head']).float()), -1)
        lps = F.log_softmax(E.softcap(F.linear(bf[i].cuda(), E.S['T_head']).float()), -1)
        kls.append((lpt.exp() * (lpt - lps)).sum(-1).cpu()); ag.append((lpt.argmax(-1) == lps.argmax(-1)).float().cpu())
k = torch.cat(kls); a = torch.cat(ag)
res = dict(rule='KL(prof_FP32 || prof_BF16) <= 1e-4 sur P-I dev -> professeur BF16 accepté', windows=len(dev), positions=int(k.numel()),
           kl_fp32_vs_bf16=float(k.mean()), kl_p99=float(torch.quantile(k.double(), 0.99)), top1_agree=float(a.mean()),
           accepted=bool(float(k.mean()) <= 1e-4), frac_nvfp4_weights_not_bf16_exact=0.931,
           note='experts FP32 déquantifiés depuis les codes ; activations d\'entrée des experts en BF16 (comme le graphe), produits FP32')
old = json.load(open(E.RES + '/teacher_check.json')) if __import__('os').path.exists(E.RES + '/teacher_check.json') else {}
old['fp32_vs_bf16_before_fix'] = old.get('fp32_vs_bf16_before_fix', dict(kl_fp32_vs_bf16=0.018765602260828018, top1_agree=0.9755226373672485, accepted=False, note='professeur BF16 de Transformers (facteur arrondi)'))
old['fp32_vs_bf16'] = res
json.dump(old, open(E.RES + '/teacher_check.json', 'w'), indent=1)
print(res)
