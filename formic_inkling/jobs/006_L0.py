"""C3 (couverture des experts sur le professeur) + C4 (passe L0 = F1 transposée, budget UD-IQ2_XXS)."""
import sys, json, time, torch
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q
for m in (E, Q): importlib.reload(m)
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model); Q.restore()
X = torch.load('/workspace/work_ink/data/cal_ink.pt')[:1024]
meta = json.load(open('/workspace/work_ink/data/cal_ink_meta.json'))
# ---------------- C3 : tokens routés par expert, couche et application, sur le professeur (calibration complète)
cnt = {}; hooks = []
def mk(i):
    def h(m, inp, out):
        idx = out[2]; c = torch.bincount(idx.reshape(-1), minlength=128).cpu()
        cnt.setdefault(i, []).append(c)
    return h
for i, L in enumerate(Q.S['lm'].layers): hooks.append(L.mlp.gate.register_forward_hook(mk(i)))
t0 = time.time(); per_app = {}
with torch.no_grad():
    for b in range(0, X.shape[0], 8):
        cnt.clear(); Q.S['lm'](input_ids=X[b:b + 8].cuda().long())
        for i, l in cnt.items():
            for a, c in enumerate(l):
                k = f'{i}.{a}'; per_app[k] = per_app.get(k, 0) + c
for h in hooks: h.remove()
tfwd = time.time() - t0
tot = {}
for k, c in per_app.items():
    i = int(k.split('.')[0]); tot[i] = tot.get(i, 0) + c
under13 = {i: [e for e in range(128) if int(tot[i][e]) < Q.NMIN['w13']] for i in tot}
under2 = {i: [e for e in range(128) if int(tot[i][e]) < Q.NMIN['w2']] for i in tot}
n_under = sum(len(v) for v in under13.values())
cal = dict(tokens=int(X.numel()), docs=int(X.shape[0]), doc_len=2048, composition_target=meta['composition_target'],
           composition_real={k: v / 1025 for k, v in meta['counts'].items()}, counts_docs=meta['counts'],
           sources=sorted(set(d['dataset'] for d in meta['docs'])), doc_source_ids=[d['source_id'] for d in meta['docs'][:1024]],
           threshold=dict(n_min_w13=Q.NMIN['w13'], n_min_w2=Q.NMIN['w2'], counted='tokens x applications (x3 dans le coeur)'),
           routed_tokens_teacher={k: v.tolist() for k, v in per_app.items()},
           routed_tokens_teacher_total_per_layer={str(i): v.tolist() for i, v in tot.items()},
           under_threshold_w13={str(i): v for i, v in under13.items() if v}, under_threshold_w2={str(i): v for i, v in under2.items() if v},
           n_experts_under_w13=n_under, frac_experts_under_w13=n_under / 3072, gate_C3='arrêt si > 5 % des experts sous le seuil',
           gate_passed=n_under / 3072 <= 0.05, teacher_forward_seconds=tfwd,
           min_count_per_layer={str(i): int(v.min()) for i, v in tot.items()})
json.dump(cal, open(E.RES + '/calibration_ink.json', 'w'), indent=0)
print('C3: forward professeur %.0fs (%.0f tok/s) ; experts sous le seuil w13 : %d / 3072 (%.2f %%) ; min par couche %s' % (
      tfwd, X.numel() / tfwd, n_under, 100 * n_under / 3072, cal['min_count_per_layer']), flush=True)
# ---------------- C4 : pilote, calage, passe
t0 = time.time(); ratio = Q.pilot(X); print('pilote (rapport bits LDLQ / plus proche) :', ratio, '%.0fs' % (time.time() - t0), flush=True)
sbytes = Q.small_bytes(); target = (Q.BUDGET_TEXT - sbytes) * 8
spec, est = Q.calibrate(ratio, target, dict(mult=0.6)); spec['_ratio'] = ratio
print('multiplicateur global L0 = %.6f ; bits estimés %.4e (cible %.4e) ; petits tenseurs %d octets' % (spec['mult'], est, target, sbytes), flush=True)
info, cov = Q.run_pass('L0', spec, X)
t1 = time.time(); per = E.evaluate(); te = time.time() - t1
rd = E.report(per, 'L0', info=dict(info, eval_seconds=te))
json.dump(dict(id='L0', info=info, dev=rd), open(E.RES + '/variant_L0.json', 'w'), indent=1)
torch.save([{k: v for k, v in w.items() if k != 'route'} for w in per], '/workspace/work_ink/sealed/perwin_L0.pt')
Q.S.setdefault('states', {})['L0'] = {n: p.detach().cpu() for n, p in model.named_parameters() if not n.startswith('model.vision_tower')}   # en RAM (disque limité)
print('eval %.0fs' % te)
