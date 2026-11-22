"""Phase 2 (demande de l'opérateur, 22/11 16h25 : « trouver des solutions pour que la KL soit plus basse ») — variante V4 du mandat :
attention (q, k, v, wr_du, wo), embed et unembed à un multiplicateur 2× plus fin (+1 bit), experts compensés à octets égaux."""
import sys, json, time, torch
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q
for m in (E, Q): importlib.reload(m)
import formic_q as fq; fq.set_limit(128)
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model); Q.restore()
VID = 'V4'; EXTRA = dict(fine=1.0)
L0 = json.load(open('/workspace/work_ink/packages/L0.pass.json'))['info']
X = torch.load('/workspace/work_ink/data/cal_ink.pt')[:L0['spec']['_ndoc']]
ratio = L0['spec']['_ratio']
target = (Q.BUDGET_TEXT - Q.small_bytes()) * 8
spec, est = Q.calibrate(ratio, target, dict(mult=L0['spec']['mult'], **EXTRA)); spec['_ratio'] = ratio; spec['_ndoc'] = L0['spec']['_ndoc']
print(VID, 'multiplicateurs :', {k: v for k, v in spec.items() if k.startswith('mult')}, flush=True)
info, cov = Q.run_pass(VID, spec, X)
t1 = time.time(); per = E.evaluate(); te = time.time() - t1
rd = E.report(per, VID, info=dict(info, eval_seconds=te))
json.dump(dict(id=VID, info=info, dev=rd), open(E.RES + f'/variant_{VID}.json', 'w'), indent=1)
l0 = json.load(open(E.RES + '/pi_dev_L0.json')); iq2 = json.load(open(E.RES + '/pi_dev_competitor_UD-IQ2_XXS.json'))
print(VID, 'contre L0 (dev) :', E.boot_ratio(rd['per_window'], l0['per_window']), flush=True)
print(VID, 'contre UD-IQ2_XXS (dev) :', E.boot_ratio(rd['per_window'], iq2['per_window']), flush=True)
