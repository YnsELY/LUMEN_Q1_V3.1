"""C5 : variante V1a — coeur bouclé (couches 6-17 : experts + attention) +0,2 bit, prélude et coda -0,2 bit, à octets égaux.
Même calibration, même pilote (rapports LDLQ/plus proche de L0), même ajustement final de la tête sur le budget."""
import sys, json, time, torch
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q
for m in (E, Q): importlib.reload(m)
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model); Q.restore()
L0 = json.load(open('/workspace/work_ink/packages/L0.pass.json'))['info']
X = torch.load('/workspace/work_ink/data/cal_ink.pt')[:L0['spec']['_ndoc']]
ratio = L0['spec']['_ratio']
sbytes = Q.small_bytes(); target = (Q.BUDGET_TEXT - sbytes) * 8
spec, est = Q.calibrate(ratio, target, dict(mult=L0['spec']['mult'], delta=0.2, delta_edge=0.2)); spec['_ratio'] = ratio; spec['_ndoc'] = L0['spec']['_ndoc']
print('V1a : multiplicateur commun %.6f -> coeur %.6f, bords %.6f (L0 : %.6f)' % (spec['mult'], spec['mult_core'], spec['mult_edge'], L0['spec']['mult']), flush=True)
info, cov = Q.run_pass('V1a', spec, X)
t1 = time.time(); per = E.evaluate(); te = time.time() - t1
rd = E.report(per, 'V1a', info=dict(info, eval_seconds=te))
json.dump(dict(id='V1a', info=info, dev=rd), open(E.RES + '/variant_V1a.json', 'w'), indent=1)
torch.save([{k: v for k, v in w.items() if k != 'route'} for w in per], '/workspace/work_ink/sealed/perwin_V1a.pt')
Q.S.setdefault('states', {})['V1a'] = {n: p.detach().cpu() for n, p in model.named_parameters() if not n.startswith('model.vision_tower')}
l0 = json.load(open(E.RES + '/pi_dev_L0.json'))
print('V1a contre L0 (dev) :', E.boot_ratio(rd['per_window'], l0['per_window']))
