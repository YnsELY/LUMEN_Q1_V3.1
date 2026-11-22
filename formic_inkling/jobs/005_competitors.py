"""C1/C2 : GGUF d'Unsloth déquantifiés et chargés dans notre graphe (rangement vérifié tenseur par tenseur), évalués sur P-I.
UD-Q8_K_XL = contrôle (KL ~ 0 attendue). Moitié dev imprimée ; moitié conf scellée."""
import sys, json, time, torch
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q, ink_gguf as GG
for m in (E, Q, GG): importlib.reload(m)
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model); Q.restore()
import os
SEC = os.path.exists('/workspace/work_ink/data/pd32_prompts.json')
if SEC: print('secondaires professeur :', E.pw_teacher(), E.pd32_teacher(), flush=True)
GD = '/workspace/work_ink/gguf/Inkling-2-Mini-%s.gguf'
for q in ['UD-Q8_K_XL', 'UD-IQ2_XXS', 'UD-Q2_K_XL', 'UD-IQ3_XXS']:
    t0 = time.time(); rep = GG.load_gguf(GD % q)
    print('[%s] chargé : %d tenseurs, cos min %.5f, %.0fs ; experts NVFP4 bit-exacts : %s' % (q, rep['loaded'], rep['min_cos'], rep['seconds'],
          sum(v['bit_exact_vs_teacher'] for v in rep['tensors'].values())), flush=True)
    t1 = time.time(); per = E.evaluate()
    rd = E.report(per, 'competitor_' + q, info=dict(file=rep['file'], file_bytes=rep['file_bytes'], text_tensor_bytes=rep['tensor_bytes']))
    out = dict(id=q, dev=rd, load_report={k: v for k, v in rep.items() if k != 'tensors'}, tensors=rep['tensors'], eval_seconds=time.time() - t1)
    json.dump(out, open(f'{E.RES}/competitor_{q}.json', 'w'), indent=1)
    torch.save([{k: v for k, v in w.items() if k != 'route'} for w in per], f'/workspace/work_ink/sealed/perwin_{q}.pt')
    if SEC and q != 'UD-Q8_K_XL':      # mesures secondaires, scellées jusqu'au gel
        json.dump(dict(pw=E.pw_eval(), pd32=E.pd32_eval()), open(f'/workspace/work_ink/sealed/secondary_{q}.json', 'w'), indent=1)
    if q == 'UD-Q8_K_XL':
        tc = json.load(open(E.RES + '/teacher_check.json')) if __import__('os').path.exists(E.RES + '/teacher_check.json') else {}
        tc['ud_q8_k_xl_in_our_graph'] = dict(kl_dev=rd['kl'], top1_dev=rd['agree'], routing_overlap_dev=rd['routing']['mean_overlap'],
            nvfp4_experts_bit_exact=[k for k, v in rep['tensors'].items() if v['bit_exact_vs_teacher']],
            not_bit_exact={k: dict(type=v['type'], cos=v['cos'], rel_err=v['rel_err']) for k, v in rep['tensors'].items() if not v['bit_exact_vs_teacher']},
            criterion='KL < 1e-4', passed=rd['kl'] < 1e-4)
        json.dump(tc, open(E.RES + '/teacher_check.json', 'w'), indent=1)
Q.restore()
print('restored teacher')
