"""Phase 2 : sélection sur P-I dev (results/selection_rule_phase2.lock.json), gel, lecture UNIQUE de P-I2 (scellé) sur le gagnant,
L0 et UD-IQ2_XXS ; remplacement du paquet Formic V1.5 seulement si le gain contre L0 sur P-I2 a un IC 95 % entièrement sous 0."""
import sys, os, json, time, subprocess, torch, traceback
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q, ink_package as PK, ink_gguf as GG
for m in (E, Q, PK, GG): importlib.reload(m)
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model); Q.restore()
RES = E.RES; SE = E.SEALED
def git(msg):
    subprocess.run(['git', '-C', '/workspace/LUMEN_Q1_V3.1', 'add', 'formic_inkling/results'], check=False)
    subprocess.run(['git', '-C', '/workspace/LUMEN_Q1_V3.1', 'commit', '-q', '-m', msg + '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_0125FrrgqPHPHdVAGaLuUD4S'], check=False)
# ------------------------------------------------ sélection de phase 2 (P-I dev)
l0 = json.load(open(f'{RES}/pi_dev_L0.json')); iq2 = json.load(open(f'{RES}/pi_dev_competitor_UD-IQ2_XXS.json'))
order = ['L0', 'V1a', 'V4', 'C1']
cands = [v for v in order if os.path.exists(f'{RES}/pi_dev_{v}.json') and os.path.exists(f'{Q.PKG_ROOT}/{v}.pass.json')]
rows = {}
for v in cands:
    r = json.load(open(f'{RES}/pi_dev_{v}.json'))
    rows[v] = dict(kl=r['kl'], routing=r['routing']['mean_overlap'], guard_routing=r['routing']['mean_overlap'] >= l0['routing']['mean_overlap'] - 0.01,
                   worst_domain_ratio_vs_iq2=max(r['by_domain'][d]['kl'] / iq2['by_domain'][d]['kl'] for d in r['by_domain']),
                   vs_L0=E.boot_ratio(r['per_window'], l0['per_window']) if v != 'L0' else None)
ok = [v for v in cands if rows[v]['guard_routing']]
best = min(ok, key=lambda v: rows[v]['kl']); first = best; ties = []
for v in ok:
    if v == first: continue
    A_ = json.load(open(f'{RES}/pi_dev_{v}.json')); B_ = json.load(open(f'{RES}/pi_dev_{first}.json'))
    br = E.boot_ratio(A_['per_window'], B_['per_window'])
    if br['ci95'][0] <= 0 <= br['ci95'][1]:
        wv = sum(A_['by_domain'][d]['kl'] < B_['by_domain'][d]['kl'] for d in A_['by_domain']); wb = len(A_['by_domain']) - wv
        ties.append(dict(other=v, vs=first, ratio=br, won_other=wv, won_first=wb))
        if wv > wb or (wv == wb and order.index(v) < order.index(best)): best = v
sel = dict(frozen_at=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()), rule='results/selection_rule_phase2.lock.json', candidates=rows, selected=best, ties=ties,
           pi2_read_before_freeze=False)
json.dump(sel, open(f'{RES}/selection_phase2.lock.json', 'w'), indent=1); git(f'formic_inkling: sélection de phase 2 gelée avant lecture de P-I2 ({best})')
print('PHASE 2 — SÉLECTION GELÉE :', best, {k: round(v['kl'], 5) for k, v in rows.items()}, flush=True)
# ------------------------------------------------ P-I2 : cache du professeur (poids du professeur)
saved = {k: E.S[k] for k in ('pi', 'teach', 'troute')}
d2 = torch.load(f'{E.D}/pi2_tokens.pt')
E.S['pi'] = dict(tok=d2['tokens'], dom=d2['domain'], part=d2['part'], sid=d2['source_ids'])
E.S.pop('teach'); E.S.pop('troute'); print('P-I2 professeur :', E.teacher_cache(path=f'{E.D}/pi2_teacher.pt'), flush=True)
res = {}
def run(name):
    per = E.evaluate(); s = E.summarize(per, 'P-I2'); res[name] = s
    json.dump(s, open(f'{SE}/pi2_{name}.json', 'w')); print('P-I2', name, E.short(s), flush=True)
for v in dict.fromkeys(['L0', best]):
    Q.restore(); PK.load_package(f'{Q.PKG_ROOT}/{v}', blank=False); run(v)
rep = GG.load_gguf('/workspace/work_ink/gguf/Inkling-2-Mini-UD-IQ2_XXS.gguf'); run('UD-IQ2_XXS')
Q.restore()
g_l0 = E.boot_ratio(res[best]['per_window'], res['L0']['per_window']) if best != 'L0' else None
g_iq2 = E.boot_ratio(res[best]['per_window'], res['UD-IQ2_XXS']['per_window'])
g_l0iq2 = E.boot_ratio(res['L0']['per_window'], res['UD-IQ2_XXS']['per_window'])
replace = bool(g_l0 and g_l0['ci95'][1] < 0)
out = dict(selected=best, pi2=res, gain_selected_vs_L0=g_l0, gain_selected_vs_iq2=g_iq2, gain_L0_vs_iq2=g_l0iq2, replace_formic_v15=replace)
json.dump(out, open(f'{RES}/phase2_pi2.json', 'w'), indent=1)
print('gain', best, 'contre L0 :', g_l0, '; contre UD-IQ2_XXS :', g_iq2, '; L0 contre UD-IQ2_XXS :', g_l0iq2, '; remplacement :', replace, flush=True)
E.S.update(saved)
# ------------------------------------------------ remplacement du paquet (si la règle l'autorise)
if replace:
    pkg = f'{Q.PKG_ROOT}/{best}'
    vis = PK.package_vision(pkg, 5.0); mtp = PK.package_mtp(pkg, 4.0); aux = PK.package_aux(pkg)
    lp = PK.load_package(pkg, blank=True); per = E.evaluate()
    dev = E.summarize([w for w in per if w['part'] == 'dev'], 'P-I-dev'); conf = E.summarize([w for w in per if w['part'] == 'conf'], 'P-I-conf (non aveugle en phase 2)')
    mem = json.load(open(f'{RES}/pi_dev_{best}.json'))['kl']
    comp = {q: json.load(open(f'{SE}/pi_competitor_{q}.json')) for q in ('UD-IQ2_XXS', 'UD-Q2_K_XL', 'UD-IQ3_XXS')}
    gconf = {q: E.boot_ratio(conf['per_window'], c['conf']['per_window']) for q, c in comp.items()}
    vr = subprocess.run([sys.executable, '-I', '-B', Q.CODEC + '/package.py', 'verify', pkg], capture_output=True, text=True)
    disk = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(pkg) for f in fs)
    def safe(f):
        try: return f()
        except Exception as e: traceback.print_exc(); return dict(error=repr(e)[:300])
    E.vision_teacher(); visf = safe(E.vision_eval); E.pw_teacher(); pw = safe(E.pw_eval)
    man = json.load(open(f'{pkg}/manifest.json'))
    json.dump(dict(package=pkg, total_bytes_on_disk_all_files=disk, verify_rc=vr.returncode, verify_stdout=vr.stdout[-500:], entries=man),
              open(f'{RES}/package_manifest_ink_phase2.json', 'w'), indent=0)
    fin = dict(selected=best, package=pkg, package_bytes_on_disk=disk, within_8_9GB=8e9 <= disk <= 9e9, verify_rc=vr.returncode, rebuilt=lp,
               dev_from_files=dev['kl'], dev_in_memory=mem, identical_6_decimals=round(dev['kl'], 6) == round(mem, 6), dev=dev, conf_non_blind=conf,
               gain_conf_vs=gconf, vision=dict(package=vis, full_package=visf), drafter=dict(mtp, evaluated=False), pw=pw)
    json.dump(fin, open(f'{RES}/final_selected_phase2.json', 'w'), indent=1)
    print('NOUVEAU PAQUET', best, disk, 'verify', vr.returncode, '; dev fichiers/mémoire', dev['kl'], mem, '; conf', E.short(conf), gconf, flush=True)
git('formic_inkling: phase 2 — confirmation sur P-I2' + (f', paquet Formic V1.5 remplacé par {best}' if replace else ', Formic V1.5 reste L0'))
