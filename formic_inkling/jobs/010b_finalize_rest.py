"""C6 (reprise après correctif de load_package et du débit Vision/drafter) : sélection relue, paquet complet (texte + Vision 5 bpw + drafter 4 bpw + annexes), verify, reconstruction
DEPUIS LES FICHIERS, lecture de la moitié conf et des mesures secondaires, Vision."""
import sys, os, json, time, subprocess, torch
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q, ink_package as PK
for m in (E, Q, PK): importlib.reload(m)
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model); Q.restore()
RES = E.RES; SEALED = E.SEALED
# ---------------------------------------------------------------- sélection : déjà gelée (results/selection.lock.json, commit 8f0adfa)
sel = json.load(open(f'{RES}/selection.lock.json')); best = sel['selected']
print('sélection gelée relue :', best, flush=True)
# ---------------------------------------------------------------- Vision : professeur, puis tour du paquet
pkg = f'{Q.PKG_ROOT}/{best}'
import traceback
def safe(f, what):
    try: return f()
    except Exception as e:
        traceback.print_exc(); print('ÉCHEC', what, repr(e)[:300], flush=True); return dict(error=repr(e)[:500])
vt = safe(E.vision_teacher, 'vision_teacher'); print('vision professeur :', vt, flush=True)
vis = PK.package_vision(pkg, 5.0); print('vision paquet', vis, flush=True)
vis_only = safe(E.vision_eval, 'vision_eval tour seule') if not isinstance(vt, dict) else dict(skipped='vision_teacher a échoué'); print('vision (tour du paquet, texte professeur) :', vis_only, flush=True)
mtp = PK.package_mtp(pkg, 4.0); print('drafter', mtp, flush=True)
aux = PK.package_aux(pkg)
# ---------------------------------------------------------------- reconstruction depuis les fichiers (processus de décodage séparé)
mem_dev = json.load(open(f'{RES}/pi_dev_{best}.json'))
t0 = time.time(); lp = PK.load_package(pkg, blank=True); print('reconstruit depuis les fichiers :', lp, flush=True)
per = E.evaluate(); tre = time.time() - t0
dev = [w for w in per if w['part'] == 'dev']; conf = [w for w in per if w['part'] == 'conf']
rdev = E.summarize(dev, 'P-I-dev'); rconf = E.summarize(conf, 'P-I-conf'); rfull = E.summarize(per, 'P-I')
same6 = round(rdev['kl'], 6) == round(mem_dev['kl'], 6)
print('KL dev depuis les fichiers %.6f / en mémoire %.6f -> identiques à 6 décimales : %s' % (rdev['kl'], mem_dev['kl'], same6), flush=True)
vis_full = safe(E.vision_eval, 'vision_eval paquet complet') if not isinstance(vt, dict) else dict(skipped='vision_teacher a échoué'); print('vision (paquet complet) :', vis_full, flush=True)
sec_sel = None
if os.path.exists(f'{E.D}/pd32_teacher.json'):
    E.pw_teacher(); E.pd32_teacher(); sec_sel = dict(pw=safe(E.pw_eval, 'P-W'), pd32=safe(E.pd32_eval, 'P-D32'))
# ---------------------------------------------------------------- confirmation : concurrents (scellés) et gain apparié
comp = {}
for q in ('UD-IQ2_XXS', 'UD-Q2_K_XL', 'UD-IQ3_XXS'):
    s = json.load(open(f'{SEALED}/pi_competitor_{q}.json'))
    comp[q] = dict(conf=s['conf'], full=s['full'], secondary=json.load(open(f'{SEALED}/secondary_{q}.json')) if os.path.exists(f'{SEALED}/secondary_{q}.json') else None,
                   gain_conf=E.boot_ratio(rconf['per_window'], s['conf']['per_window']),
                   gain_full=E.boot_ratio(rfull['per_window'], s['full']['per_window']))
g = comp['UD-IQ2_XXS']['gain_conf']
level = 'échec' if (g['gain'] >= 0 or g['ci95'][1] >= 0) else ('renforcé' if g['gain'] <= -0.25 and g['ci95'][1] < 0 else ('atteint' if g['gain'] <= -0.10 and g['ci95'][1] < 0 else 'gain < 10 % (ni échec ni atteint)'))
# ---------------------------------------------------------------- paquet : verify, octets
vr = subprocess.run([sys.executable, '-I', '-B', Q.CODEC + '/package.py', 'verify', pkg], capture_output=True, text=True)
man = json.load(open(f'{pkg}/manifest.json'))
def blk(k):
    if k.startswith(('visual.', 'raw.visual.')): return 'vision'
    if k.startswith(('mtp.', 'raw.mtp.')): return 'drafter'
    if k in ('llm.embed', 'llm.unembed'): return 'vocabulaire'
    if '.mlp.experts.' in k:
        i = int(k.split('.')[2]); return 'experts_prelude' if i < 6 else ('experts_coeur' if i < 18 else 'experts_coda')
    if k.startswith('llm.layers.'): return 'attention'
    return 'petits_tenseurs_texte'
by = {}
for k, v in man.items():
    b = by.setdefault(blk(k), dict(bytes=0, params=0, files=0)); b['bytes'] += v['bytes']; b['files'] += 1
    b['params'] += (v['shape'][0] * v['shape'][1]) if v['kind'] == 'e8' else 0
for b in by.values(): b['bpw_e8'] = 8 * b['bytes'] / b['params'] if b['params'] else None
total = Q.pkg_bytes(pkg); disk = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(pkg) for f in fs)
json.dump(dict(package=pkg, total_bytes_manifest_entries=total, total_bytes_on_disk_all_files=disk, verify_stdout=vr.stdout[-2000:], verify_rc=vr.returncode,
               by_block=by, entries=man), open(f'{RES}/package_manifest_ink.json', 'w'), indent=0)
final = dict(selected=best, package=pkg, package_bytes_on_disk=disk, within_8_9GB=8e9 <= disk <= 9e9, rebuilt_from_files=lp, rebuild_eval_seconds=tre,
             dev_from_files=rdev, dev_in_memory_kl=mem_dev['kl'], identical_6_decimals=same6, conf=rconf, full=rfull,
             competitors=comp, success_level_vs_UD_IQ2_XXS_conf=level, secondary_selected=sec_sel,
             vision=dict(package=vis, tower_only=vis_only, full_package=vis_full), drafter=dict(mtp, evaluated=False), aux_bytes=aux,
             verify_rc=vr.returncode, by_block=by)
json.dump(final, open(f'{RES}/final_selected.json', 'w'), indent=1)
print('CONF :', E.short(rconf)); [print(q, E.short(c['conf']), c['gain_conf']) for q, c in comp.items()]
print('niveau :', level, '; paquet', disk, 'octets ; verify rc', vr.returncode, vr.stdout[-300:])
