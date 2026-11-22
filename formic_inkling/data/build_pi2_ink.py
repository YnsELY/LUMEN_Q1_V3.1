"""P-I2 : jeu de confirmation scellé de la phase 2 (construit après le gel de phase 1, avant tout résultat de phase 2).
36 fenêtres de 4096 (6 domaines bruts x 6), plages disjointes de P-I, P-W, P-D32 et de la calibration ; contrôle par identifiant."""
import os, sys, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets import load_dataset
import build_data_ink as B
N = 6
used = set(json.loads(l)['source_id'] for l in open(f'{B.D}/pi_raw.jsonl'))
used |= set(s for d in json.load(open(f'{B.D}/cal_ink_meta.json'))['docs'] for s in d['source_id'].split('|'))
used |= set(p['source_id'] for p in json.load(open(f'{B.D}/pd32_prompts.json')))
def add(dom, sid, t, n_have):
    if sid in used: return False
    return B.raw_window(dom, sid, t, '', '', '')
def take(dom, it, f):
    n = 0
    for sid, t in it:
        if n >= N: break
        if f(t) and add(dom, sid, t, n): n += 1
    print(dom, n, flush=True)
take('code', ((f'codeparrot-clean/{100000 + j}', r['content']) for j, r in enumerate(load_dataset('codeparrot/codeparrot-clean', split='train', streaming=True).skip(100000))), lambda t: len(t) > 15000 and sum(1 for r in B.PI if r['domain'] == 'code') < 3)
take('code', ((f'the-stack-rust-clean/{10000 + j}', r['content']) for j, r in enumerate(load_dataset('ammarnasr/the-stack-rust-clean', split='train', streaming=True).skip(10000))), lambda t: len(t) > 15000 and sum(1 for r in B.PI if r['domain'] == 'code') < 6)
buf = []
for sub in ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'):
    for j, r in enumerate(load_dataset('EleutherAI/hendrycks_math', sub, split='test', streaming=True)):
        if j < 260: continue
        if j >= 420: break
        buf.append(f"Problem: {r['problem']}\nSolution: {r['solution']}\n\n")
k = 0; n = 0
while n < N and k < len(buf):
    t = ''; s0 = k
    while len(B.ids(t)) < 4096 and k < len(buf): t += buf[k]; k += 1
    if B.raw_window('math', f'hendrycks_math/test/pi2pack{s0}-{k}', t, '', '', ''): n += 1
print('math', n)
take('encyclopedic', ((f'wikipedia/20231101.en/{r["id"]}', r['text']) for r in load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True).skip(400000)), lambda t: len(t) > 9000)
for lg in ('fr', 'de', 'es', 'ru', 'zh', 'ja'):
    n0 = sum(1 for r in B.PI if r['domain'] == 'multilingual')
    for r in load_dataset('wikimedia/wikipedia', f'20231101.{lg}', split='train', streaming=True).skip(300000):
        if len(r['text']) > 9000 and add('multilingual', f'wikipedia/20231101.{lg}/{r["id"]}', r['text'], 0): break
print('multilingual', sum(1 for r in B.PI if r['domain'] == 'multilingual'))
take('long', ((f'pg19/validation/{j}', r['text'][20000:]) for j, r in enumerate(load_dataset('emozilla/pg19', split='validation', streaming=True))), lambda t: len(t) > 40000)
take('web', ((f'c4/en/validation/{50000 + j}', r['text']) for j, r in enumerate(load_dataset('allenai/c4', 'en', split='validation', streaming=True).skip(50000))), lambda t: len(t) > 14000)
X = torch.tensor([r['ids'] for r in B.PI], dtype=torch.int32)
h = hashlib.sha256(X.numpy().tobytes()).hexdigest()
torch.save(dict(tokens=X, domain=[r['domain'] for r in B.PI], part=['conf2'] * len(B.PI), source_ids=[[r['source_id']] for r in B.PI]), f'{B.D}/pi2_tokens.pt')
json.dump(dict(name='P-I2', sha256=h, windows=len(B.PI), domains={d: sum(r['domain'] == d for r in B.PI) for d in ('code', 'math', 'encyclopedic', 'multilingual', 'long', 'web')},
               sources=[r['source_id'] for r in B.PI], id_intersection_with_PI_cal_PD32=[r['source_id'] for r in B.PI if r['source_id'] in used]),
          open('/workspace/LUMEN_Q1_V3.1/formic_inkling/results/protocol_pi2.json', 'w'), indent=1)
print('P-I2', X.shape, h[:16])
