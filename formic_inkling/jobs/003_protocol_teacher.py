"""C2 : assemblage et gel du protocole P-I (fenêtres de 4096, 9 domaines), contrôle de disjonction avec la calibration,
puis cache des états finaux du professeur aux positions scorées (2048–4095) et de son routage."""
import json, hashlib, sys, re, torch
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E; importlib.reload(E)
D = '/workspace/work_ink/data'; RES = '/workspace/LUMEN_Q1_V3.1/formic_inkling/results'
BOT = tok.convert_tokens_to_ids('<|begin_of_text|>'); W = 4096
raw = [json.loads(l) for l in open(f'{D}/pi_raw.jsonl')]
gen = [json.loads(l) for l in open(f'{D}/pi_teacher_gen.jsonl')]
rows = []
for r in raw: rows.append(dict(domain=r['domain'], ids=r['ids'], sources=[r['source_id']]))
NPI = 12
for dom in ('agentic', 'reasoning', 'instructions'):
    convs = [g for g in gen if g['domain'] == dom]
    seqs = [(g['source_id'], tok(g['prompt'], add_special_tokens=False)['input_ids'] + g['gen_ids']) for g in convs]
    k = 0; made = 0
    while made < NPI and k < len(seqs):
        ids = [BOT]; src = []
        while len(ids) < W and k < len(seqs):
            ids += seqs[k][1]; src.append(seqs[k][0]); k += 1
        if len(ids) < W: break
        rows.append(dict(domain=dom, ids=ids[:W], sources=src)); made += 1
    print(dom, 'windows', made, 'conversations used', k, 'of', len(seqs))
order = ['agentic', 'code', 'math', 'reasoning', 'encyclopedic', 'multilingual', 'long', 'instructions', 'web']
rows = sorted(rows, key=lambda r: order.index(r['domain']))
part = []; seen = {}
for r in rows:
    part.append('dev' if seen.get(r['domain'], 0) % 2 == 0 else 'conf'); seen[r['domain']] = seen.get(r['domain'], 0) + 1
X = torch.tensor([r['ids'] for r in rows], dtype=torch.int32)
assert X.shape[1] == W
h = hashlib.sha256(X.numpy().tobytes() + json.dumps([[r['domain'] for r in rows], part]).encode()).hexdigest()
torch.save(dict(tokens=X, domain=[r['domain'] for r in rows], part=part, source_ids=[r['sources'] for r in rows]), f'{D}/pi_tokens.pt')
# disjonction avec la calibration (identifiants)
cal = json.load(open(f'{D}/cal_ink_meta.json'))
cal_ids = set()
for d in cal['docs']:
    for s in d['source_id'].split('|'):
        if not s.startswith('+'): cal_ids.add(s)
pi_ids = set(s for r in rows for s in r['sources'])
inter = sorted(pi_ids & cal_ids)
prot = dict(name='P-I', frozen_at='2026-11-22', sha256=h, windows=len(rows), window_len=W, scored_positions='2048-4095 (2048 par fenêtre)',
            positions_scored=len(rows) * 2048, domains={d: sum(r['domain'] == d for r in rows) for d in order},
            halves={p: part.count(p) for p in ('dev', 'conf')}, split_rule='alternance par domaine (comme arki_teacher) : 1re, 3e, ... fenêtre = dev',
            first_token='<|begin_of_text|> en tête de chaque fenêtre', teacher_generation=dict(decoding='glouton', reasoning_effort='high (0,9)',
            max_new_tokens=dict(reasoning=1024, instructions=768, agentic=256), batch='remplissage à gauche, vérifié identique au décodage individuel (job 001)'),
            vocab='200 058 entrées utiles (unembed non rembourré dans le graphe)', logits='30·tanh(z/30)', kl='KL(professeur || candidat), nats, moyenne par token',
            disjointness=dict(id_intersection_with_calibration=inter, n_pi_source_ids=len(pi_ids), n_cal_source_ids=len(cal_ids),
                              filter_13gram='appliqué à la construction de la calibration (build_data_ink.py ; bloc tool_declare exclu)'),
            deviations=['216 fenêtres prévues ; 108 retenues (12 par domaine) pour tenir le budget de temps du pod',
                        'llama.cpp abandonné (instruction opérateur) : pas de vérification llama-tokenize'],
            sources=[dict(domain=r['domain'], part=p, sources=r['sources'][:6] + (['+%d' % (len(r['sources']) - 6)] if len(r['sources']) > 6 else [])) for r, p in zip(rows, part)])
json.dump(prot, open(f'{RES}/protocol_ink.json', 'w'), indent=1, ensure_ascii=False)
print('P-I', X.shape, prot['domains'], prot['halves'], 'sha', h[:16], 'id intersection', len(inter))
E.init(model, tok); E.load_protocol()
print(E.teacher_cache())
