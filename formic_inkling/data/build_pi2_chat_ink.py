"""Amendement P-I2 (avant tout résultat de phase 2) : + 6 fenêtres agentiques (glaive-function-calling-v2 j∈[5000,15000), conversations
complètes, réponses du jeu de données) et + 6 fenêtres d'instructions (UltraChat test_sft, lignes >= 2000), rendues au gabarit d'Inkling."""
import os, sys, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets import load_dataset
import build_data_ink as B
used = set(json.loads(l)['source_id'] for l in open(f'{B.D}/pi_prompts.jsonl'))
used |= set(s for d in json.load(open(f'{B.D}/cal_ink_meta.json'))['docs'] for s in d['source_id'].split('|'))
used |= set(p['source_id'] for p in json.load(open(f'{B.D}/pd32_prompts.json')))
def pack(dom, units, n):
    made = 0; cur = [B.BOT]; src = []
    for sid, t in units:
        if made >= n: break
        if sid in used: continue
        cur += B.ids(t); src.append(sid)
        if len(cur) >= 4096:
            B.PI.append(dict(domain=dom, source_id='|'.join(src), ids=cur[:4096])); made += 1; cur = [B.BOT]; src = []
    print(dom, made, flush=True)
def glaive():
    for j, r in enumerate(load_dataset('glaiveai/glaive-function-calling-v2', split='train', streaming=True)):
        if j < 5000: continue
        if j >= 15000: break
        try:
            msgs, tools = B.glaive_conv(r); yield f'glaive/{j}', B.chat(msgs, tools=tools)
        except Exception: pass
def ultra():
    for j, r in enumerate(load_dataset('HuggingFaceH4/ultrachat_200k', split='test_sft', streaming=True)):
        if j < 2000: continue
        try: yield f'ultrachat/test_sft/{r["prompt_id"]}', B.chat(r['messages'])
        except Exception: pass
pack('agentic', glaive(), 6); pack('instructions', ultra(), 6)
d = torch.load(f'{B.D}/pi2_tokens.pt')
X = torch.cat([d['tokens'], torch.tensor([r['ids'] for r in B.PI], dtype=torch.int32)])
dom = d['domain'] + [r['domain'] for r in B.PI]; sids = d['source_ids'] + [r['source_id'].split('|') for r in B.PI]
torch.save(dict(tokens=X, domain=dom, part=['conf2'] * len(dom), source_ids=sids), f'{B.D}/pi2_tokens.pt')
P = '/workspace/LUMEN_Q1_V3.1/formic_inkling/results/protocol_pi2.json'; m = json.load(open(P))
m.update(sha256=hashlib.sha256(X.numpy().tobytes()).hexdigest(), windows=len(dom), domains={k: dom.count(k) for k in dict.fromkeys(dom)},
         amendment='+6 agentique (glaive, réponses du jeu de données) +6 instructions (UltraChat test_sft) ajoutées avant tout résultat de phase 2',
         sources_chat=[r['source_id'] for r in B.PI])
json.dump(m, open(P, 'w'), indent=1); print('P-I2', X.shape, m['domains'])
