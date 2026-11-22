"""Mesures secondaires (recoupement, jamais de sélection) :
P-W   : Wikitext-2 raw test, fenêtres de 512 tokens, 40 fenêtres (comme ggml-org), positions scorées 256–511 (comme llama-perplexity).
P-D32 : 300 invites hors calibration et hors P-I (100 agentiques ToolACE j∈[1500,2000) / glaive j∈[15000,20000),
        100 mathématiques MATH test j>=220, 100 non latines Wikipedia ru/zh/ja/ar/hi/ko), décodage glouton de 32 tokens."""
import os, sys, json, re
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from datasets import load_dataset
import build_data_ink as B
tok = B.tok; D = B.D
# P-W
t = '\n'.join(r['text'] for r in load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test'))
ids = B.ids(t)
X = torch.tensor([ids[i * 512:(i + 1) * 512] for i in range(40)], dtype=torch.int32)
torch.save(dict(tokens=X), f'{D}/pw_tokens.pt'); print('P-W', X.shape)
# P-D32
P = []
for j, r in enumerate(load_dataset('Team-ACE/ToolACE', split='train', streaming=True)):
    if j < 1500: continue
    if j >= 2000 or sum(p['domain'] == 'agentic' for p in P) >= 50: break
    try:
        msgs, tools = B.toolace_conv(r); last = max(i for i, m in enumerate(msgs) if m['role'] == 'assistant'); ctx = msgs[:last]
        if not ctx or ctx[-1]['role'] not in ('user', 'tool'): continue
        P.append(dict(domain='agentic', source_id=f'toolace/{j}', text=B.chat(ctx, tools=tools, reasoning_effort='none') + '<|message_model|>'))
    except Exception: pass
for j, r in enumerate(load_dataset('glaiveai/glaive-function-calling-v2', split='train', streaming=True)):
    if j < 15000: continue
    if j >= 20000 or sum(p['domain'] == 'agentic' for p in P) >= 100: break
    try:
        msgs, tools = B.glaive_conv(r); last = max(i for i, m in enumerate(msgs) if m['role'] == 'assistant'); ctx = msgs[:last]
        if not ctx or ctx[-1]['role'] not in ('user', 'tool'): continue
        P.append(dict(domain='agentic', source_id=f'glaive/{j}', text=B.chat(ctx, tools=tools, reasoning_effort='none') + '<|message_model|>'))
    except Exception: pass
n = 0
for sub in ('algebra', 'number_theory', 'geometry', 'precalculus'):
    for j, r in enumerate(load_dataset('EleutherAI/hendrycks_math', sub, split='test', streaming=True)):
        if j < 220: continue
        if j >= 245: break
        P.append(dict(domain='math', source_id=f'hendrycks_math/{sub}/test/{j}', text=B.chat([{'role': 'user', 'content': r['problem']}], reasoning_effort='none') + '<|message_model|>'))
for lg in ('ru', 'zh', 'ja', 'ar', 'hi', 'ko'):
    k = 0
    for r in load_dataset('wikimedia/wikipedia', f'20231101.{lg}', split='train', streaming=True).skip(200000):
        if k >= (17 if lg != 'ko' else 15): break
        if len(r['text']) < 800: continue
        P.append(dict(domain='non_latin', source_id=f'wikipedia/20231101.{lg}/{r["id"]}', text='<|begin_of_text|>' + r['text'][:400])); k += 1
print('P-D32', len(P), {d: sum(p['domain'] == d for p in P) for d in ('agentic', 'math', 'non_latin')})
json.dump(P, open(f'{D}/pd32_prompts.json', 'w'), ensure_ascii=False)
