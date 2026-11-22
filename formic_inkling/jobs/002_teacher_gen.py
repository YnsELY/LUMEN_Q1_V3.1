"""Réponses du professeur pour les domaines à invite de P-I (agentique, raisonnement, instructions).
Décodage glouton, effort de raisonnement 'high' (0,9, défaut du gabarit), lots avec remplissage à gauche (vérifié identique
au décodage individuel, job 001). Sortie : /workspace/work_ink/data/pi_teacher_gen.jsonl"""
import json, time, torch
P = [json.loads(l) for l in open('/workspace/work_ink/data/pi_prompts.jsonl')]
def render(p):
    kw = dict(tools=p['tools']) if p['tools'] else {}
    return tok.apply_chat_template(p['messages'], tokenize=False, reasoning_effort='high', **kw) + '<|message_model|>'
CFG = dict(reasoning=(32, 1024), instructions=(32, 768), agentic=(48, 256))
EOS = tok.convert_tokens_to_ids('<|content_model_end_sampling|>')
tok.padding_side = 'left'
out = open('/workspace/work_ink/data/pi_teacher_gen.jsonl', 'w')
for dom, (n, mx) in CFG.items():
    cand = [p for p in P if p['domain'] == dom]
    sel = []
    for p in cand:
        t = render(p); L = len(tok(t, add_special_tokens=False)['input_ids'])
        if L <= 1500: sel.append((p, t))
        if len(sel) >= n: break
    t0 = time.time()
    x = tok([t for _, t in sel], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    o = model.generate(**x, max_new_tokens=mx, do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
    for k, (p, t) in enumerate(sel):
        g = o[k, x['input_ids'].shape[1]:].tolist()
        if EOS in g: g = g[:g.index(EOS) + 1]
        out.write(json.dumps(dict(domain=dom, source_id=p['source_id'], prompt=t, gen_ids=g, gen_text=tok.decode(g), finished=EOS in g), ensure_ascii=False) + '\n')
    out.flush()
    print(dom, len(sel), 'prompts, max_new', mx, 'time %.0fs' % (time.time() - t0), 'peak %.1f GB' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    del x, o; torch.cuda.empty_cache()
