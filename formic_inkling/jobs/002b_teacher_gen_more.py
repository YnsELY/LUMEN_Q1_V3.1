"""Suite de 002 (OOM au pré-remplissage du lot agentique de 48) : invites restantes, lots de 16. Même réglage (glouton, effort high)."""
import json, time, torch
P = [json.loads(l) for l in open('/workspace/work_ink/data/pi_prompts.jsonl')]
done = set(json.loads(l)['source_id'] for l in open('/workspace/work_ink/data/pi_teacher_gen.jsonl'))
def render(p):
    kw = dict(tools=p['tools']) if p['tools'] else {}
    return tok.apply_chat_template(p['messages'], tokenize=False, reasoning_effort='high', **kw) + '<|message_model|>'
CFG = dict(reasoning=(16, 1024), instructions=(32, 768), agentic=(64, 256))
EOS = tok.convert_tokens_to_ids('<|content_model_end_sampling|>')
tok.padding_side = 'left'
out = open('/workspace/work_ink/data/pi_teacher_gen.jsonl', 'a')
for dom, (n, mx) in CFG.items():
    sel = []
    for p in P:
        if p['domain'] != dom or p['source_id'] in done: continue
        t = render(p); L = len(tok(t, add_special_tokens=False)['input_ids'])
        if L <= 1500: sel.append((p, t))
        if len(sel) >= n: break
    for b in range(0, len(sel), 16):
        bb = sel[b:b + 16]; t0 = time.time()
        x = tok([t for _, t in bb], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        o = model.generate(**x, max_new_tokens=mx, do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
        for k, (p, t) in enumerate(bb):
            g = o[k, x['input_ids'].shape[1]:].tolist()
            if EOS in g: g = g[:g.index(EOS) + 1]
            out.write(json.dumps(dict(domain=dom, source_id=p['source_id'], prompt=t, gen_ids=g, gen_text=tok.decode(g), finished=EOS in g), ensure_ascii=False) + '\n')
        out.flush(); del x, o; torch.cuda.empty_cache()
        print(dom, 'batch', b, len(bb), 'time %.0fs' % (time.time() - t0), flush=True)
