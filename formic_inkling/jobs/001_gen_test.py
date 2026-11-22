import json, time, torch
P = [json.loads(l) for l in open('/workspace/work_ink/data/pi_prompts.jsonl')]
sel = [P[0], P[45], P[90], P[150]]
def render(p): return tok.apply_chat_template(p['messages'], tools=p['tools'], tokenize=False, add_generation_prompt=True, reasoning_effort='high') if p['tools'] else tok.apply_chat_template(p['messages'], tokenize=False, add_generation_prompt=True, reasoning_effort='high')
texts = [render(p) + '<|message_model|>' for p in sel]
print(repr(texts[2][-300:]))
eos = tok.convert_tokens_to_ids('<|content_model_end_sampling|>'); print('eos', eos, model.generation_config.eos_token_id)
singles = []
for t in texts:
    x = tok(t, return_tensors='pt', add_special_tokens=False).to('cuda')
    o = model.generate(**x, max_new_tokens=24, do_sample=False)
    singles.append(o[0, x['input_ids'].shape[1]:].tolist())
tok.padding_side = 'left'
x = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
t0 = time.time(); o = model.generate(**x, max_new_tokens=24, do_sample=False); print('batch gen time', time.time() - t0)
for k in range(4):
    b = o[k, x['input_ids'].shape[1]:].tolist()
    print(k, b == singles[k], repr(tok.decode(singles[k])), '|', repr(tok.decode(b)))
