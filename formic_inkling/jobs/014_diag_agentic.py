"""Diagnostic agentique (après la phase 2, sans sélection) : KL par type de segment sur les fenêtres agentiques et de raisonnement de
P-I, pour Formic V4 et UD-IQ2_XXS, chacun complet puis « experts seuls » (attention, embed et unembed remis au professeur)."""
import sys, json, time, torch, collections
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E, ink_quant as Q, ink_package as PK, ink_gguf as GG
for m in (E, Q, PK, GG): importlib.reload(m)
E.init(model, tok); E.load_protocol(); E.teacher_cache(); Q.init(model)
A = E.S['pi']; rows = [i for i, d in enumerate(A['dom']) if d in ('agentic', 'reasoning')]
SP = {n: tok.convert_tokens_to_ids(n) for n in ('<|message_system|>', '<|message_user|>', '<|message_model|>', '<|message_tool|>', '<|content_text|>',
      '<|content_thinking|>', '<|content_invoke_tool_json|>', '<|content_xml|>', '<|end_message|>', '<|content_model_end_sampling|>', '<|begin_of_text|>')}
SPIDS = set(v for v in SP.values() if isinstance(v, int))
def labels(ids):
    """étiquette de segment pour chaque position (le jeton prédit est ids[p+1] ; on étiquette par la position du jeton cible)."""
    lab = []; role = 'other'; content = 'other'
    for t in ids:
        if t == SP['<|message_system|>']: role = 'system'
        elif t == SP['<|message_user|>']: role = 'user'
        elif t == SP['<|message_model|>']: role = 'model'
        elif t == SP['<|message_tool|>']: role = 'tool'
        if t == SP['<|content_thinking|>']: content = 'thinking'
        elif t == SP['<|content_invoke_tool_json|>']: content = 'tool_call'
        elif t == SP['<|content_xml|>']: content = 'tool_declare'
        elif t == SP['<|content_text|>']: content = 'text'
        if t in SPIDS: lab.append('special_token')
        elif content == 'tool_declare': lab.append('tool_declare_json')
        elif role == 'tool': lab.append('tool_response')
        elif role == 'model' and content == 'tool_call': lab.append('model_tool_call')
        elif role == 'model' and content == 'thinking': lab.append('model_thinking')
        elif role == 'model': lab.append('model_text')
        elif role == 'user': lab.append('user_text')
        elif role == 'system': lab.append('system_text')
        else: lab.append('other')
    return lab
LAB = {}
for i in rows:
    ids = A['tok'][i].tolist(); lb = labels(ids)
    LAB[i] = [lb[p + 1] if p + 1 < len(ids) else 'other' for p in range(E.P0, E.P1)]   # cible de la position p = jeton p+1
def agg(per):
    out = collections.defaultdict(lambda: [0.0, 0]); dom = collections.defaultdict(lambda: [0.0, 0])
    for w in per:
        k = w['kl_vec'].float()
        for p, l in enumerate(LAB[w['row']]):
            o = out[(w['domain'], l)]; o[0] += float(k[p]); o[1] += 1
        d = dom[w['domain']]; d[0] += float(k.sum()); d[1] += k.numel()
    return {f'{a}/{b}': dict(kl=v[0] / v[1], n=v[1]) for (a, b), v in sorted(out.items())}, {a: v[0] / v[1] for a, v in dom.items()}
ATT = lambda n: ('self_attn' in n and 'experts' not in n) or 'embed_tokens.weight' in n or n == 'lm_head.weight'
def restore_attn():
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in Q.S['orig'] and ATT(n): p.copy_(Q.S['orig'][n].to(p.device))
res = {}
def run(name):
    t = time.time(); per = E.evaluate(rows=rows, route=False); res[name] = agg(per); print(name, {k: round(v, 4) for k, v in res[name][1].items()}, '%.0fs' % (time.time() - t), flush=True)
Q.restore(); PK.load_package('/workspace/work_ink/packages/V4', blank=False); run('V4'); restore_attn(); run('V4_experts_only')
GG.load_gguf('/workspace/work_ink/gguf/Inkling-2-Mini-UD-IQ2_XXS.gguf'); run('IQ2'); restore_attn(); run('IQ2_experts_only')
Q.restore()
json.dump(dict(rows=rows, results={k: dict(by_segment=v[0], by_domain=v[1]) for k, v in res.items()}), open(E.RES + '/diag_agentic.json', 'w'), indent=1)
print('\nsegment                               V4      V4exp    IQ2     IQ2exp     n')
for key in res['V4'][0]:
    if res['V4'][0][key]['n'] < 300: continue
    print('%-36s %.4f  %.4f  %.4f  %.4f  %6d' % (key, res['V4'][0][key]['kl'], res['V4_experts_only'][0][key]['kl'], res['IQ2'][0][key]['kl'], res['IQ2_experts_only'][0][key]['kl'], res['V4'][0][key]['n']))
