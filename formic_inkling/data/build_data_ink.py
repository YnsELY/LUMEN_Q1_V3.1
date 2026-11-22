"""formic_inkling — construction du protocole P-I (textes bruts + invites) et de la calibration (cal_ink).
Dérivé de formic_arch/data/build_corpus.py et formic_v7/data/build_cal2.py (sources reprises, tokenizer et gabarit
d'Inkling). Disjonction : plages d'indices disjointes par source (P-I d'abord, calibration ensuite) + filtre de 13 mots
communs de build_cal2.py (calibration contre P-I, gabarit vide exclu).
Sorties dans /workspace/work_ink/data/ :
  pi_raw.jsonl       fenêtres brutes déjà tokenisées (domaines code, math, encyclopédique, multilingue, long, web)
  pi_prompts.jsonl   invites des domaines agentique / raisonnement / instructions (réponses générées ensuite par le professeur)
  cal_ink.pt         calibration [N, 2048] int32 (ordre entrelacé : tout préfixe garde la composition)
  cal_ink_meta.json  composition, identifiants de sources
Lancer : python -B build_data_ink.py
"""
import os, sys, re, json, ast, random, hashlib, glob, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from concurrent.futures import ThreadPoolExecutor
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 1357; TOK = '/root/models/inkling-2-mini'; D = '/workspace/work_ink/data'; os.makedirs(D, exist_ok=True)
L = 2048; W = 4096
tok = AutoTokenizer.from_pretrained(TOK)
BOT = tok.convert_tokens_to_ids('<|begin_of_text|>'); assert isinstance(BOT, int) and BOT < 201024
def ids(t): return tok(t, add_special_tokens=False)['input_ids']
def stream(*a, **k): return load_dataset(*a, streaming=True, **k)
def log(*a): print(*a, flush=True)
def chat(msgs, **kw): return tok.apply_chat_template(msgs, tokenize=False, **kw)

N_PI = int(os.environ.get('N_PI', 12))           # fenêtres par domaine
N_CAL = int(os.environ.get('N_CAL', 1024))       # documents de 2048
CAL_COMP = dict(chat=0.35, code=0.20, prose=0.15, multi=0.15, math=0.10, web=0.05)

# ------------------------------------------------------------------ helpers
def words(t): return re.findall(r'\w+', t.lower())
def grams(t, n=13):
    w = words(t); return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}

PI = []          # dict(domain, source_id, ids(4096) | None, text)
PROMPTS = []     # dict(domain, source_id, messages, tools)

def raw_window(domain, sid, text, lang, ds, lic):
    i = ids(text)
    if len(i) < W - 1: return False
    w = [BOT] + i[:W - 1]
    PI.append(dict(domain=domain, source_id=sid, lang=lang, dataset=ds, license=lic, ids=w, text=tok.decode(w[1:])))
    return True

# ================================================================== P-I : domaines bruts
def pi_code():
    per = N_PI // 4; got = {'py': 0, 'c': 0, 'js': 0, 'rs': 0}
    ds = stream('codeparrot/codeparrot-clean', split='train')
    for j, r in enumerate(ds):
        if got['py'] >= per: break
        if len(r['content']) < 15000: continue
        if raw_window('code', f'codeparrot-clean/{j}', r['content'], 'python', 'codeparrot/codeparrot-clean', r['license']): got['py'] += 1
    ds = stream('ammarnasr/the-stack-rust-clean', split='train')
    for j, r in enumerate(ds):
        if got['rs'] >= per: break
        if len(r['content']) < 15000: continue
        if raw_window('code', f'the-stack-rust-clean/{j}', r['content'], 'rust', 'ammarnasr/the-stack-rust-clean', 'the-stack (permissive)'): got['rs'] += 1
    # C/C++ et JS : fichiers locaux de l'image (moitié paire du hash du chemin pour P-I, impaire pour la calibration)
    for ext, key, lic in ((('.c', '.cpp', '.h', '.hpp', '.cc'), 'c', 'local (licences des paquets système)'), (('.js',), 'js', 'local (licences des paquets)')):
        files = sorted(f for e in ext for f in glob.glob(f'/usr/**/*{e}', recursive=True) if os.path.getsize(f) > 20000)
        files = [f for f in files if int(hashlib.sha1(f.encode()).hexdigest(), 16) % 2 == 0]
        random.Random(SEED).shuffle(files)
        for f in files:
            if got[key] >= per: break
            try: t = open(f, encoding='utf-8').read()
            except Exception: continue
            if raw_window('code', 'local:' + f, t, key, 'local', lic): got[key] += 1
    log('pi code', got)

def pi_math():
    n = 0; buf = []; sids = []
    for sub in ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'):
        for j, r in enumerate(stream('EleutherAI/hendrycks_math', sub, split='test')):
            if j >= 220: break
            buf.append(f"Problem: {r['problem']}\nSolution: {r['solution']}\n\n"); sids.append(f'hendrycks_math/{sub}/test/{j}')
    random.Random(SEED).shuffle(buf)
    k = 0
    while n < N_PI and k < len(buf):
        t = ''; s0 = k
        while len(ids(t)) < W and k < len(buf): t += buf[k]; k += 1
        if raw_window('math', f'hendrycks_math/test/pack{s0}-{k}', t, 'en', 'EleutherAI/hendrycks_math:test', 'MIT'): n += 1
    log('pi math', n)

def pi_wiki(domain, cfgs, per_cfg, skip):
    n = 0
    for cfg in cfgs:
        got = 0
        ds = stream('wikimedia/wikipedia', cfg, split='train').shuffle(seed=SEED, buffer_size=2000).skip(skip)
        for r in ds:
            if got >= per_cfg: break
            if len(r['text']) < 9000: continue
            if raw_window(domain, f'wikipedia/{cfg}/{r["id"]}', r['text'], cfg.split('.')[1], f'wikimedia/wikipedia:{cfg}', 'CC-BY-SA-3.0 and GFDL'):
                got += 1; n += 1
    log('pi', domain, n)

def pi_long():
    n = 0
    for j, r in enumerate(stream('emozilla/pg19', split='test')):
        if n >= N_PI: break
        t = r['text']
        if len(t) < 60000: continue
        t = t[20000:]                                  # hors page de garde : préfixe utile > 2048 tokens
        if raw_window('long', f'pg19/test/{j}', t, 'en', 'emozilla/pg19:test', 'public domain (Project Gutenberg)'): n += 1
    log('pi long', n)

def pi_web():
    n = 0
    for j, r in enumerate(stream('allenai/c4', 'en', split='validation')):
        if n >= N_PI: break
        if len(r['text']) < 14000: continue
        if raw_window('web', f'c4/en/validation/{j}', r['text'], 'en', 'allenai/c4:en:validation', 'ODC-BY'): n += 1
    log('pi web', n)

# ================================================================== invites (P-I domaines 1, 4, 8)
def parse_toolace_calls(v):
    v = v.strip()
    if not (v.startswith('[') and v.endswith(']')): return None
    def split_calls(s):
        out = []; depth = 0; cur = ''
        for ch in s:
            if ch in '([{': depth += 1
            if ch in ')]}': depth -= 1
            if ch == ',' and depth == 0: out.append(cur.strip()); cur = ''
            else: cur += ch
        if cur.strip(): out.append(cur.strip())
        return out
    calls = []
    for p in split_calls(v[1:-1]):
        m = re.match(r'^([^()]+?)\((.*)\)$', p, re.S)
        if not m: return None
        node = ast.parse(f'f({m.group(2)})', mode='eval').body
        args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        if node.args: return None
        calls.append({"function": {"name": m.group(1).strip(), "arguments": args}})
    return calls or None

def toolace_conv(r):
    sysm = r['system']; k = sysm.index('[', sysm.index('list of functions'))
    fl, _ = json.JSONDecoder().raw_decode(sysm[k:])
    tools = [{"type": "function", "function": f} for f in fl]
    msgs = []
    for m in r['conversations']:
        if m['from'] == 'user': msgs.append({"role": "user", "content": m['value']})
        elif m['from'] == 'assistant':
            calls = parse_toolace_calls(m['value']) if m['value'].strip().startswith('[') else None
            msgs.append({"role": "assistant", "content": "", "tool_calls": calls} if calls else {"role": "assistant", "content": m['value']})
        elif m['from'] == 'tool': msgs.append({"role": "tool", "content": m['value']})
        else: raise ValueError
    return msgs, tools

def glaive_conv(r):
    sysm = r['system']; k = sysm.index('{'); dec = json.JSONDecoder(); tools = []
    while True:
        while k < len(sysm) and sysm[k] != '{': k += 1
        if k >= len(sysm): break
        o, e = dec.raw_decode(sysm[k:]); tools.append({"type": "function", "function": o}); k += e
    if not tools: raise ValueError
    chat_ = r['chat'].replace('<|endoftext|>', '')
    segs = re.split(r'(?:^|\n\n\n)(USER|ASSISTANT|FUNCTION RESPONSE): ', chat_)
    msgs = []
    for role, body in zip(segs[1::2], segs[2::2]):
        body = body.strip()
        if role == 'USER': msgs.append({"role": "user", "content": body})
        elif role == 'FUNCTION RESPONSE': msgs.append({"role": "tool", "content": body})
        elif body.startswith('<functioncall>'):
            b = body[len('<functioncall>'):].strip()
            name = re.match(r'\{\s*"name":\s*"([^"]+)"', b).group(1)
            a = b[b.index('"arguments":') + len('"arguments":'):].strip()
            assert a.endswith('}'); a = a[:-1].strip()
            if a[0] in '\'"' and a[-1] == a[0]: a = a[1:-1]
            args = json.loads(a)
            if not isinstance(args, dict): raise ValueError
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
        else: msgs.append({"role": "assistant", "content": body})
    if not msgs or msgs[0]['role'] != 'user' or not any(m.get('tool_calls') for m in msgs): raise ValueError
    return msgs, tools

def prompts_agentic(n_toolace, n_glaive):
    got = 0
    for j, r in enumerate(stream('Team-ACE/ToolACE', split='train')):
        if j >= 1500 or got >= n_toolace: break
        try:
            msgs, tools = toolace_conv(r)
            # contexte jusqu'au dernier tour de l'assistant (exclu) : le professeur produit ce tour
            last = max(i for i, m in enumerate(msgs) if m['role'] == 'assistant')
            ctx = msgs[:last]
            if not ctx or ctx[-1]['role'] not in ('user', 'tool'): continue
            chat(ctx, tools=tools)
        except Exception: continue
        PROMPTS.append(dict(domain='agentic', source_id=f'toolace/{j}', messages=ctx, tools=tools)); got += 1
    g2 = 0
    for j, r in enumerate(stream('glaiveai/glaive-function-calling-v2', split='train')):
        if j >= 15000 or g2 >= n_glaive: break
        try:
            msgs, tools = glaive_conv(r)
            last = max(i for i, m in enumerate(msgs) if m['role'] == 'assistant')
            ctx = msgs[:last]
            if not ctx or ctx[-1]['role'] not in ('user', 'tool'): continue
            chat(ctx, tools=tools)
        except Exception: continue
        PROMPTS.append(dict(domain='agentic', source_id=f'glaive/{j}', messages=ctx, tools=tools)); g2 += 1
    log('prompts agentic', got, g2)

def prompts_reasoning(n):
    got = 0
    for j, r in enumerate(stream('open-thoughts/OpenThoughts-114k', 'default', split='train')):
        if j >= 2500 or got >= n: break
        q = r['conversations'][0]['value']
        if len(q) > 3000: continue
        PROMPTS.append(dict(domain='reasoning', source_id=f'openthoughts/{j}', messages=[{"role": "user", "content": q}], tools=None)); got += 1
    log('prompts reasoning', got)

def prompts_instr(n):
    got = 0
    for j, r in enumerate(stream('HuggingFaceH4/ultrachat_200k', split='test_sft')):
        if got >= n: break
        q = r['messages'][0]['content']
        if len(q) > 3000: continue
        PROMPTS.append(dict(domain='instructions', source_id=f'ultrachat/test_sft/{r["prompt_id"]}', messages=[{"role": "user", "content": q}], tools=None)); got += 1
    log('prompts instructions', got)

# ================================================================== calibration
CAL = {k: [] for k in CAL_COMP}       # domain -> list of (source_id, ids2048, dataset, license)
PI_G = set()
BOILER = set()
def init_boiler():
    dummy_tool = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    for msgs, kw in [([{"role": "user", "content": ""}], {}), ([{"role": "user", "content": ""}, {"role": "assistant", "content": ""}], {}),
                     ([{"role": "user", "content": ""}], {"tools": dummy_tool}),
                     ([{"role": "user", "content": ""}, {"role": "assistant", "content": "", "reasoning_content": "x"}], {}),
                     ([{"role": "user", "content": ""}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}, {"role": "tool", "content": ""}], {"tools": dummy_tool})]:
        BOILER.update(grams(chat(msgs, **kw)))
PI_IDS = set()
def clash(sid, t):
    if sid in PI_IDS: return True
    return bool((grams(t) & PI_G) - BOILER)

def need(dom): return int(round(N_CAL * CAL_COMP[dom]))

def cal_raw(dom, sid, text, ds, lic, bot):
    if clash(sid, text[:40000]): return False
    i = ids(text)
    if len(i) < L - 1: return False
    d = ([BOT] + i[:L - 1]) if bot else i[:L]
    CAL[dom].append((sid, d, ds, lic)); return True

def cal_wiki():
    langs_multi = ['fr', 'de', 'es', 'ru', 'zh', 'ja', 'ar']        # + en (prose) = 8 langues
    k = 0; got = 0
    ds = stream('wikimedia/wikipedia', '20231101.en', split='train').shuffle(seed=SEED + 1, buffer_size=3000).skip(5000)
    for r in ds:
        if got >= need('prose'): break
        if len(r['text']) < 9000: continue
        if cal_raw('prose', f'wikipedia/20231101.en/{r["id"]}', r['text'], 'wikimedia/wikipedia:20231101.en', 'CC-BY-SA-3.0 and GFDL', got % 2 == 0): got += 1
    per = -(-need('multi') // len(langs_multi)); tot = 0
    for lg in langs_multi:
        g = 0
        ds = stream('wikimedia/wikipedia', f'20231101.{lg}', split='train').shuffle(seed=SEED + 1, buffer_size=2000).skip(5000)
        for r in ds:
            if g >= per or tot >= need('multi'): break
            if len(r['text']) < 6000: continue
            if cal_raw('multi', f'wikipedia/20231101.{lg}/{r["id"]}', r['text'], f'wikimedia/wikipedia:20231101.{lg}', 'CC-BY-SA-3.0 and GFDL', g % 2 == 0): g += 1; tot += 1
    log('cal prose', len(CAL['prose']), 'multi', len(CAL['multi']))

def cal_web():
    g = 0
    for j, r in enumerate(stream('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train').skip(30000)):
        if g >= need('web'): break
        if len(r['text']) < 9000: continue
        if cal_raw('web', f'fineweb-edu/{r["id"]}', r['text'], 'HuggingFaceFW/fineweb-edu:sample-10BT', 'ODC-BY', g % 2 == 0): g += 1
    log('cal web', g)

def pack_units(units, dom, n, prefix, ds, lic):
    """concatène des unités rendues (conversations) jusqu'à 2048 tokens."""
    cur = []; srcs = []; made = 0
    for sid, t in units:
        if made >= n: break
        # filtre de 13 mots sur le contenu des messages : le bloc tool_declare (JSON d'outils réutilisés) est exclu
        if clash(sid, re.sub(r'<\|message_system\|>tool_declare<\|content_xml\|>.*?<\|end_message\|>', ' ', t, flags=re.S)): continue
        i = ids(t)
        cur += i; srcs.append(sid)
        if len(cur) >= L:
            CAL[dom].append(('|'.join(srcs[:4]) + ('|+%d' % (len(srcs) - 4) if len(srcs) > 4 else ''), cur[:L], ds, lic)); made += 1
            cur = []; srcs = []
    return made

def cal_math():
    def units():
        for j, r in enumerate(stream('meta-math/MetaMathQA', split='train').shuffle(seed=SEED, buffer_size=20000)):
            yield f'metamathqa/{j}', f"Problem: {r['query']}\nSolution: {r['response']}\n\n"
    log('cal math', pack_units(units(), 'math', need('math'), '', 'meta-math/MetaMathQA', 'MIT'))

def cal_chat():
    n = need('chat'); parts = dict(ultra=0.25, toolace=0.20, glaive=0.13, ot=0.21, gr=0.21)
    def ultra():
        for j, r in enumerate(stream('HuggingFaceH4/ultrachat_200k', split='train_sft').shuffle(seed=SEED, buffer_size=5000)):
            try: yield f'ultrachat/{r["prompt_id"]}', chat(r['messages'])
            except Exception: continue
    def toolace():
        for j, r in enumerate(stream('Team-ACE/ToolACE', split='train')):
            if j < 2000: continue
            try:
                msgs, tools = toolace_conv(r)
                if not any(m.get('tool_calls') for m in msgs): continue
                yield f'toolace/{j}', chat(msgs, tools=tools)
            except Exception: continue
    def glaive():
        for j, r in enumerate(stream('glaiveai/glaive-function-calling-v2', split='train')):
            if j < 20000: continue
            try:
                msgs, tools = glaive_conv(r); yield f'glaive/{j}', chat(msgs, tools=tools)
            except Exception: continue
    def rr(q, th, ans): return chat([{"role": "user", "content": q}, {"role": "assistant", "content": ans.strip(), "reasoning_content": th.strip()}])
    def ot():
        for j, r in enumerate(stream('open-thoughts/OpenThoughts-114k', 'default', split='train')):
            if j < 3000: continue
            try:
                q = r['conversations'][0]['value']; a = r['conversations'][1]['value']
                m = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>\s*<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>', a, re.S)
                if not m or len(m.group(1)) + len(m.group(2)) > 11000: continue
                yield f'openthoughts/{j}', rr(q, m.group(1), m.group(2))
            except Exception: continue
    def gr():
        for j, r in enumerate(stream('glaiveai/reasoning-v1-20m', split='train')):
            if j < 3000: continue
            try:
                a = r['response']
                if not a.lstrip().startswith('<think>') or '</think>' not in a: continue
                th, ans = a.split('</think>', 1); th = th.split('<think>', 1)[1]
                if not ans.strip(): continue
                yield f'glaive-reasoning/{j}', rr(r['prompt'], th, ans)
            except Exception: continue
    srcs = dict(ultra=(ultra, 'HuggingFaceH4/ultrachat_200k', 'MIT'), toolace=(toolace, 'Team-ACE/ToolACE', 'apache-2.0'),
                glaive=(glaive, 'glaiveai/glaive-function-calling-v2', 'apache-2.0'), ot=(ot, 'open-thoughts/OpenThoughts-114k', 'apache-2.0'),
                gr=(gr, 'glaiveai/reasoning-v1-20m', 'apache-2.0'))
    def one(k):
        f, ds, lic = srcs[k]; return k, pack_units(f(), 'chat', int(round(n * parts[k])), k, ds, lic)
    for k in srcs: log('cal chat', one(k))

def cal_code():
    n = need('code'); per = n // 4; got = dict(py=0, rs=0, c=0, js=0)
    for j, r in enumerate(stream('codeparrot/codeparrot-clean', split='train').skip(200000)):
        if got['py'] >= per: break
        if len(r['content']) < 12000: continue
        if cal_raw('code', f'codeparrot-clean/{200000 + j}', r['content'], 'codeparrot/codeparrot-clean', r['license'], got['py'] % 2 == 0): got['py'] += 1
    for j, r in enumerate(stream('ammarnasr/the-stack-rust-clean', split='train').skip(20000)):
        if got['rs'] >= per: break
        if len(r['content']) < 12000: continue
        if cal_raw('code', f'the-stack-rust-clean/{20000 + j}', r['content'], 'ammarnasr/the-stack-rust-clean', 'the-stack (permissive)', got['rs'] % 2 == 0): got['rs'] += 1
    for ext, key in ((('.c', '.cpp', '.h', '.hpp', '.cc', '.py'), 'c'), (('.js', '.ts'), 'js')):
        files = sorted(f for e in ext for f in glob.glob(f'/usr/**/*{e}', recursive=True) if os.path.getsize(f) > 12000)
        files = [f for f in files if int(hashlib.sha1(f.encode()).hexdigest(), 16) % 2 == 1]
        random.Random(SEED).shuffle(files)
        lim = per if key == 'c' else n - per * 3
        for f in files:
            if got[key] >= lim: break
            try: t = open(f, encoding='utf-8').read()
            except Exception: continue
            if cal_raw('code', 'local:' + f, t, 'local', 'local (licences des paquets)', got[key] % 2 == 0): got[key] += 1
    log('cal code', got)

# ================================================================== main
if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('all', 'pi'):
        jobs = [pi_code, pi_math, lambda: pi_wiki('encyclopedic', ['20231101.en'], N_PI, 0),
                lambda: pi_wiki('multilingual', ['20231101.' + l for l in ['fr', 'de', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ar', 'hi', 'ko', 'tr'][:N_PI]], 1, 0),
                pi_long, pi_web, lambda: prompts_agentic(40, 40), lambda: prompts_reasoning(48), lambda: prompts_instr(64)]
        with ThreadPoolExecutor(9) as ex:
            for f in [ex.submit(j) for j in jobs]:
                f.result()
        with open(f'{D}/pi_raw.jsonl', 'w') as f:
            for r in PI: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(f'{D}/pi_prompts.jsonl', 'w') as f:
            for r in PROMPTS: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        log('PI raw', len(PI), 'prompts', len(PROMPTS))
    if stage in ('all', 'cal'):
        PI = [json.loads(l) for l in open(f'{D}/pi_raw.jsonl')]; PROMPTS = [json.loads(l) for l in open(f'{D}/pi_prompts.jsonl')]
        for r in PI: PI_G.update(grams(r['text'])); PI_IDS.add(r['source_id'])
        for r in PROMPTS:
            PI_IDS.add(r['source_id'])
            PI_G.update(grams(chat(r['messages'])))          # contenu des messages (sans le bloc tool_declare)
        init_boiler()
        with ThreadPoolExecutor(6) as ex:
            for f in [ex.submit(j) for j in (cal_wiki, cal_web, cal_math, cal_chat, cal_code)]:
                f.result()
        # entrelacement proportionnel
        order = []; rng = random.Random(SEED)
        for k in CAL: rng.shuffle(CAL[k])
        tot = sum(len(v) for v in CAL.values()); pos = {k: 0 for k in CAL}
        for t in range(tot):
            k = min(CAL, key=lambda k: (pos[k] + 0.5) / max(len(CAL[k]), 1e-9) if pos[k] < len(CAL[k]) else 9e9)
            order.append((k,) + CAL[k][pos[k]]); pos[k] += 1
        X = torch.tensor([o[2] for o in order], dtype=torch.int32)
        torch.save(X, f'{D}/cal_ink.pt')
        meta = dict(n_docs=len(order), tokens=int(X.numel()), doc_len=L, composition_target=CAL_COMP,
                    composition_real={k: len(v) / tot for k, v in CAL.items()}, counts={k: len(v) for k, v in CAL.items()},
                    docs=[dict(domain=o[0], source_id=o[1], dataset=o[3], license=o[4]) for o in order])
        json.dump(meta, open(f'{D}/cal_ink_meta.json', 'w'), ensure_ascii=False, indent=0)
        log('CAL', X.shape, meta['counts'])
