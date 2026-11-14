import os, sys, re, json, ast, random, hashlib, glob, torch, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 2468; TOK = '/root/models/qwen3.8-27b'; D = '/workspace/work/data'; OUT = D + '/cal2'
L = 2048; BOS = 248044; VOCAB = 248320
COMP = '/workspace/work/competitors/arki05/eval'
tok = AutoTokenizer.from_pretrained(TOK)
assert tok.convert_tokens_to_ids('<|endoftext|>') == BOS
ids = lambda t: tok(t, add_special_tokens=False)['input_ids']
def stream(*a, **k): return load_dataset(*a, streaming=True, **k)

# ---------- old sets
old = [json.loads(l) for l in open(f'{D}/corpus.jsonl')] + [json.loads(l) for l in open(f'{D}/big.jsonl')]
old_sids = {r['source_id'] for r in old}
lm = json.load(open(f'{D}/long_manifest.json'))
old_sids |= {m['source_id'] for m in lm['sources']}
old_starts = {r['text'][:300] for r in old}
lt = torch.load(f'{D}/long_tokens.pt')['long']
long_starts = [tok.decode(r[:150].tolist())[:200] for r in lt]
old_queries = set()
for r in old:
    for m in re.finditer(r'Problem: (.*?)\nSolution:', r['text'], re.S): old_queries.add(m.group(1).strip())
old_chat_pids = {s for s in old_sids if s.startswith('ultrachat/')}
def old_clash(sid, text):
    if sid in old_sids: return True
    s = text[:300]
    if s in old_starts: return True
    if any(s[:200] == p for p in long_starts): return True
    return False

# ---------- competitor
W = json.load(open(f'{COMP}/windows.json'))['windows']
comp_ids = {}
for w in W:
    for d in w['documents']: comp_ids.setdefault(d['dataset'], set()).add(d['id'])
def comp_idx(ds, prefix):
    return {int(x[len(prefix):]) for x in comp_ids.get(ds, set()) if x.startswith(prefix) and x[len(prefix):].isdigit()}
comp_glaive = comp_idx('glaiveai/glaive-function-calling-v2', 'tools/glaive-function-calling-v2/')
comp_toolace = comp_idx('Team-ACE/ToolACE', 'tools/toolace/')
comp_fw = {x.split('/')[-1] for x in comp_ids.get('HuggingFaceFW/fineweb-edu', set()) | comp_ids.get('mlfoundations/dclm-baseline-1.0-parquet', set())}
def near(i, S, m=100): return any(abs(i - j) <= m for j in S)
def words(t): return re.findall(r'\w+', t.lower())
def grams(t, n=13):
    w = words(t); return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}
comp_g = set()
comp_texts = [json.loads(l)['text'] for l in open(f'{COMP}/text.jsonl')]
for t in comp_texts: comp_g |= grams(t)
# boilerplate: template rendering of (near) empty conversations
dummy_tool = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
boiler = set()
for msgs, kw in [([{"role": "user", "content": ""}], {}), ([{"role": "user", "content": ""}, {"role": "assistant", "content": ""}], {}),
                 ([{"role": "user", "content": ""}], {"tools": dummy_tool}),
                 ([{"role": "user", "content": ""}, {"role": "assistant", "content": "", "reasoning_content": ""}], {}),
                 ([{"role": "user", "content": ""}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}, {"role": "tool", "content": ""}], {"tools": dummy_tool})]:
    boiler |= grams(tok.apply_chat_template(msgs, tokenize=False, **kw))
def comp_bad(t): return bool((grams(t) & comp_g) - boiler)

def mostly_list(t):
    ls = [l for l in t.split('\n') if l.strip()]
    return not ls or sum(len(l) < 40 for l in ls) / len(ls) > 0.5
def cand(sid, text, lang, dsname, lic, tids=None, check_text=None):
    """returns candidate dict with >=L tokens or None"""
    if old_clash(sid, text): return None
    i = tids if tids is not None else ids(text)
    if len(i) < L: return None
    ct = tok.decode(i[:L])
    if comp_bad(ct): return None
    return dict(source_id=sid, lang=lang, dataset=dsname, license=lic, ids=i[:L])

WL = 'CC-BY-SA-3.0 and GFDL'
TARGET = dict(en_prose=12, web=12, multiling=20, code=12, math=10, science=10, chat=16, tools=16, reasoning=20)
MARGIN = 6
LOG = {}
def log(*a): print(*a, flush=True)

# ---------- raw domains
def wiki(cfg, lang, n, minc, buf=3000, skip=8000):
    out = []
    ds = stream('wikimedia/wikipedia', cfg, split='train').shuffle(seed=SEED, buffer_size=buf)
    for j, r in enumerate(ds):
        if j < skip: continue
        if len(r['text']) >= minc and not mostly_list(r['text'][:3000]):
            c = cand(f'wikipedia_{lang}/' + r['id'], r['text'], lang, 'wikimedia/wikipedia:' + cfg, WL)
            if c: out.append(c)
        if len(out) >= n: break
    return out
def en_prose(): return wiki('20231101.en', 'en', TARGET['en_prose'] + MARGIN, 9000, 5000)
LANGS = ['fr', 'zh', 'de', 'es', 'ja', 'ar', 'ru', 'ko']
QUOTA = {lg: (3 if k < 4 else 2) for k, lg in enumerate(LANGS)}
def multiling():
    def one(lg):
        mn = {'zh': 2500, 'ja': 3000, 'ko': 4000}.get(lg, 6500)
        return wiki('20231101.' + lg, lg, QUOTA[lg] + 3, mn)
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, LANGS))
    return dict(zip(LANGS, res))
def web():
    rng = random.Random(SEED)
    ds = stream('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train').skip(20000 + rng.randint(0, 5000))
    out = []
    for r in ds:
        if r['id'] in comp_fw or r['id'].split('/')[-1] in comp_fw: continue
        if len(r['text']) < 9000: continue
        c = cand('fineweb-edu/' + r['id'], r['text'], 'en', 'HuggingFaceFW/fineweb-edu:sample-10BT', 'ODC-By 1.0')
        if c: out.append(c)
        if len(out) >= TARGET['web'] + MARGIN: break
    return out
def code():
    rng = random.Random(SEED)
    py = sorted(glob.glob('/usr/lib/python3.12/**/*.py', recursive=True)) + sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/**/*.py', recursive=True))
    cc = sorted(glob.glob('/usr/include/**/*.h', recursive=True)) + sorted(glob.glob('/usr/include/**/*.hpp', recursive=True)) + sorted(glob.glob('/usr/include/**/*.c', recursive=True)) + sorted(glob.glob('/usr/include/**/*.cc', recursive=True))
    res = {}
    for name, files, lang, n in [('py', py, 'python', 6 + 3), ('c', cc, 'c', 6 + 3)]:
        rng.shuffle(files); out = []; seen = set(); starts = set()
        for f in files:
            if f in old_sids: continue
            try: t = open(f, encoding='utf-8').read()
            except Exception: continue
            if len(t) < 8000 or t[:300] in starts: continue
            h = hashlib.sha1(t.encode()).hexdigest()
            if h in seen: continue
            if lang == 'python': dsn, lic = ('local:python3.12-stdlib', 'PSF License (CPython stdlib)') if f.startswith('/usr/lib/') else ('local:dist-packages', 'see package metadata')
            else: dsn, lic = 'local:/usr/include', 'system headers (various: GPL/LGPL/BSD etc.) - research use'
            c = cand(f, t, 'python' if lang == 'python' else ('cpp' if f.endswith(('.hpp', '.cc')) or '/c++/' in f else 'c'), dsn, lic)
            if c: seen.add(h); starts.add(t[:300]); out.append(c)
            if len(out) >= n: break
        res[name] = out
    return res
def math():
    ds = stream('meta-math/MetaMathQA', split='train').shuffle(seed=SEED, buffer_size=20000)
    out = []; cur = []; ct = 0
    for idx, r in enumerate(ds):
        if idx < 12000: continue
        q = r['query'].strip()
        if q in old_queries: continue
        s = f"Problem: {q}\nSolution: {r['response'].strip()}\n\n"
        cur.append((idx, s)); ct += len(ids(s))
        if ct >= L + 50:
            t = ''.join(s for _, s in cur)
            c = cand('metamathqa/s2468/' + '+'.join(str(a) for a, _ in cur), t, 'en', 'meta-math/MetaMathQA:train', 'MIT')
            if c: out.append(c)
            cur = []; ct = 0
        if len(out) >= TARGET['math'] + MARGIN: break
    return out
def science():
    name = 'ccdv/arxiv-summarization'; out = []
    for j, r in enumerate(stream(name, split='train')):
        if j < 30000: continue
        t = r['article']
        if len(t) < 10000: continue
        c = cand(f'{name}/unshuffled/{j}', t, 'en', name + ':train', 'unknown/arXiv (dataset card apache-2.0; papers retain author licenses)')
        if c: out.append(c)
        if len(out) >= TARGET['science'] + MARGIN: break
    return out

# ---------- template-rendered domains
def pack(units, n, sidprefix, lang, dsname, lic):
    """units: iterable of (sid, rendered_text, ntok). Packs consecutive units to >= L tokens."""
    out = []; cur = []; ct = 0
    for sid, t, nt in units:
        cur.append((sid, t)); ct += nt
        if ct >= L:
            text = ''.join(t for _, t in cur); sids = '+'.join(s for s, _ in cur)
            c = cand(sidprefix + sids, text, lang, dsname, lic, tids=ids(text))
            if c: out.append(c)
            cur = []; ct = 0
        if len(out) >= n: break
    return out
def chat():
    ds = stream('HuggingFaceH4/ultrachat_200k', split='train_sft').shuffle(seed=SEED, buffer_size=5000)
    def units():
        for j, r in enumerate(ds):
            if j < 12000: continue
            sid = r['prompt_id']
            if 'ultrachat/' + sid in old_sids: continue
            t = tok.apply_chat_template(r['messages'], tokenize=False)
            if comp_bad(t) or old_clash('ultrachat/' + sid, t): continue
            yield sid, t, len(ids(t))
    return pack(units(), TARGET['chat'] + MARGIN, 'ultrachat/s2468/', 'en', 'HuggingFaceH4/ultrachat_200k:train_sft', 'MIT')

def split_calls(s):
    parts = []; depth = 0; q = None; cur = ''
    for ch in s:
        if q:
            cur += ch
            if ch == q and not cur.endswith('\\' + q): q = None
            continue
        if ch in '"\'': q = ch; cur += ch; continue
        if ch in '([{': depth += 1
        if ch in ')]}': depth -= 1
        if ch == ',' and depth == 0: parts.append(cur.strip()); cur = ''; continue
        cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts
def parse_toolace_calls(v):
    v = v.strip()
    if not (v.startswith('[') and v.endswith(']')): return None
    calls = []
    for p in split_calls(v[1:-1]):
        m = re.match(r'^([^()]+?)\((.*)\)$', p, re.S)
        if not m: return None
        node = ast.parse(f'f({m.group(2)})', mode='eval').body
        args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        if node.args: return None
        calls.append({"function": {"name": m.group(1).strip(), "arguments": args}})
    return calls or None
def toolace_units():
    ds = stream('Team-ACE/ToolACE', split='train')
    for j, r in enumerate(ds):
        if j < 2000 or near(j, comp_toolace, 30): continue
        try:
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
            if not any(m.get('tool_calls') for m in msgs): continue
            t = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
        except Exception: continue
        if comp_bad(t) or old_clash(f'toolace/{j}', t): continue
        yield f'toolace/{j}', t, len(ids(t))
def glaive_units():
    ds = stream('glaiveai/glaive-function-calling-v2', split='train')
    for j, r in enumerate(ds):
        if j < 20000 or near(j, comp_glaive, 100): continue
        try:
            sysm = r['system']; k = sysm.index('{'); dec = json.JSONDecoder(); tools = []
            while True:
                while k < len(sysm) and sysm[k] != '{': k += 1
                if k >= len(sysm): break
                o, e = dec.raw_decode(sysm[k:]); tools.append({"type": "function", "function": o}); k += e
            if not tools: continue
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
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
                else: msgs.append({"role": "assistant", "content": body})
            if not msgs or msgs[0]['role'] != 'user' or not any(m.get('tool_calls') for m in msgs): continue
            t = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
        except Exception: continue
        if comp_bad(t) or old_clash(f'glaive/{j}', t): continue
        yield f'glaive/{j}', t, len(ids(t))
def tools():
    a = pack(toolace_units(), 16, 'toolace/', 'en', 'Team-ACE/ToolACE', 'apache-2.0')
    b = pack(glaive_units(), 8 + 3, 'glaive/', 'en', 'glaiveai/glaive-function-calling-v2', 'apache-2.0')
    return a, b

def render_reasoning(q, think, ans):
    return tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ans.strip(), "reasoning_content": think.strip()}], tokenize=False)
def ot_units():
    ds = stream('open-thoughts/OpenThoughts-114k', 'default', split='train')
    for j, r in enumerate(ds):
        if j < 3000: continue
        try:
            q = r['conversations'][0]['value']; a = r['conversations'][1]['value']
            m = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>\s*<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>', a, re.S)
            if not m: continue
            if len(m.group(1)) + len(m.group(2)) > 11000: continue  # keep short enough that think + answer are visible
            t = render_reasoning(q, m.group(1), m.group(2))
        except Exception: continue
        if comp_bad(t) or old_clash(f'openthoughts/{j}', t): continue
        yield f'openthoughts/{j}', t, len(ids(t))
def gr_units():
    ds = stream('glaiveai/reasoning-v1-20m', split='train')
    for j, r in enumerate(ds):
        if j < 3000: continue
        try:
            a = r['response']
            if not a.lstrip().startswith('<think>') or '</think>' not in a: continue
            th, ans = a.split('</think>', 1); th = th.split('<think>', 1)[1]
            if not ans.strip(): continue
            t = render_reasoning(r['prompt'], th, ans)
        except Exception: continue
        if comp_bad(t) or old_clash(f'glaive-reasoning/{j}', t): continue
        yield f'glaive-reasoning/{j}', t, len(ids(t))
def reasoning():
    a = pack(ot_units(), 10 + 3, 'openthoughts/', 'en', 'open-thoughts/OpenThoughts-114k', 'apache-2.0')
    b = pack(gr_units(), 10 + 3, 'glaive-reasoning/', 'en', 'glaiveai/reasoning-v1-20m', 'apache-2.0')
    return a, b

# ---------- run
jobs = dict(en_prose=en_prose, web=web, multiling=multiling, code=code, math=math, science=science, chat=chat, tools=tools, reasoning=reasoning)
with ThreadPoolExecutor(9) as ex: futs = {k: ex.submit(f) for k, f in jobs.items()}
res = {}
for k, f in futs.items():
    try: res[k] = f.result()
    except Exception: traceback.print_exc(); res[k] = None
# assemble per-domain lists of exactly target (or shortfall)
dom = {}
dom['en_prose'] = res['en_prose'] or []
dom['web'] = res['web'] or []
ml = res['multiling'] or {}
dom['multiling'] = [c for lg in LANGS for c in ml.get(lg, [])[:QUOTA[lg]]]
dom['code'] = (res['code']['py'][:6] + res['code']['c'][:6]) if res['code'] else []
dom['math'] = res['math'] or []; dom['science'] = res['science'] or []; dom['chat'] = res['chat'] or []
def interleave(a, b): return [x for p in zip(a, b) for x in p] + a[len(b):] + b[len(a):]
dom['tools'] = interleave(*res['tools']) if res['tools'] else []
dom['reasoning'] = interleave(*res['reasoning']) if res['reasoning'] else []
# multiling interleave languages for stripe order irrelevant; apply targets, uniqueness
RAW = ['en_prose', 'web', 'multiling', 'code', 'math', 'science']
final = {}
seen_sid = set(); seen_start = set()
for d, lst in dom.items():
    keep = []
    for c in lst:
        t = tok.decode(c['ids'])
        if c['source_id'] in seen_sid or (d in RAW and t[:300] in seen_start): continue
        seen_sid.add(c['source_id']); seen_start.add(t[:300]); keep.append(c)
    final[d] = keep[:TARGET[d]]
RAW = ['en_prose', 'web', 'multiling', 'code', 'math', 'science']
SUF = {}
for d in final:
    for k, c in enumerate(final[d]):
        c['domain'] = d; c['has_bos'] = False
        if d in RAW and k % 2 == 1:
            c['ids'] = [BOS] + c['ids'][:L - 1]; c['has_bos'] = True
        assert len(c['ids']) == L
        c['text'] = tok.decode(c['ids'])
        c['id'] = f'cal2_{d}_{k:03d}'
# final comp check on kept text (drop offenders)
bad = [c['id'] for d in final for c in final[d] if comp_bad(c['text'])]
log('post-assembly docs sharing non-boilerplate 13-grams:', bad)
for d in final: final[d] = [c for c in final[d] if c['id'] not in bad]
# round-robin interleave
order = list(TARGET); rows = []; k = 0
while any(k < len(final[d]) for d in order):
    for d in order:
        if k < len(final[d]): rows.append(final[d][k])
    k += 1
T = torch.tensor([c['ids'] for c in rows], dtype=torch.long)
meta = [{kk: c[kk] for kk in ['id', 'source_id', 'domain', 'lang', 'dataset', 'license', 'has_bos']} for c in rows]
torch.save({'cal2': T, 'meta': meta}, f'{OUT}/cal2_tokens.pt')
with open(f'{OUT}/cal2.jsonl', 'w') as f:
    for c in rows: f.write(json.dumps({kk: c[kk] for kk in ['id', 'source_id', 'domain', 'lang', 'dataset', 'license', 'has_bos', 'text']}, ensure_ascii=False) + '\n')
# overlap checks
all_g = set()
for c in rows: all_g |= grams(c['text'])
shared = all_g & comp_g
sh = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
cnt = {d: len(final[d]) for d in TARGET}
from collections import Counter
checks = dict(
    shared_13grams_with_competitor_text_before_boilerplate_discount=len(shared),
    shared_13grams_after_boilerplate_discount=len(shared - boiler),
    n_boilerplate_13grams=len(boiler), n_competitor_13grams=len(comp_g),
    source_id_in_old_sets=sum(c['source_id'] in old_sids for c in rows),
    first300_start_in_old_sets=sum(c['text'][:300] in old_starts or (c['text'][:300].startswith('<|endoftext|>') and False) for c in rows),
    duplicate_source_ids=len(rows) - len({c['source_id'] for c in rows}),
    duplicate_texts=len(rows) - len({c['text'] for c in rows}),
    competitor_doc_ids_used=sum(any(c['source_id'].endswith(x.split('/')[-1]) for x in comp_ids.get('HuggingFaceFW/fineweb-edu', [])) for c in rows if c['domain'] == 'web') +
        sum(int(c['source_id'].split('/')[0] in ('glaive', 'toolace') and any(near(int(s.split('/')[-1]), comp_glaive if s.startswith('glaive') else comp_toolace, 0) for s in c['source_id'].split('+') if s.split('/')[-1].isdigit())) for c in rows if c['domain'] == 'tools'))
man = dict(seed=SEED, tokenizer=TOK, tokens_per_doc=L, bos_token_id=BOS, counts_per_domain=cnt, targets=TARGET,
           counts_multiling_language=dict(Counter(c['lang'] for c in rows if c['domain'] == 'multiling')),
           counts_code_language=dict(Counter(c['lang'] for c in rows if c['domain'] == 'code')),
           datasets=sorted({(c['domain'], c['dataset'], c['license']) for c in rows}),
           bos_rows=sum(c['has_bos'] for c in rows), overlap_checks=checks,
           sha256={'cal2_tokens.pt': sh(f'{OUT}/cal2_tokens.pt'), 'cal2.jsonl': sh(f'{OUT}/cal2.jsonl')})
json.dump(man, open(f'{OUT}/cal2_manifest.json', 'w'), indent=1, default=list)
log(json.dumps(man, indent=1, default=list))
