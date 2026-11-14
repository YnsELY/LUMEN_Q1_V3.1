import os, json, random, hashlib, glob, torch, itertools
from datasets import load_dataset
from transformers import AutoTokenizer
SEED=1234; TOK='/root/models/qwen3.8-27b'; OUT='/workspace/work/data'
N={'cal':24,'select':12,'confirm':12}; NEED=48; L=1024
tok=AutoTokenizer.from_pretrained(TOK)
def ids(t): return tok(t,add_special_tokens=False)['input_ids']
rng=random.Random(SEED)
LIC={}
domains={}  # domain -> list of dict(source_id,text,lang,dataset)
def add(dom,sid,text,lang,ds):
    i=ids(text)
    if len(i)>=L: domains.setdefault(dom,[]).append(dict(source_id=sid,text=text,lang=lang,dataset=ds,ids=i[:L]))
def cnt(d): return len(domains.get(d,[]))
def stream(*a,**k): return load_dataset(*a,streaming=True,**k)

def mostly_list(t):
    ls=[l for l in t.split('\n') if l.strip()]
    return not ls or sum(len(l)<40 for l in ls)/len(ls)>0.5

# 1 en_prose
LIC['en_prose']=dict(dataset='wikimedia/wikipedia',config='20231101.en',split='train',license='CC-BY-SA-3.0 and GFDL')
ds=stream('wikimedia/wikipedia','20231101.en',split='train').shuffle(seed=SEED,buffer_size=5000)
for r in ds:
    if len(r['text'])>=5000 and not mostly_list(r['text'][:3000]): add('en_prose','wikipedia_en/'+r['id'],r['text'],'en','wikimedia/wikipedia:20231101.en')
    if cnt('en_prose')>=NEED+12: break

# 2 multiling
langs=['fr','zh','de','es','ja','ar','ru']
LIC['multiling']=dict(dataset='wikimedia/wikipedia',config='20231101.{fr,zh,de,es,ja,ar,ru}',split='train',license='CC-BY-SA-3.0 and GFDL')
per={}
for lg in langs:
    ds=stream('wikimedia/wikipedia','20231101.'+lg,split='train').shuffle(seed=SEED,buffer_size=3000)
    lst=[]
    for r in ds:
        mn={'zh':1500,'ja':2000}.get(lg,4500)
        if len(r['text'])>=mn and not mostly_list(r['text'][:3000]):
            i=ids(r['text'])
            if len(i)>=L: lst.append(dict(source_id=f'wikipedia_{lg}/'+r['id'],text=r['text'],lang=lg,dataset=f'wikimedia/wikipedia:20231101.{lg}',ids=i[:L]))
        if len(lst)>=9: break
    per[lg]=lst
# round-robin over langs for 48 docs: 7 each + 6 extra -> balanced
ml=[]
for k in range(9):
    for lg in langs:
        if k<len(per[lg]): ml.append(per[lg][k])
# assign later by split with per-language striping
domains['multiling']=ml  # ordering is lang-striped; split handled specially

# 3 code
LIC['code']=dict(dataset='local:/usr/lib/python3.12 + /usr/include',config='files>=6000 chars',split='n/a',license='PSF License (CPython stdlib); system headers (various, e.g. GPL/LGPL/BSD) - research use')
files=sorted(glob.glob('/usr/lib/python3.12/**/*.py',recursive=True))
rng2=random.Random(SEED); rng2.shuffle(files)
seen=set()
for f in files:
    try: t=open(f,encoding='utf-8').read()
    except Exception: continue
    if len(t)<6000: continue
    h=hashlib.sha1(t.encode()).hexdigest()
    if h in seen: continue
    seen.add(h); add('code',f,t,'python','local:python3.12-stdlib')
    if cnt('code')>=NEED+8: break
if cnt('code')<NEED:
    hs=sorted(glob.glob('/usr/include/**/*.h',recursive=True)); rng2.shuffle(hs)
    for f in hs:
        try: t=open(f,encoding='utf-8').read()
        except Exception: continue
        if len(t)<6000: continue
        h=hashlib.sha1(t.encode()).hexdigest()
        if h in seen: continue
        seen.add(h); add('code',f,t,'c','local:/usr/include')
        if cnt('code')>=NEED+8: break

# 4 math: concatenate MetaMathQA items
LIC['math_reasoning']=dict(dataset='meta-math/MetaMathQA',config='default',split='train',license='MIT')
ds=stream('meta-math/MetaMathQA',split='train').shuffle(seed=SEED,buffer_size=20000)
cur=[];curtok=0;k=0
for idx,r in enumerate(ds):
    s=f"Problem: {r['query'].strip()}\nSolution: {r['response'].strip()}\n\n"
    cur.append((idx,s)); curtok+=len(ids(s))
    if curtok>=L+50:
        add('math_reasoning','metamathqa/'+'+'.join(str(a) for a,_ in cur),''.join(s for _,s in cur),'en','meta-math/MetaMathQA:train')
        cur=[];curtok=0
    if cnt('math_reasoning')>=NEED+4: break

# 5 chat
LIC['chat']=dict(dataset='HuggingFaceH4/ultrachat_200k',config='default',split='train_sft',license='MIT')
ds=stream('HuggingFaceH4/ultrachat_200k',split='train_sft').shuffle(seed=SEED,buffer_size=5000)
for r in ds:
    t=tok.apply_chat_template(r['messages'],tokenize=False)
    add('chat','ultrachat/'+r['prompt_id'],t,'en','HuggingFaceH4/ultrachat_200k:train_sft')
    if cnt('chat')>=NEED+8: break

# 6 science
sci=[('ccdv/arxiv-summarization',None,'train','article','unknown/arXiv (declared: apache-2.0 on dataset card; underlying papers retain author licenses)'),
     ('armanc/scientific_papers','arxiv','train','article','unknown (arXiv/PubMed OA)'),]
done=False
for name,cfg,sp,fld,lic in sci:
    try:
        a=(name,cfg) if cfg else (name,)
        ds=stream(*a,split=sp,trust_remote_code=True).shuffle(seed=SEED,buffer_size=2000)
        for j,r in enumerate(ds):
            add('science_tech',f'{name}/{j}',r[fld],'en',f'{name}:{sp}')
            if cnt('science_tech')>=NEED+8: break
        if cnt('science_tech')>=NEED: LIC['science_tech']=dict(dataset=name,config=cfg,split=sp,license=lic); done=True; break
    except Exception as e: print('sci fail',name,repr(e)[:200])
if not done:
    name='ccdv/pubmed-summarization'
    ds=stream(name,split='train').shuffle(seed=SEED,buffer_size=2000)
    for j,r in enumerate(ds):
        add('science_tech',f'{name}/{j}',r['article'],'en',name+':train')
        if cnt('science_tech')>=NEED: break
    LIC['science_tech']=dict(dataset=name,split='train',license='unknown (PubMed OA)')

# ---- split
DOMS=['en_prose','multiling','code','math_reasoning','chat','science_tech']
split={s:{} for s in N}
for d in DOMS:
    pool=domains[d]
    seen_t=set(); uniq=[]
    for x in pool:
        h=hashlib.sha1(x['text'].encode()).hexdigest()
        if h in seen_t: continue
        seen_t.add(h); uniq.append(x)
    pool=uniq
    assert len(pool)>=NEED,(d,len(pool))
    if d=='multiling':
        bylang={}
        for x in pool: bylang.setdefault(x['lang'],[]).append(x)
        # stripe: deal docs by language into splits cal/select/confirm proportionally
        order=['cal','cal','select','confirm']  # ratio 2:1:1
        buckets={s:[] for s in N}; c=0
        for k in range(9):
            for lg in langs:
                if k<len(bylang.get(lg,[])):
                    s=order[c%4]; c+=1
                    if len(buckets[s])<N[s]: buckets[s].append(bylang[lg][k])
        # fill any shortfall
        rest=[x for x in pool if all(x not in b for b in buckets.values())]
        for s in N:
            while len(buckets[s])<N[s]: buckets[s].append(rest.pop(0))
        for s in N: split[s][d]=buckets[s]
    else:
        random.Random(SEED).shuffle(pool)
        o=0
        for s in N: split[s][d]=pool[o:o+N[s]]; o+=N[s]

rows=[];T={};M={}
for s in N:
    T[s]=[];M[s]=[]
    for k in range(N[s]):
        for d in DOMS:
            x=split[s][d][k]; did=f'{s}_{d}_{k:02d}'
            rows.append(dict(id=did,source_id=x['source_id'],split=s,domain=d,lang=x['lang'],license=LIC[d]['license'],dataset=x['dataset'],text=tok.decode(x['ids'])))
            T[s].append(x['ids']); M[s].append(dict(id=did,source_id=x['source_id'],domain=d,lang=x['lang']))
with open(f'{OUT}/corpus.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
out={s:torch.tensor(T[s],dtype=torch.long) for s in N}; out['meta']=M
torch.save(out,f'{OUT}/corpus_tokens.pt')
# checks
sid={s:{r['source_id'] for r in rows if r['split']==s} for s in N}
txt={s:{hashlib.sha1(r['text'].encode()).hexdigest() for r in rows if r['split']==s} for s in N}
ok=True
for a,b in itertools.combinations(N,2): ok&= not(sid[a]&sid[b]) and not(txt[a]&txt[b])
ok&= len({r['source_id'] for r in rows})==len(rows)
sh=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
counts={s:{d:sum(1 for r in rows if r['split']==s and r['domain']==d) for d in DOMS} for s in N}
mlang={s:sorted({r['lang'] for r in rows if r['split']==s and r['domain']=='multiling'}) for s in N}
json.dump(dict(datasets=LIC,counts=counts,multiling_langs_per_split=mlang,sha256={'corpus.jsonl':sh(f'{OUT}/corpus.jsonl'),'corpus_tokens.pt':sh(f'{OUT}/corpus_tokens.pt')},tokenizer=TOK,seed=SEED,tokens_per_doc=L,
  checks=dict(no_source_id_or_text_overlap_between_splits=bool(ok))),open(f'{OUT}/data_manifest.json','w'),indent=1)
print('overlap check',ok,counts,mlang)
