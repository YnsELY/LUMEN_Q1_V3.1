import os, sys, re, json, random, hashlib, glob, torch
os.environ['CUDA_VISIBLE_DEVICES']=''
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from transformers import AutoTokenizer
SEED=1357; TOK='/root/models/qwen3.8-27b'; D='/workspace/work/data'; OUT=D+'/conf2'; L=1024; N=50; SKIP=12000; VOCAB=248320
COMP='/workspace/work/competitors/arki05/eval'
tok=AutoTokenizer.from_pretrained(TOK)
ids=lambda t: tok(t,add_special_tokens=False)['input_ids']
def stream(*a,**k): return load_dataset(*a,streaming=True,**k)
# ---- earlier sets
old=[]
for p in ['corpus.jsonl','big.jsonl','cal2/cal2.jsonl']: old+=[json.loads(l) for l in open(f'{D}/{p}')]
lm=json.load(open(f'{D}/long_manifest.json'))
old_sids={r['source_id'] for r in old}|{m['source_id'] for m in lm['sources']}
old_starts={r['text'][:300] for r in old}; old_hash={hashlib.sha1(r['text'].encode()).hexdigest() for r in old}
lt=torch.load(f'{D}/long_tokens.pt')['long']
long_starts=[tok.decode(r[:150].tolist())[:200] for r in lt]
old_queries=set()
for r in old:
    for m in re.finditer(r'Problem: (.*?)\nSolution:',r['text'],re.S): old_queries.add(m.group(1).strip())
old_wiki={s for s in old_sids if s.startswith('wikipedia_')}
old_pids=set()
for s in old_sids:
    if s.startswith('ultrachat/'): old_pids|=set(re.findall(r'[0-9a-f]{64}',s))
old_paths={s for s in old_sids if s.startswith('/')}
old_arx={int(s.rsplit('/',1)[1]) for s in old_sids if s.startswith('ccdv/arxiv')}
def clash(sid,text):
    if sid in old_sids: return True
    s=text[:300]
    if s in old_starts or any(s[:200]==p for p in long_starts): return True
    return hashlib.sha1(text.encode()).hexdigest() in old_hash
# ---- competitor
W=json.load(open(f'{COMP}/windows.json'))['windows']
comp_docids={d['id'] for w in W for d in w['documents']}
words=lambda t: re.findall(r'\w+',t.lower())
def grams(t,n=13):
    w=words(t); return {hash(tuple(w[i:i+n])) for i in range(len(w)-n+1)}
comp_g=set()
for l in open(f'{COMP}/text.jsonl'): comp_g|=grams(json.loads(l)['text'])
dummy_tool=[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{}}}}]
boiler=set()
for msgs,kw in [([{"role":"user","content":""}],{}),([{"role":"user","content":""},{"role":"assistant","content":""}],{}),
  ([{"role":"user","content":""}],{"tools":dummy_tool}),
  ([{"role":"user","content":""},{"role":"assistant","content":"","reasoning_content":""}],{}),
  ([{"role":"user","content":""},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},{"role":"tool","content":""}],{"tools":dummy_tool})]:
    boiler|=grams(tok.apply_chat_template(msgs,tokenize=False,**kw))
comp_bad_set=comp_g-boiler
def comp_shared(t): return len(grams(t)&comp_bad_set)
def mostly_list(t):
    ls=[l for l in t.split('\n') if l.strip()]
    return not ls or sum(len(l)<40 for l in ls)/len(ls)>0.5
def mk(sid,text,lang,ds,lic,dom,i): return dict(source_id=sid,lang=lang,dataset=ds,license=lic,domain=dom,ids=i[:L])
def accept(sid,text,i):
    """final per-doc checks on the 1024-token truncation"""
    if len(i)<L or sid in comp_docids: return False
    ct=tok.decode(i[:L])
    return comp_shared(ct)==0
WL='CC-BY-SA-3.0 and GFDL'
def wiki(cfg,lang,n,minc,buf):
    ds=stream('wikimedia/wikipedia',cfg,split='train').shuffle(seed=SEED,buffer_size=buf); out=[]
    for j,r in enumerate(ds):
        if j<SKIP: continue
        if len(r['text'])>=minc and not mostly_list(r['text'][:3000]):
            sid=f'wikipedia_{lang}/'+r['id']
            if sid in old_sids or clash(sid,r['text']): continue
            i=ids(r['text'])
            if accept(sid,r['text'],i): out.append(mk(sid,r['text'],lang,'wikimedia/wikipedia:'+cfg,WL,'en_prose' if lang=='en' else 'multiling',i))
        if len(out)>=n: break
    return out
def en(): return wiki('20231101.en','en',N,5000,5000)
langs=['fr','zh','de','es','ja','ar','ru']
quota={lg:(8 if k<1 else 7) for k,lg in enumerate(langs)}
def multiling():
    def one(lg): return wiki('20231101.'+lg,lg,quota[lg],{'zh':1500,'ja':2000}.get(lg,4500),3000)
    with ThreadPoolExecutor(7) as ex: res=list(ex.map(one,langs))
    out=[]
    for k in range(8):
        for r in res:
            if k<len(r): out.append(r[k])
    return out
def code():
    files=sorted(glob.glob('/usr/lib/python3.12/**/*.py',recursive=True))+sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/**/*.py',recursive=True))
    a=[f for f in files if f.startswith('/usr/lib/')]; b=[f for f in files if not f.startswith('/usr/lib/')]
    r=random.Random(SEED); r.shuffle(a); r.shuffle(b); out=[]; seen=set()
    for f in a+b:
        if f in old_paths or f in old_sids: continue
        try: t=open(f,encoding='utf-8').read()
        except Exception: continue
        if len(t)<6000: continue
        h=hashlib.sha1(t.encode()).hexdigest()
        if h in seen or clash(f,t): continue
        i=ids(t)
        if not accept(f,t,i): continue
        seen.add(h)
        lic,dsn=('PSF License (CPython stdlib)','local:python3.12-stdlib') if f.startswith('/usr/lib/') else ('see package metadata','local:dist-packages')
        out.append(mk(f,t,'python',dsn,lic,'code',i))
        if len(out)>=N: break
    return out
def math():
    ds=stream('meta-math/MetaMathQA',split='train').shuffle(seed=SEED,buffer_size=20000)
    out=[];cur=[];ct=0
    for idx,r in enumerate(ds):
        if idx<SKIP: continue
        q=r['query'].strip()
        if q in old_queries: continue
        s=f"Problem: {q}\nSolution: {r['response'].strip()}\n\n"
        cur.append((idx,s)); ct+=len(ids(s))
        if ct>=L+50:
            t=''.join(s for _,s in cur); sid='metamathqa/s1357/'+'+'.join(str(a) for a,_ in cur)
            if not clash(sid,t):
                i=ids(t)
                if accept(sid,t,i): out.append(mk(sid,t,'en','meta-math/MetaMathQA:train','MIT','math_reasoning',i))
            cur=[];ct=0
        if len(out)>=N: break
    return out
def chat():
    ds=stream('HuggingFaceH4/ultrachat_200k',split='train_sft').shuffle(seed=SEED,buffer_size=5000); out=[]
    for j,r in enumerate(ds):
        if j<SKIP: continue
        if r['prompt_id'] in old_pids: continue
        sid='ultrachat/'+r['prompt_id']; t=tok.apply_chat_template(r['messages'],tokenize=False)
        if clash(sid,t): continue
        i=ids(t)
        if accept(sid,t,i): out.append(mk(sid,t,'en','HuggingFaceH4/ultrachat_200k:train_sft','MIT','chat',i))
        if len(out)>=N: break
    return out
def sci():
    # unshuffled arXiv stream; read only parquet shards starting past row 40000 (bandwidth limit), global index from footers
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs=HfFileSystem(); name='ccdv/arxiv-summarization'
    shards=[f'datasets/{name}/document/train-{k:05d}-of-00015.parquet' for k in range(15)]
    off=0; start=None
    for k,p in enumerate(shards):
        n=pq.ParquetFile(fs.open(p)).metadata.num_rows
        if off>=40000: start=(k,off); break
        off+=n
    k0,off0=start; print('arxiv start shard',k0,'global offset',off0,flush=True)
    out=[]; off=off0
    for k in range(k0,15):
        url=f'hf://datasets/{name}/document/train-{k:05d}-of-00015.parquet'
        ds=stream('parquet',data_files=url,split='train')
        for j,r in enumerate(ds):
            g=off+j; sid=f'{name}/unshuffled/{g}'; t=r['article']
            if len(t)<10000 or sid in old_sids or g in old_arx or clash(sid,t): continue
            i=ids(t)
            if accept(sid,t,i): out.append(mk(sid,t,'en',name+':train','unknown/arXiv (dataset card apache-2.0; papers retain author licenses)','science_tech',i))
            if len(out)>=N: return out
        off+=pq.ParquetFile(fs.open(shards[k])).metadata.num_rows
    return out
DOMS=['en_prose','multiling','code','math_reasoning','chat','science_tech']
fns=dict(en_prose=en,multiling=multiling,code=code,math_reasoning=math,chat=chat,science_tech=sci)
with ThreadPoolExecutor(6) as ex: futs={d:ex.submit(fns[d]) for d in DOMS}
res={d:futs[d].result() for d in DOMS}
for d in DOMS: print(d,len(res[d]),flush=True)
rows=[];T=[];M=[]
for k in range(N):
    for d in DOMS:
        if k>=len(res[d]): continue
        x=res[d][k]; did=f'conf2_{d}_{k:03d}'
        rows.append(dict(id=did,source_id=x['source_id'],split='conf2',domain=d,lang=x['lang'],dataset=x['dataset'],license=x['license'],text=tok.decode(x['ids'])))
        T.append(x['ids']); M.append(dict(id=did,source_id=x['source_id'],domain=d,lang=x['lang'],dataset=x['dataset'],license=x['license']))
torch.save({'conf2':torch.tensor(T,dtype=torch.long),'meta':M},f'{OUT}/conf2_tokens.pt')
with open(f'{OUT}/conf2.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
sh=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
# ---- verification by reload
dd=torch.load(f'{OUT}/conf2_tokens.pt'); X=dd['conf2']; R=[json.loads(l) for l in open(f'{OUT}/conf2.jsonl')]
cnt={d:sum(r['domain']==d for r in R) for d in DOMS}
lcnt={}
for r in R:
    if r['domain']=='multiling': lcnt[r['lang']]=lcnt.get(r['lang'],0)+1
qs=[m.group(1).strip() for r in R if r['domain']=='math_reasoning' for m in re.finditer(r'Problem: (.*?)\nSolution:',r['text'],re.S)]
chk=dict(
 source_id_collisions=sum(r['source_id'] in old_sids for r in R),
 text_start_300_collisions=sum(r['text'][:300] in old_starts or any(r['text'][:200]==p for p in long_starts) for r in R),
 identical_text_collisions=sum(hashlib.sha1(r['text'].encode()).hexdigest() in old_hash for r in R),
 code_path_collisions=sum(r['source_id'] in old_paths for r in R if r['domain']=='code'),
 wikipedia_article_collisions=sum(r['source_id'] in old_wiki for r in R if r['source_id'].startswith('wikipedia_')),
 ultrachat_prompt_id_collisions=sum(r['source_id'][10:] in old_pids for r in R if r['domain']=='chat'),
 arxiv_index_collisions=sum(int(r['source_id'].rsplit('/',1)[1]) in old_arx for r in R if r['domain']=='science_tech'),
 math_problem_collisions=sum(q in old_queries for q in qs), math_problems_checked=len(qs),
 competitor_window_doc_id_collisions=sum(r['source_id'] in comp_docids for r in R),
 competitor_13gram_shared_docs_excl_boilerplate=sum(comp_shared(r['text'])>0 for r in R),
 retokenized_text_matches_tokens=sum(ids(r['text'])[:L]==X[k].tolist() for k,r in enumerate(R)),
 duplicate_texts_within=len(R)-len({r['text'] for r in R}), duplicate_source_ids_within=len(R)-len({r['source_id'] for r in R}))
dsl={d:dict(dataset=res[d][0]['dataset'] if res[d] else None,license=res[d][0]['license'] if res[d] else None) for d in DOMS}
man=dict(seed=SEED,tokenizer=TOK,tokens_per_doc=L,skip_items_per_stream=SKIP,arxiv_start_index_min=40000,datasets=dsl,counts_per_domain=cnt,counts_per_language_multiling=lcnt,
 shape=list(X.shape),min_token_id=int(X.min()),max_token_id=int(X.max()),vocab_limit=VOCAB,
 sha256={'conf2_tokens.pt':sh(f'{OUT}/conf2_tokens.pt'),'conf2.jsonl':sh(f'{OUT}/conf2.jsonl')},overlap_check=chk)
json.dump(man,open(f'{OUT}/conf2_manifest.json','w'),indent=1)
print(json.dumps(man,indent=1))
