import os, json, random, hashlib, glob, torch, itertools
os.environ['CUDA_VISIBLE_DEVICES']=''
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from transformers import AutoTokenizer
SEED=9876; TOK='/root/models/qwen3.8-27b'; OUT='/workspace/work/data'; L=1024; N=100; SKIP=5000
tok=AutoTokenizer.from_pretrained(TOK)
ids=lambda t: tok(t,add_special_tokens=False)['input_ids']
old=[json.loads(l) for l in open(f'{OUT}/corpus.jsonl')]
old_sids={r['source_id'] for r in old}; old_starts={r['text'][:300] for r in old}
old_texts=[r['text'] for r in old]
lm=json.load(open(f'{OUT}/long_manifest.json'))
long_sids={m['source_id'] for m in lm['sources']}
lt=torch.load(f'{OUT}/long_tokens.pt')['long']
long_starts=[tok.decode(r[:150].tolist())[:200] for r in lt]
used_paths=old_sids|long_sids
def clash(sid,text):
    if sid in old_sids or sid in long_sids: return True
    s=text[:300]
    if s in old_starts: return True
    if any(s[:200]==p for p in long_starts): return True
    return False
def mostly_list(t):
    ls=[l for l in t.split('\n') if l.strip()]
    return not ls or sum(len(l)<40 for l in ls)/len(ls)>0.5
def stream(*a,**k): return load_dataset(*a,streaming=True,**k)
def mk(sid,text,lang,ds,lic,dom):
    return dict(source_id=sid,text=text,lang=lang,dataset=ds,license=lic,domain=dom)
WL='CC-BY-SA-3.0 and GFDL'
def wiki(cfg,lang,n,minc,dom,buf):
    ds=stream('wikimedia/wikipedia',cfg,split='train').shuffle(seed=SEED,buffer_size=buf)
    out=[]
    for j,r in enumerate(ds):
        if j<SKIP: continue
        if len(r['text'])>=minc and not mostly_list(r['text'][:3000]):
            sid=f'wikipedia_{lang}/'+r['id']
            if clash(sid,r['text']): continue
            i=ids(r['text'])
            if len(i)>=L: out.append((mk(sid,r['text'],lang,'wikimedia/wikipedia:'+cfg,WL,dom),i[:L]))
        if len(out)>=n: break
    return out
def en(): return wiki('20231101.en','en',N,5000,'en_prose',5000)
langs=['fr','zh','de','es','ja','ar','ru']
quota={lg:(15 if k<2 else 14) for k,lg in enumerate(langs)}
def ml_lang(lg):
    mn={'zh':1500,'ja':2000}.get(lg,4500)
    return wiki('20231101.'+lg,lg,quota[lg],mn,'multiling',3000)
def multiling():
    with ThreadPoolExecutor(7) as ex: res=list(ex.map(ml_lang,langs))
    out=[]
    for k in range(15):
        for lg,r in zip(langs,res):
            if k<len(r): out.append(r[k])
    return out
def code():
    files=sorted(glob.glob('/usr/lib/python3.12/**/*.py',recursive=True))
    files+=sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/**/*.py',recursive=True))
    a=[f for f in files if f.startswith('/usr/lib/')]; b=[f for f in files if not f.startswith('/usr/lib/')]
    r=random.Random(SEED); r.shuffle(a); r.shuffle(b)
    out=[];seen=set()
    for f in a+b:
        if f in used_paths: continue
        try: t=open(f,encoding='utf-8').read()
        except Exception: continue
        if len(t)<6000: continue
        h=hashlib.sha1(t.encode()).hexdigest()
        if h in seen or clash(f,t): continue
        i=ids(t)
        if len(i)<L: continue
        seen.add(h)
        if f.startswith('/usr/lib/'): lic,dsn='PSF License (CPython stdlib)','local:python3.12-stdlib'
        else: lic,dsn='see package metadata','local:dist-packages'
        out.append((mk(f,t,'python',dsn,lic,'code'),i[:L]))
        if len(out)>=N: break
    return out
def math():
    ds=stream('meta-math/MetaMathQA',split='train').shuffle(seed=SEED,buffer_size=20000)
    out=[];cur=[];ct=0
    for idx,r in enumerate(ds):
        if idx<SKIP: continue
        s=f"Problem: {r['query'].strip()}\nSolution: {r['response'].strip()}\n\n"
        cur.append((idx,s)); ct+=len(ids(s))
        if ct>=L+50:
            t=''.join(s for _,s in cur); sid='metamathqa/s9876/'+'+'.join(str(a) for a,_ in cur)
            # underlying-item check: no problem line may appear in an old text
            qs=[s.split('\nSolution:')[0] for _,s in cur]
            bad=clash(sid,t) or any(any(q in ot for ot in old_texts if ot.startswith('Problem:') or 'Problem:' in ot) for q in qs)
            if not bad:
                i=ids(t)
                if len(i)>=L: out.append((mk(sid,t,'en','meta-math/MetaMathQA:train','MIT','math_reasoning'),i[:L]))
            cur=[];ct=0
        if len(out)>=N: break
    return out
def chat():
    ds=stream('HuggingFaceH4/ultrachat_200k',split='train_sft').shuffle(seed=SEED,buffer_size=5000)
    out=[]
    for j,r in enumerate(ds):
        if j<SKIP: continue
        sid='ultrachat/'+r['prompt_id']
        t=tok.apply_chat_template(r['messages'],tokenize=False)
        if clash(sid,t): continue
        i=ids(t)
        if len(i)>=L: out.append((mk(sid,t,'en','HuggingFaceH4/ultrachat_200k:train_sft','MIT','chat'),i[:L]))
        if len(out)>=N: break
    return out
def sci():
    name='ccdv/arxiv-summarization'
    ds=stream(name,split='train')
    out=[]
    for j,r in enumerate(ds):
        if j<SKIP: continue
        sid=f'{name}/unshuffled/{j}'; t=r['article']
        if len(t)<10000 or clash(sid,t): continue
        i=ids(t)
        if len(i)>=L: out.append((mk(sid,t,'en',name+':train','unknown/arXiv (dataset card apache-2.0; papers retain author licenses)','science_tech'),i[:L]))
        if len(out)>=N: break
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
        x,i=res[d][k]; did=f'big_{d}_{k:03d}'
        rows.append(dict(id=did,source_id=x['source_id'],split='big',domain=d,lang=x['lang'],dataset=x['dataset'],license=x['license'],text=tok.decode(i)))
        T.append(i); M.append(dict(id=did,source_id=x['source_id'],domain=d,lang=x['lang'],dataset=x['dataset'],license=x['license']))
torch.save({'big':torch.tensor(T,dtype=torch.long),'meta':M},f'{OUT}/big_tokens.pt')
with open(f'{OUT}/big.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
sh=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
# overlap checks
sid_coll=sum(r['source_id'] in old_sids or r['source_id'] in long_sids for r in rows)
start_coll=sum(r['text'][:300] in old_starts for r in rows)
dup=len(rows)-len({hashlib.sha1(r['text'].encode()).hexdigest() for r in rows}); dup_sid=len(rows)-len({r['source_id'] for r in rows})
cnt={d:sum(r['domain']==d for r in rows) for d in DOMS}
lcnt={}
for r in rows:
    if r['domain']=='multiling': lcnt[r['lang']]=lcnt.get(r['lang'],0)+1
dsl={d:dict(dataset=res[d][0][0]['dataset'] if res[d] else None,license=res[d][0][0]['license'] if res[d] else None) for d in DOMS}
json.dump(dict(seed=SEED,tokenizer=TOK,tokens_per_doc=L,skip_items_per_stream=SKIP,datasets=dsl,counts_per_domain=cnt,counts_per_language_multiling=lcnt,
 sha256={'big_tokens.pt':sh(f'{OUT}/big_tokens.pt'),'big.jsonl':sh(f'{OUT}/big.jsonl')},
 overlap_check=dict(source_id_collisions_with_corpus_and_long=sid_coll,text_start_collisions=start_coll,duplicate_texts_within_big=dup,duplicate_source_ids_within_big=dup_sid)),open(f'{OUT}/big_manifest.json','w'),indent=1)
print(cnt,lcnt,sid_coll,start_coll,dup,dup_sid)
