import json, random, hashlib, glob, torch
from datasets import load_dataset
from transformers import AutoTokenizer
SEED=4321; TOK='/root/models/qwen3.8-27b'; OUT='/workspace/work/data'; L=8192; K=4; SKIP=3000
random.seed(SEED)
tok=AutoTokenizer.from_pretrained(TOK)
ids=lambda t: tok(t,add_special_tokens=False)['input_ids']
old=[json.loads(l) for l in open(f'{OUT}/corpus.jsonl')]
old_sids={r['source_id'] for r in old}; old_starts={r['text'][:300] for r in old}
old_text_pref=[r['text'][:300] for r in old]
def clash(sid,text):
    return sid in old_sids or text[:300] in old_starts or any(text.startswith(p) or p.startswith(text[:300]) for p in old_text_pref)
docs={'code':[],'en_prose':[],'science_tech':[]}
# code
files=sorted(glob.glob('/usr/lib/python3.12/**/*.py',recursive=True)); random.Random(SEED).shuffle(files)
seen=set()
for f in files:
    if len(docs['code'])>=K: break
    try: t=open(f,encoding='utf-8').read()
    except Exception: continue
    if len(t)<25000 or clash(f,t): continue
    h=hashlib.sha1(t.encode()).hexdigest()
    if h in seen: continue
    i=ids(t)
    if len(i)>=L: seen.add(h); docs['code'].append((f,i[:L],'local:python3.12-stdlib','PSF License (CPython stdlib)'))
def streamsel(dom,name,cfg,fld,idf,lic,minc,pref):
    a=(name,cfg) if cfg else (name,)
    ds=load_dataset(*a,split='train',streaming=True)
    for j,r in enumerate(ds):
        if j<SKIP: continue
        if len(docs[dom])>=K: break
        t=r[fld]
        if len(t)<minc: continue
        sid=pref+str(idf(r,j))
        if clash(sid,t): continue
        i=ids(t)
        if len(i)>=L: docs[dom].append((sid,i[:L],name+(':'+cfg if cfg else ''),lic))
streamsel('en_prose','wikimedia/wikipedia','20231101.en','text',lambda r,j:r['id'],'CC-BY-SA-3.0 and GFDL',30000,'wikipedia_en/')
streamsel('science_tech','ccdv/arxiv-summarization',None,'article',lambda r,j:j,'unknown/arXiv (dataset card apache-2.0; papers retain author licenses)',30000,'ccdv/arxiv-summarization/unshuffled/')
rows=[];meta=[]
for k in range(K):
    for d in ['code','en_prose','science_tech']:
        sid,i,ds_,lic=docs[d][k]; rows.append(i); meta.append(dict(id=f'long_{d}_{k:02d}',source_id=sid,domain=d,dataset=ds_,license=lic))
T=torch.tensor(rows,dtype=torch.long)
torch.save({'long':T,'meta':meta},f'{OUT}/long_tokens.pt')
# overlap check on decoded text too
ok=all(m['source_id'] not in old_sids for m in meta) and len({m['source_id'] for m in meta})==12
for r in rows:
    s=tok.decode(r[:150])[:200]
    ok&=not any(p[:200]==s for p in old_text_pref)
sha=hashlib.sha256(open(f'{OUT}/long_tokens.pt','rb').read()).hexdigest()
json.dump(dict(sources=meta,tokenizer=TOK,seed=SEED,tokens_per_doc=L,skip=SKIP,sha256_long_tokens_pt=sha,overlap_check=dict(passed=bool(ok),checked_against='corpus.jsonl all splits: source_id + first 300 chars')),open(f'{OUT}/long_manifest.json','w'),indent=1)
print('overlap ok',ok)
