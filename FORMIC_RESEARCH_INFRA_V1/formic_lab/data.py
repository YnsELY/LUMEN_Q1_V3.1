"""Small batch-one document protocol. No benchmark or corpus downloaded."""
from __future__ import annotations
import json, re
from pathlib import Path
from .common import digest,integer,read_json,write_json
SPLITS=('cal','select','confirm')

def validate_records(records):
    if not records:raise ValueError('Empty corpus')
    ids=set();sources={s:set() for s in SPLITS};contents={}
    for r in records:
        if not isinstance(r,dict):raise ValueError('Record must be an object')
        for key in ('id','source_id','split','license'):
            if not isinstance(r.get(key),str) or not r[key].strip():raise ValueError('Missing '+key)
        if r['id'] in ids:raise ValueError('Duplicate document id')
        ids.add(r['id']);s=r['split']
        if s not in SPLITS:raise ValueError('Unknown split')
        sources[s].add(r['source_id'])
        modes=[k for k in ('text','input_ids') if k in r]
        if len(modes)!=1:raise ValueError('Provide exactly one of text or input_ids')
        if modes[0]=='text':
            if not isinstance(r['text'],str) or not r['text'].strip():raise ValueError('Empty text')
            norm=' '.join(r['text'].split());h=digest(norm)
        else:
            if not isinstance(r['input_ids'],list) or len(r['input_ids'])<2:raise ValueError('Need >=2 tokens')
            for t in r['input_ids']:integer(t,'token id')
            h=digest(r['input_ids'])
        if h in contents and contents[h]!=s:raise ValueError('Exact duplicate content across splits')
        contents[h]=s
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if sources[a]&sources[b]:raise ValueError('Source shared across splits')
    return {'documents':len(records),'splits':{s:sum(r['split']==s for r in records) for s in SPLITS},
            'semantic_decontamination_verified':False,'fingerprint':digest(records)}

def load_records(path,max_bytes=64_000_000):
    p=Path(path)
    if p.stat().st_size>integer(max_bytes,"max_bytes",1):raise MemoryError("Corpus loader cap; stream a larger corpus explicitly")
    if p.suffix=='.jsonl':
        records=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:records=read_json(p)
    return records,validate_records(records)

def encode_records(records,tokenizer=None,max_length=256,vocab_size=None):
    max_length=integer(max_length,'max_length',2);out=[]
    for r in records:
        if 'input_ids' in r:ids=r['input_ids']
        else:
            if tokenizer is None:raise ValueError('Real tokenizer required for text')
            ids=tokenizer(r['text'],add_special_tokens=True,truncation=True,max_length=max_length)['input_ids']
        ids=list(ids[:max_length])
        if len(ids)<2:raise ValueError('Document tokenized to fewer than two tokens')
        if any(type(t)!=int or t<0 or (vocab_size is not None and t>=vocab_size) for t in ids):
            raise ValueError('Token outside model vocabulary')
        out.append({**r,'input_ids':ids,'used_tokens':len(ids),'token_hash':digest(ids)})
    # Detect different texts producing the same truncated sequence across splits.
    check={}
    for r in out:
        h=r['token_hash']
        if h in check and check[h]!=r['split']:raise ValueError('Tokenized/truncated duplicate across splits')
        check[h]=r['split']
    return out

def fixture_records(seed=123,docs_per_split=4,length=24,vocab=64):
    import numpy as np
    rng=np.random.default_rng(seed);out=[]
    for split in SPLITS:
        for j in range(docs_per_split):
            out.append({'id':f'{split}-{j}','source_id':f'synthetic-{split}-{j}','split':split,
                        'license':'original synthetic fixture; no model capability claim',
                        'input_ids':rng.integers(0,vocab,size=length).tolist()})
    return out
