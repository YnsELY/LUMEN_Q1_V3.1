"""Read local safetensors headers without loading a 27B checkpoint.

A commit-like directory name is a declaration, NOT proof of publisher identity.
A provided trusted manifest can verify local file hashes; the caller owns its origin.
"""
from __future__ import annotations
import json, math, struct
from pathlib import Path
from .common import EXPECTED_REPO,EXPECTED_REVISION,read_json,sha256,digest
DTYPE_BYTES={'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'I16':2,'U16':2,'BF16':2,'F16':2,
             'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}

def header(path):
    path=Path(path);size=path.stat().st_size
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8: raise ValueError('Truncated safetensors header')
        n=struct.unpack('<Q',raw)[0]
        if n<2 or n>64*1024*1024 or n>size-8: raise ValueError('Invalid header length')
        data=f.read(n)
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('Duplicate header key')
            d[k]=v
        return d
    h=json.loads(data,object_pairs_hook=pairs);records=[];spans=[]
    for name,spec in h.items():
        if name=='__metadata__':continue
        shape=spec['shape'];dtype=spec['dtype'];a,b=spec['data_offsets']
        if not isinstance(shape,list) or any(type(x)!=int or x<0 for x in shape):raise ValueError('Invalid shape')
        if dtype not in DTYPE_BYTES:raise ValueError(f'Unknown dtype {dtype}; extend parser explicitly')
        if type(a)!=int or type(b)!=int or not 0<=a<=b<=size-8-n:raise ValueError('Invalid tensor offsets')
        count=math.prod(shape)
        if b-a!=count*DTYPE_BYTES[dtype]:raise ValueError('Inconsistent tensor size')
        if a!=b:spans.append((a,b))
        records.append({'name':name,'shape':shape,'dtype':dtype,'params':count,'bytes':b-a})
    spans.sort();end=0
    for a,b in spans:
        if a!=end:raise ValueError('Overlapping tensors or unaccounted data')
        end=b
    if end!=size-8-n:raise ValueError('Trailing unaccounted tensor bytes')
    return records,digest(h)

def inspect_checkpoint(path,identity=None,full_hash=False,expected_revision=EXPECTED_REVISION):
    root=Path(path).resolve()
    config=read_json(root/'config.json')
    idx=root/'model.safetensors.index.json'
    weight_map=read_json(idx).get('weight_map') if idx.exists() else None
    if weight_map is not None:
        if not isinstance(weight_map,dict) or not weight_map:raise ValueError('Invalid weight index')
        names=sorted(set(weight_map.values()))
    else:names=[p.name for p in sorted(root.glob('*.safetensors'))]
    if not names:raise ValueError('No local safetensors. No download is automatic.')
    all_records=[];headers={};file_meta={};by_name={}
    for name in names:
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.safetensors'):
            raise ValueError('Unsafe shard path')
        p=root/name
        rec,h=header(p);headers[name]=h
        file_meta[name]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        if full_hash:file_meta[name]['sha256']=sha256(p)
        for r in rec:
            if r['name'] in by_name:raise ValueError('Duplicate tensor across shards')
            by_name[r['name']]=name;r['shard']=name
        all_records.extend(rec)
    if weight_map is not None and weight_map!=by_name:raise ValueError('Index/header tensor mapping differs')
    declared=config.get('_commit_hash')
    if root.parent.name=='snapshots':declared=root.name
    evidence='local_config_or_snapshot_declaration' if declared else 'absent'
    trusted_hashes=False
    if identity:
        ident=read_json(identity)
        if ident.get('repo_id')!=EXPECTED_REPO or ident.get('revision')!=expected_revision:raise ValueError('Identity mismatch')
        hashes=ident.get('sha256',{})
        required=set(names)|{'config.json'}
        if idx.exists():required.add(idx.name)
        if not required.issubset(hashes):raise ValueError('Trusted manifest lacks all shards/config/index hashes')
        for name in sorted(required):
            if Path(name).name!=name or sha256(root/name)!=hashes[name]:raise ValueError('Checkpoint hash mismatch: '+name)
        declared=ident['revision'];evidence='hashes_match_supplied_identity_manifest';trusted_hashes=True
    if declared and declared!=expected_revision:raise ValueError(f'Revision {declared} != {expected_revision}')
    return {'root':str(root),'repo_expected':EXPECTED_REPO,'revision_expected':expected_revision,
            'revision_declared':declared,'identity_evidence':evidence,'trusted_manifest_hashes_checked':trusted_hashes,
            'publisher_provenance_verified':False,'config':config,'tensors':all_records,'headers':headers,'files':file_meta,
            'tensor_count':len(all_records),'parameter_count':sum(r['params'] for r in all_records),
            'weight_bytes':sum(r['bytes'] for r in all_records),
            'fingerprint':digest({'config':sha256(root/'config.json'),'headers':headers,'files':file_meta})}
