#!/usr/bin/env python3
"""Fail-closed capture schema validation. Does NOT prove provenance or run Qwen.

Schema 2 distinguishes document IDs from per-position source IDs. Equal Xfp/Xq
is legal (first layer/intact prefix). A declaration is not a proof of a forward.
"""
from pathlib import Path
import argparse, hashlib, json, re, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from lumen_ref.metrics import assert_disjoint_splits
EXPECTED_REV='37561816e66b5b9153b1b64c10968c18c1e6f4a0'


def _text(v, field):
    if not isinstance(v,str) or not v.strip() or any(t in v.upper() for t in ('A_REMPLIR','TODO','PLACEHOLDER','TO_FILL')):
        raise ValueError('Invalid or placeholder field '+field)
    return v


def _positive_int(v, field):
    if type(v) is not int or v<=0:
        raise ValueError(field+' must be a positive JSON integer')
    return v


def validate(npz_path:Path, manifest_path:Path, *, allow_synthetic=False):
    npz_path=Path(npz_path);manifest_path=Path(manifest_path)
    meta=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(meta,dict):raise ValueError('Manifest root must be an object')
    required=('schema_version','capture_kind','checkpoint','revision','runtime_commit',
              'activation_path','full_input_width','splits','sample_source_ids',
              'array_sha256','source_kind','architecture_id','weight_status')
    for key in required:
        if key not in meta:raise ValueError('Missing manifest field '+key)
    if type(meta['schema_version']) is not int or meta['schema_version']!=2:
        raise ValueError('Require capture schema_version=2')
    for key in ('checkpoint','revision','runtime_commit','activation_path','architecture_id','weight_status'):
        _text(meta[key],key)
    if not re.fullmatch(r'[0-9a-fA-F]{7,64}',meta['runtime_commit']):
        raise ValueError('runtime_commit must identify the executed code revision')
    synthetic=meta['source_kind']=='synthetic_fixture'
    if not (meta['source_kind']=='captured_from_model_forward' or (synthetic and allow_synthetic)):
        raise ValueError('Real forward required; synthetic fixtures need explicit opt-in')
    if not synthetic and (meta['checkpoint']!='Qwen/Qwen3.8-27B' or meta['revision']!=EXPECTED_REV):
        raise ValueError('Unexpected checkpoint; no silent substitution')
    if not isinstance(meta['array_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',meta['array_sha256']):
        raise ValueError('Invalid array checksum')
    digest=hashlib.sha256()
    with npz_path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):digest.update(part)
    if digest.hexdigest()!=meta['array_sha256']:raise ValueError('Capture checksum mismatch')
    splits=meta['splits'];mapping=meta['sample_source_ids']
    if not isinstance(splits,dict) or set(splits)!={'cal','select','confirm'}:
        raise ValueError('Invalid splits dictionary')
    if not isinstance(mapping,dict) or set(mapping)!=set(splits):raise ValueError('Invalid source mapping')
    for split,ids in splits.items():
        if not isinstance(ids,list) or not ids:raise ValueError('Nonempty source list required')
        for sid in ids:_text(sid,'source id')
        if len(set(ids))!=len(ids):raise ValueError('Duplicate source identity within split')
    assert_disjoint_splits(splits['cal'],splits['select'],splits['confirm'])
    n=_positive_int(meta['full_input_width'],'full_input_width')
    kind=meta['capture_kind']
    if kind not in ('input_pair','linear','swiglu'):raise ValueError('Unknown capture_kind')
    expected_width={'mlp_gate':5120,'mlp_up':5120,'mlp_down':17408,
                    'attention_input':5120,'attention_output':6144,
                    'deltanet_input':5120,'deltanet_output':6144,
                    'lm_head':5120,'swiglu':5120}
    if not synthetic and meta['architecture_id']=='A0':
        projection=meta.get('projection_kind','swiglu' if kind=='swiglu' else None)
        if projection not in expected_width:raise ValueError('Specify projection_kind for A0')
        if n!=expected_width[projection]:raise ValueError('Not the full input width of this projection')
    records={}
    with np.load(npz_path,allow_pickle=False) as arrays:
        def arr(name):
            if name not in arrays:raise ValueError('Missing array '+name)
            x=np.asarray(arrays[name])
            if not np.issubdtype(x.dtype,np.floating) or not np.all(np.isfinite(x)) or x.size==0:
                raise ValueError('Nonempty finite floating array required: '+name)
            return x
        for split in ('cal','select','confirm'):
            f=arr('X_fp_'+split);q=arr('X_q_'+split)
            if f.ndim!=2 or f.shape!=q.shape or f.shape[0]!=n or f.shape[1]==0:
                raise ValueError('Bad input shape for '+split)
            ids=mapping[split]
            if not isinstance(ids,list) or len(ids)!=f.shape[1] or any(not isinstance(i,str) or i not in splits[split] for i in ids):
                raise ValueError('One valid source identity is required PER POSITION')
            if set(ids)!=set(splits[split]):raise ValueError('Listed sources do not match captured positions')
            records[split]={'shape':list(f.shape),'dtype':str(f.dtype),'inputs_identical':bool(np.array_equal(f,q))}
            if kind!='input_pair':
                y=arr('Y_fp_'+split)
                if y.ndim!=2 or y.shape[1]!=f.shape[1]:raise ValueError('Invalid output target for '+split)
                records[split]['target_shape']=list(y.shape)
        if kind=='linear':
            w=arr('W_fp')
            if w.ndim!=2 or w.shape[1]!=n:raise ValueError('Invalid linear weight')
            for split in splits:
                if records[split]['target_shape'][0]!=w.shape[0]:raise ValueError('Output dimension mismatch')
            if 'bias_fp' in arrays and arr('bias_fp').shape!=(w.shape[0],):raise ValueError('Invalid bias')
        if kind=='swiglu':
            wg=arr('W_gate_fp');wu=arr('W_up_fp');d=arr('W_down_fp')
            if wg.ndim!=2 or wg.shape!=wu.shape or wg.shape[1]!=n or d.ndim!=2 or d.shape[1]!=wg.shape[0]:
                raise ValueError('Inconsistent SwiGLU weights')
            for split in splits:
                if records[split]['target_shape'][0]!=d.shape[0]:raise ValueError('SwiGLU output dimension mismatch')
    return {'schema_valid':True,'capture_kind':kind,'provenance_verified':False,
            'synthetic_fixture':synthetic,'arrays':records,
            'note':'Shapes/hash/source partition checked; original forward, token alignment, masks, norm/activation/state/MTP semantics still require an independent collector audit.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('npz',type=Path);p.add_argument('manifest',type=Path)
    p.add_argument('--allow-synthetic',action='store_true',help='Test fixtures only, never counts as a Qwen capture')
    a=p.parse_args();print(json.dumps(validate(a.npz,a.manifest,allow_synthetic=a.allow_synthetic),indent=2))
if __name__=='__main__':main()
