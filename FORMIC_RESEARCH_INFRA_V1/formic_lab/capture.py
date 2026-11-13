"""Bounded forward hooks; whole feature width, batch one, no fake deep activations."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from .common import integer,digest,save_npz,write_json,sha256,load_npz,read_json

@dataclass
class Capture:
    x: np.ndarray  # [positions, full input width]
    y: np.ndarray  # [positions, full output width]
    meta: dict

def _sequence(t,name):
    if not torch.is_tensor(t):raise TypeError(f'{name} must be a tensor; provide a specialized collector')
    if t.ndim==3 and t.shape[0]==1:return t[0]
    if t.ndim==2:return t
    raise ValueError(f'{name} shape {tuple(t.shape)} unsupported; no silent flatten of states/heads')

def capture_module(bundle,path,records,*,max_positions=256,per_document=64,seed=0):
    max_positions=integer(max_positions,'max_positions',1);per_document=integer(per_document,'per_document',1)
    module=bundle.model.get_submodule(path);xs=[];ys=[];ids=[];positions=[];sources=[];calls={};rows=0
    active=None;t0=time.perf_counter()
    def hook(mod,args,kwargs,output):
        nonlocal rows
        calls[active['id']]=calls.get(active['id'],0)+1
        if calls[active['id']]>1:raise ValueError('Target executed more than once per forward. Extend collector with explicit call/state identity.')
        x=args[0] if args else kwargs.get('hidden_states',kwargs.get('input'))
        y=output if torch.is_tensor(output) else output[0] if isinstance(output,(tuple,list)) else None
        x=_sequence(x,'input');y=_sequence(y,'output')
        if x.shape[0]!=y.shape[0] or x.shape[0]!=len(active['input_ids']):
            raise ValueError('Sequence alignment differs; inspect fused/multimodal runtime')
        remaining=max_positions-rows;k=min(per_document,remaining,len(x))
        # Stable across variants: sampling depends only on document identity.
        ds=int(digest({'seed':seed,'doc':active['id']})[:16],16)
        choose=np.sort(np.random.default_rng(ds).choice(len(x),size=k,replace=False))
        ix=torch.tensor(choose,device=x.device,dtype=torch.long)
        a=x.index_select(0,ix).detach().float().cpu().numpy()
        b=y.index_select(0,ix.to(y.device)).detach().float().cpu().numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite forward capture')
        xs.append(a);ys.append(b);rows+=k
        for pos in choose:
            ids.append(active['id']);positions.append(int(pos));sources.append(active['source_id'])
    handle=module.register_forward_hook(hook,with_kwargs=True)
    try:
        with torch.inference_mode():
            for r in records:
                if rows>=max_positions:break
                active=r;bundle.forward(r['input_ids'],keep_last=1)
                if calls.get(r['id'],0)!=1:raise ValueError(f'Target {path} not executed. No fabricated capture.')
    finally:handle.remove()
    if not xs:raise ValueError('No capture positions')
    meta={'schema':'formic_capture_v1','module_path':path,'module_type':type(module).__name__,
          'layout':'positions_features','full_input_width':xs[0].shape[1],
          'sample_ids':[f'{d}:{p}' for d,p in zip(ids,positions)],'document_ids':ids,'source_ids':sources,
          'token_positions':positions,'split_names':sorted(set(r['split'] for r in records)),
          'record_hashes':{r['id']:r['token_hash'] for r in records},
          'source_kind':'synthetic_fixture' if bundle.metadata['synthetic'] else 'captured_from_model_forward',
          'checkpoint':bundle.metadata['checkpoint'],'revision':bundle.metadata['revision'],
          'dtype_saved':'float32','context_policy':'one complete truncated document, use_cache=False; no state carry across documents',
          'provenance_independently_verified':False,'elapsed_seconds':time.perf_counter()-t0}
    return Capture(np.concatenate(xs),np.concatenate(ys),meta)

def save_capture(folder,capture):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save_npz(folder/'arrays.npz',X=capture.x,Y=capture.y)
    write_json(folder/'capture.json',{**capture.meta,'array_sha256':sha256(folder/'arrays.npz')})

def read_capture(folder):
    folder=Path(folder);m=read_json(folder/'capture.json')
    if sha256(folder/'arrays.npz')!=m['array_sha256']:raise ValueError('Capture hash mismatch')
    a=load_npz(folder/'arrays.npz');x,y=a['X'],a['Y']
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0] or not x.size or not y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid capture arrays')
    if len(m['sample_ids'])!=len(x) or len(set(m['sample_ids']))!=len(x):raise ValueError('Invalid position identities')
    return Capture(x,y,m)

def pair_captures(reference,student):
    for key in ('sample_ids','record_hashes','checkpoint','revision'):
        if reference.meta[key]!=student.meta[key]:raise ValueError('Capture pairs not aligned: '+key)
    if reference.x.shape!=student.x.shape:raise ValueError('Different bases/dimensions require explicit alignment')
    return reference.x.T,student.x.T,reference.y.T

def export_lumen_schema2(pairs,output,*,runtime_commit,architecture_id='A0',projection_kind='mlp_up',weight=None,bias=None):
    """Pairs are real reference/student Capture objects. Explicit bridge to V3.1."""
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True);arrays={};splits={};mapping={};base=None
    for split in ('cal','select','confirm'):
        ref,stu=pairs[split];xf,xq,y=pair_captures(ref,stu)
        if ref.meta['split_names']!=[split] or stu.meta['split_names']!=[split]:raise ValueError('Wrong split in pair')
        arrays['X_fp_'+split]=xf;arrays['X_q_'+split]=xq
        if weight is not None:arrays['Y_fp_'+split]=y
        splits[split]=sorted(set(ref.meta['source_ids']));mapping[split]=ref.meta['source_ids'];base=ref.meta
    if weight is not None:arrays['W_fp']=np.asarray(weight,dtype=np.float32)
    if bias is not None:arrays['bias_fp']=np.asarray(bias,dtype=np.float32)
    save_npz(folder/'capture.npz',**arrays)
    meta={'schema_version':2,'capture_kind':'linear' if weight is not None else 'input_pair',
          'checkpoint':base['checkpoint'],'revision':base['revision'],'runtime_commit':runtime_commit,
          'activation_path':base['module_path'],'full_input_width':arrays['X_fp_cal'].shape[0],
          'splits':splits,'sample_source_ids':mapping,'array_sha256':sha256(folder/'capture.npz'),
          'source_kind':base['source_kind'],'architecture_id':architecture_id,'projection_kind':projection_kind,
          'weight_status':'Reference and student capture metadata retained; inspect source captures',
          'provenance_independently_verified':False}
    write_json(folder/'capture.json',meta);return meta
