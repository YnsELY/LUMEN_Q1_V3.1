from __future__ import annotations
import math,time
import numpy as np
import torch
from .common import integer,finite

def logits_metrics(teacher,student):
    a=np.asarray(teacher,dtype=np.float64);b=np.asarray(student,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite logits [positions,vocab] required')
    def logsoftmax(x):
        x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
    p,q=logsoftmax(a),logsoftmax(b);kl=np.sum(np.exp(p)*(p-q),-1)
    if kl.min() < -1e-10:raise ArithmeticError('Significantly negative KL')
    kl=np.maximum(kl,0)
    return {'kl_nats_mean':float(kl.mean()),'kl_nats_p99':float(np.quantile(kl,.99)),
            'top1_agreement':float(np.mean(a.argmax(-1)==b.argmax(-1))),'positions':len(kl)}

def relative_mse(reference,candidate):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(candidate,dtype=np.float64)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid outputs')
    e=float(np.sum((a-b)**2));den=float(np.sum(a*a))
    return {'squared_error':e,'reference_energy':den,'nmse':e/den if den else (0. if e==0 else None)}

def _cluster_ci(values,sources,replicates,seed):
    groups={}
    for value,source in zip(values,sources):groups.setdefault(source,[]).append(value)
    if len(groups)<2:return None,len(groups)
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(seed);ix=rng.integers(len(groups),size=(integer(replicates,'replicates',1),len(groups)))
    means=sums[ix].sum(1)/counts[ix].sum(1)
    return np.quantile(means,[.025,.975]).tolist(),len(groups)

def document_summary(rows,replicates=500,seed=1):
    if not rows:raise ValueError('No documents')
    vals=np.array([r['kl_nats_mean'] for r in rows]);weights=np.array([r['positions'] for r in rows])
    if not np.isfinite(vals).all() or np.any(weights<=0):raise ValueError('Invalid metric rows')
    sources=[r.get('source_id',r.get('id',str(i))) for i,r in enumerate(rows)]
    ci,clusters=_cluster_ci(vals,sources,replicates,seed)
    return {'documents':len(rows),'positions':int(weights.sum()),'document_mean_kl':float(vals.mean()),
            'token_weighted_mean_kl':float(vals@weights/weights.sum()),'bootstrap_document_mean_95pct':ci,
            'bootstrap_unit':'source_id (documents clustered)','source_clusters':clusters,
            'bootstrap_note':'Descriptive interval resampling sources/episodes; not a capacity guarantee.'}

def paired_difference(control,candidate,replicates=500,seed=2):
    a={r['id']:r for r in control};b={r['id']:r for r in candidate}
    if a.keys()!=b.keys() or len(a)!=len(control) or len(b)!=len(candidate):raise ValueError('Paired document ids differ or repeat')
    dif=[];sources=[]
    for key in sorted(a):
        if a[key]['token_hash']!=b[key]['token_hash'] or a[key]['positions']!=b[key]['positions'] or a[key].get('source_id')!=b[key].get('source_id'):
            raise ValueError('Histories/positions/sources differ')
        dif.append(b[key]['kl_nats_mean']-a[key]['kl_nats_mean']);sources.append(a[key].get('source_id',key))
    if not dif:raise ValueError('No paired documents')
    v=np.asarray(dif);ci,clusters=_cluster_ci(v,sources,replicates,seed)
    return {'candidate_minus_control_document_kl':float(v.mean()),'bootstrap95':ci,'documents':len(v),
            'source_clusters':clusters,'bootstrap_unit':'source_id'}


def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)

def evaluate_documents(bundle,records,reference,keep_last=8):
    rows=[]
    with torch.inference_mode():
        for r in records:
            count=min(keep_last,len(r['input_ids']))
            dev=bundle.model.get_input_embeddings().weight.device;synchronize(dev);t=time.perf_counter()
            z=bundle.forward(r['input_ids'],keep_last=count);synchronize(dev);elapsed=time.perf_counter()-t
            values=logits_metrics(reference[r['id']],z.detach().float().cpu().numpy())
            rows.append({**values,'id':r['id'],'source_id':r['source_id'],'token_hash':r['token_hash'],
                         'elapsed_forward_seconds':elapsed,'input_tokens':len(r['input_ids']),
                         'teacher_forced_prefill_input_tokens_per_second':len(r['input_ids'])/elapsed,
                         'not_autoregressive_generation_throughput':True})
    return rows
