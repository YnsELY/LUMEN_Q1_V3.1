"""Strict small-matrix bridge to the UNMODIFIED Lumen Q1-v3.1 reference.

Dense decoded weights are used at runtime: an E8 file size is NOT runtime VRAM.
No silent uniform-quantization substitute is called Lumen.
"""
from __future__ import annotations
import sys,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import ROOT,integer,finite,atomic_bytes,write_json,sha256,load_plugin

VENDOR=ROOT/'vendor'
if str(VENDOR) not in sys.path:sys.path.insert(0,str(VENDOR))
import lumen_ref
if not Path(lumen_ref.__file__).resolve().is_relative_to(VENDOR.resolve()):
    raise RuntimeError('A different lumen_ref was already imported; use a clean process')
from lumen_ref.quantization import quantize_ldlq
from lumen_ref.entropy import decode_matrix


def quantize_linear(linear,x,folder,*,step_multiplier=.8,rate_lambda=0.,damping=.01,
                    max_elements=65536,max_input_width=512,max_output_width=256):
    if not isinstance(linear,nn.Linear):raise TypeError('Reference bridge supports nn.Linear only')
    m,n=linear.weight.shape
    if m*n>integer(max_elements,'max_elements',1) or n>integer(max_input_width,'max_input_width',8) or m>integer(max_output_width,'max_output_width',1):
        raise MemoryError('CPU Lumen oracle capped before dense Gram/factorization. Port or override after a timed pilot; no truncated-column substitution.')
    if n%8:raise ValueError('Input width must be divisible by 8 for this oracle; no hidden padding')
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=n or not len(x) or not np.isfinite(x).all():raise ValueError('Require real full-width input captures [positions,features]')
    damping=finite(damping,'damping',0)
    if damping==0:raise ValueError('Use a strictly positive explicit damping')
    w=linear.weight.detach().float().cpu().numpy().astype(np.float64)
    h=x.T@x/len(x);alpha=damping*max(float(np.trace(h)/n),1e-12);h+=np.eye(n)*alpha
    q=quantize_ldlq(w,h,step_multiplier=finite(step_multiplier,'step_multiplier',1e-12),rate_lambda=finite(rate_lambda,'rate_lambda',0))
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    atomic_bytes(out/'weight.lmat',q.artifact,overwrite=False)
    decoded,_,_,_=decode_matrix((out/'weight.lmat').read_bytes(),max_weights=w.size)
    result=nn.Linear(n,m,linear.bias is not None,device=linear.weight.device,dtype=linear.weight.dtype)
    with torch.no_grad():
        result.weight.copy_(torch.from_numpy(decoded).to(result.weight))
        if linear.bias is not None:result.bias.copy_(linear.bias)
    bias_bytes=0
    if linear.bias is not None:
        from safetensors.torch import save_file
        save_file({'bias':linear.bias.detach().cpu().contiguous()},str(out/'bias.safetensors'))
        bias_bytes=(out/'bias.safetensors').stat().st_size
    meta={'method':'LUMEN_Q1_v3.1_CPU_matrix_reference','shape':[m,n],'matrix_blob_bytes':len(q.artifact),
          'bias_file_bytes':bias_bytes,'encoded_tensor_files_bytes':len(q.artifact)+bias_bytes,
          'encoded_bpw_weight_blob_only':len(q.artifact)*8/(m*n),'runtime_weight_dtype':str(result.weight.dtype),
          'runtime_dense_parameter_bytes':sum(p.numel()*p.element_size() for p in result.parameters()),
          'gram_from':'full-width inputs of actual selected module','damping_absolute':alpha,
          'hadamard_applied':False,'historical_L0_reproduced':False,'gpu_codec_implemented':False,
          'runtime_cast_max_abs_error':float(np.max(np.abs(result.weight.detach().float().cpu().numpy()-decoded))),
          'weight_sha256':sha256(out/'weight.lmat')}
    write_json(out/'quantization.json',meta);return result,meta


def scalar_proxy(linear,x,folder,*,bits=4):
    """GPU-capable numerical screening ONLY. Dense saved tensors, not a codec."""
    bits=integer(bits,'bits',2)
    if bits>8:raise ValueError('Proxy bit range 2..8')
    if not isinstance(linear,nn.Linear):raise TypeError('Linear required')
    w=linear.weight.detach().float();qmax=(1<<(bits-1))-1
    scale=w.abs().amax(1,keepdim=True)/qmax;scale=torch.where(scale==0,torch.ones_like(scale),scale)
    reconstructed=(w/scale).round().clamp(-qmax,qmax)*scale
    import copy
    new=copy.deepcopy(linear)
    with torch.no_grad():new.weight.copy_(reconstructed)
    from safetensors.torch import save_file
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in new.state_dict().items()},str(folder/'dense_proxy.safetensors'))
    meta={'method':'UNIFORM_SCALAR_PROXY_NOT_LUMEN','bits':bits,'simulation_only':True,
          'actual_dense_file_bytes':(folder/'dense_proxy.safetensors').stat().st_size,
          'nominal_packed_weight_bytes_NOT_encoded':math.ceil(w.numel()*bits/8),
          'lumen_compatibility_proven':False}
    write_json(folder/'quantization.json',meta);return new,meta


def apply_quantizer(linear,x,folder,spec):
    kind=spec.get('kind','lumen_cpu');kw=spec.get('options',{})
    if kind=='lumen_cpu':return quantize_linear(linear,x,folder,**kw)
    if kind=='scalar_proxy':return scalar_proxy(linear,x,folder,**kw)
    if kind=='plugin':
        import copy
        fn,meta=load_plugin(spec['factory']);result,report=fn(copy.deepcopy(linear),x,Path(folder),kw)
        if not isinstance(result,nn.Module) or not isinstance(report,dict):raise TypeError('Quantizer plugin contract violated')
        return result,{**report,'plugin_provenance':meta}
    raise ValueError('Unknown quantizer')
