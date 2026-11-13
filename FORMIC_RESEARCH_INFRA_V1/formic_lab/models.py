"""Explicit toy fixture and offline Transformers adapter. No automatic downloads."""
from __future__ import annotations
from dataclasses import dataclass
import inspect,math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from .common import integer,digest,code_fingerprint,load_plugin
from .checkpoint import inspect_checkpoint

class ToyMLP(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.gate_proj=nn.Linear(d,h,bias=False);self.up_proj=nn.Linear(d,h,bias=False);self.down_proj=nn.Linear(h,d,bias=False)
    def forward(self,x):return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))

class ToyBlock(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.norm=nn.LayerNorm(d);self.qkv=nn.Linear(d,d*3,bias=False);self.o_proj=nn.Linear(d,d,bias=False);self.post_norm=nn.LayerNorm(d);self.mlp=ToyMLP(d,h)
    def forward(self,x):
        q,k,v=self.qkv(self.norm(x)).chunk(3,-1)
        scores=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]);s=x.shape[1]
        scores=scores.masked_fill(torch.ones(s,s,device=x.device,dtype=torch.bool).triu(1),float('-inf'))
        x=x+self.o_proj(scores.softmax(-1)@v)
        return x+self.mlp(self.post_norm(x))

class ToyLM(nn.Module):
    """Tiny causal fixture, NOT a Qwen/DeltaNet simulation."""
    def __init__(self,vocab=64,width=16,hidden=32,layers=2):
        super().__init__();self.embed_tokens=nn.Embedding(vocab,width);self.blocks=nn.ModuleList([ToyBlock(width,hidden) for _ in range(layers)])
        self.norm=nn.LayerNorm(width);self.lm_head=nn.Linear(width,vocab,bias=False)
        self.config=SimpleNamespace(vocab_size=vocab,model_type='formic_synthetic_fixture',use_cache=False)
    def get_input_embeddings(self):return self.embed_tokens
    def forward(self,input_ids,attention_mask=None,use_cache=False,logits_to_keep=0,return_dict=True):
        if use_cache:raise ValueError('Fixture does not implement stateful decode')
        x=self.embed_tokens(input_ids)
        for b in self.blocks:x=b(x)
        x=self.norm(x)
        if logits_to_keep:x=x[:,-logits_to_keep:]
        return SimpleNamespace(logits=self.lm_head(x))

@dataclass
class Bundle:
    model: nn.Module
    tokenizer: object
    metadata: dict
    max_logit_elements: int=64_000_000
    forward_fn: object=None

    def forward(self,ids,keep_last=1):
        if self.forward_fn is not None:return self.forward_fn(self,ids,keep_last)
        dev=self.model.get_input_embeddings().weight.device
        if dev.type=='meta':raise RuntimeError('Input embedding is meta; provide a compatible factory')
        x=torch.tensor([ids],device=dev,dtype=torch.long)
        sig=inspect.signature(self.model.forward)
        kwargs={'input_ids':x,'attention_mask':torch.ones_like(x),'use_cache':False,'return_dict':True}
        vocab=self.metadata['vocab_size'];keep_last=min(integer(keep_last,'keep_last',1),len(ids))
        if 'logits_to_keep' in sig.parameters:kwargs['logits_to_keep']=keep_last;estimate=vocab*keep_last
        else:estimate=vocab*len(ids)
        if estimate>self.max_logit_elements:
            raise MemoryError('Logit safety cap; reduce sequence length or supply a bounded forward adapter')
        out=self.model(**kwargs)
        logits=out.logits if hasattr(out,'logits') else out['logits']
        if logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[-1]!=vocab or logits.shape[1]<keep_last:
            raise ValueError('Unexpected logits shape; adapt the runtime explicitly')
        return logits[0,-keep_last:]

def load_bundle(cfg):
    backend=cfg.get('backend','toy');device=cfg.get('device','cpu');seed=integer(cfg.get('seed',17),'seed')
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    if backend=='plugin':
        factory,prov=load_plugin(cfg['factory']);b=factory(cfg)
        if not isinstance(b,Bundle):raise TypeError('Factory must return formic_lab.models.Bundle')
        b.metadata['factory']=prov;return b
    if backend=='toy':
        with torch.random.fork_rng():
            torch.manual_seed(seed);model=ToyLM(**cfg.get('toy',{}))
        model.to(device=device,dtype=torch.float32).eval()
        meta={'backend':'toy','synthetic':True,'checkpoint':'formic/toy-fixture','revision':code_fingerprint(),
              'vocab_size':model.config.vocab_size,'seed':seed,'model_settings':cfg,
              'validation_scope':'synthetic_causal_fixture_not_Qwen'}
    elif backend=='hf_local':
        report=inspect_checkpoint(cfg['path'],cfg.get('identity_manifest'),cfg.get('full_hash',False))
        if report['revision_declared'] is None and not cfg.get('allow_unverified_revision',False):
            raise ValueError('No pinned revision evidence. Supply identity manifest or explicitly allow unverified local identity for diagnosis.')
        expected_count=integer(cfg.get('expected_parameter_count',27_781_427_952),'expected_parameter_count',1)
        if report['parameter_count']!=expected_count:
            raise ValueError('Checkpoint inventory differs from the expected project model; no silent substitution')
        dtype_name=cfg.get('dtype','bfloat16')
        if dtype_name not in ('bfloat16','float32','float16'):raise ValueError('Unsupported dtype')
        if device.startswith('cuda'):
            if dtype_name=='bfloat16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported on this runtime')
            free,total=torch.cuda.mem_get_info(torch.device(device))
            required=report['parameter_count']*({'float32':4,'float16':2,'bfloat16':2}[dtype_name])
            margin=integer(cfg.get('vram_margin_bytes',6_000_000_000),'vram_margin_bytes')
            if free<required+margin:raise MemoryError('Weights + configured VRAM reserve do not fit. No implicit offload/quantization.')
        import transformers
        clsname=cfg.get('auto_class','AutoModelForImageTextToText')
        if clsname not in ('AutoModelForImageTextToText','AutoModelForCausalLM'):raise ValueError('Use a factory for a custom runtime')
        cls=getattr(transformers,clsname)
        model=cls.from_pretrained(cfg['path'],local_files_only=True,trust_remote_code=False,
                                 use_safetensors=True,dtype=getattr(torch,dtype_name),device_map={'':device},
                                 attn_implementation=cfg.get('attn_implementation','sdpa'))
        model.eval()
        tokenizer=transformers.AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True,trust_remote_code=False)
        vocab=model.get_input_embeddings().weight.shape[0]
        meta={'backend':'hf_local','synthetic':False,'checkpoint':report['repo_expected'],
              'revision':report['revision_expected'],'vocab_size':vocab,'checkpoint_report':report,
              'validation_scope':'text_teacher_forced_only; Vision/MTP/stateful_decode_not_validated',
              'transformers_version':transformers.__version__,'dtype':dtype_name,'auto_class':clsname}
        return Bundle(model,tokenizer,meta,integer(cfg.get('max_logit_elements',64_000_000),'max_logit_elements',1))
    else:raise ValueError('Unknown backend')
    return Bundle(model,None,meta,integer(cfg.get('max_logit_elements',64_000_000),'max_logit_elements',1))

def inspect_model(model):
    modules=[]
    for path,m in model.named_modules():
        local=sum(p.numel() for p in m.parameters(recurse=False))
        if local or len(list(m.children())):
            record={'path':path,'type':type(m).__name__,'local_parameters':local}
            if isinstance(m,nn.Linear):record.update(in_features=m.in_features,out_features=m.out_features,bias=m.bias is not None)
            modules.append(record)
    params=list(model.parameters());storages={}
    for p in params:
        if p.device.type=='meta': continue
        st=p.untyped_storage();storages[(str(p.device),st.data_ptr())]=st.nbytes()
    return {'unique_parameters':sum(p.numel() for p in params),
            'parameter_count_note':'Unique Parameter objects; separately wrapped shared views can overlap. Storage bytes deduplicated by allocation.',
            'parameter_storage_bytes':sum(storages.values()),
            'loaded_mtp_names':[n for n,_ in model.named_parameters() if 'mtp' in n.lower()],
            'modules':modules,'note':'Loaded module tree, not a proof that all paths execute in text forward.'}
