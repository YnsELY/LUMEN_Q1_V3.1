"""CLI shared by the small scripts. All network/model downloads are disabled by design."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import ROOT,read_json,write_json,save_npz,load_npz,safe_id,digest,integer


def main(argv=None):
    p=argparse.ArgumentParser(description='Formic research infrastructure — explicit scope, reversible examples')
    sub=p.add_subparsers(dest='command',required=True)
    def command(name):return sub.add_parser(name)
    def out(q):q.add_argument('--out',type=Path,required=True)
    def config(q):q.add_argument('--config',type=Path,required=True)
    q=command('environment');q.add_argument('--workdir',type=Path,default=Path.cwd());out(q)
    q=command('checkpoint');q.add_argument('--path',type=Path,required=True);q.add_argument('--identity',type=Path);q.add_argument('--full-hash',action='store_true');out(q)
    for name in ('inspect','load'):
        q=command(name);config(q);out(q)
    q=command('prepare-data');q.add_argument('--input',type=Path,required=True);q.add_argument('--tokenizer',type=Path);q.add_argument('--max-length',type=int,default=256);out(q)
    q=command('capture');config(q);q.add_argument('--target',required=True);q.add_argument('--split',choices=('cal','select'),default='cal');out(q)
    q=command('redundancy');q.add_argument('--npz',type=Path,required=True);q.add_argument('--key',default='W');q.add_argument('--max-dimension',type=int,default=1024);out(q)
    q=command('sensitivity');q.add_argument('--npz',type=Path,required=True);out(q)
    q=command('candidates');q.add_argument('--grid',type=Path,required=True);out(q)
    q=command('apply');config(q);q.add_argument('--variant',required=True);out(q)
    q=command('quantize');q.add_argument('--npz',type=Path,required=True);q.add_argument('--step',type=float,default=.8);q.add_argument('--rate-lambda',type=float,default=0.);out(q)
    q=command('recover');q.add_argument('--npz',type=Path,required=True);q.add_argument('--steps',type=int,default=30);out(q)
    q=command('evaluate');q.add_argument('--npz',type=Path,required=True);out(q)
    q=command('run');config(q);out(q);q.add_argument('--resume',action='store_true')
    q=command('compare');q.add_argument('--run',type=Path,required=True);q.add_argument('--control',required=True);q.add_argument('--candidate',required=True);out(q)
    q=command('monitor');q.add_argument('--budget',type=Path,required=True);q.add_argument('--cwd',type=Path,default=Path.cwd());out(q);q.add_argument('--poll',type=float,default=2.);q.add_argument('--grace',type=float,default=10.);q.add_argument('argv',nargs=argparse.REMAINDER)
    q=command('archive');q.add_argument('--run',type=Path,required=True);q.add_argument('--dest',type=Path,required=True);q.add_argument('--include-arrays',action='store_true');q.add_argument('--max-file-bytes',type=int,default=20_000_000)
    q=command('report');q.add_argument('--run',type=Path,required=True)
    q=command('freeze');q.add_argument('--run',type=Path,required=True);q.add_argument('--ids',nargs='+',required=True)
    q=command('confirm');q.add_argument('--run',type=Path,required=True)
    q=command('check-release')
    a=p.parse_args(argv);result=None
    if a.command=='environment':
        from .environment import inspect_environment
        result=inspect_environment(a.workdir)
    elif a.command=='checkpoint':
        from .checkpoint import inspect_checkpoint
        result=inspect_checkpoint(a.path,a.identity,a.full_hash)
    elif a.command in ('inspect','load','capture','apply'):
        from .runner import load_config,records_for
        from .models import load_bundle,inspect_model
        c=load_config(a.config)
        from .common import separate_output
        separate_output(a.out,c.get('model',{}).get('path'))
        b=load_bundle(c['model'])
        if a.command=='inspect':result={'model':b.metadata,'architecture':inspect_model(b.model)}
        elif a.command=='load':
            docs,_=records_for(c,b)
            with torch.inference_mode():z=b.forward(docs[0]['input_ids'],keep_last=1)
            result={'model':b.metadata,'smoke_logits_shape':list(z.shape),'finite':bool(torch.isfinite(z).all()),'scope':'one short text forward only'}
        elif a.command=='capture':
            from .capture import capture_module,save_capture
            docs,_=records_for(c,b);docs=[r for r in docs if r['split']==a.split]
            cap=capture_module(b,a.target,docs,max_positions=c.get('capture_positions',128),per_document=c.get('positions_per_document',32))
            save_capture(a.out,cap);print(json.dumps(cap.meta,indent=2));return 0
        else:
            from .candidates import substitutions
            from safetensors.torch import save_file
            match=[v for v in c['variants'] if v['id']==a.variant]
            if len(match)!=1:raise ValueError('Unknown candidate')
            if match[0].get('quantization') or match[0].get('recovery'):raise ValueError('apply handles architecture only; use run for recovery/quantization')
            a.out.mkdir(parents=True,exist_ok=False)
            with substitutions(b.model,match[0].get('changes',[])) as info:
                result={'changes':info,'architecture':inspect_model(b.model),'scope':'in-memory surgery only; no fidelity claim'}
                for change in match[0].get('changes',[]):
                    if change.get('kind')=='identity':continue
                    mod=b.model.get_submodule(change['target'])
                    size=sum(t.numel()*t.element_size() for t in mod.state_dict().values())
                    if size>c.get('max_candidate_save_bytes',1_000_000_000):raise MemoryError('Candidate save cap')
                    save_file({k:v.detach().cpu().contiguous().clone() for k,v in mod.state_dict().items()},str(a.out/(digest(change['target'])[:12]+'.safetensors')))
            write_json(a.out/'candidate.json',result);print(json.dumps(result,indent=2));return 0
    elif a.command=='prepare-data':
        from .data import load_records,encode_records,validate_records
        docs,result=load_records(a.input)
        if a.tokenizer:
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(str(a.tokenizer),local_files_only=True,trust_remote_code=False)
            docs=encode_records(docs,tok,max_length=a.max_length)
            for d in docs:
                if 'text' in d:d['raw_text_sha256']=digest(d.pop('text'))
            result={**validate_records(docs),'tokenizer_path':str(a.tokenizer.resolve()),'max_length':a.max_length,
                    'note':'CPU tokenization only; no model weights loaded; runtime still verifies vocabulary.'}
        write_json(a.out,docs);write_json(a.out.with_suffix('.manifest.json'),result)
        print(json.dumps(result,indent=2));return 0
    elif a.command=='redundancy':
        from .analysis import redundancy
        result=redundancy(load_npz(a.npz)[a.key],a.max_dimension)
    elif a.command=='sensitivity':
        from .analysis import sensitivity
        ar=load_npz(a.npz);result=sensitivity(ar['W'],ar['Q'],ar['X'],ar.get('Y'))
    elif a.command=='candidates':
        grid=read_json(a.grid);values=grid['values'];param=grid['parameter']
        if param not in ('rank','keep_indices') or not isinstance(values,list) or not 1<=len(values)<=32:raise ValueError('Explicit small rank/channel grid required; use a plugin for other searches')
        result=[]
        for j,value in enumerate(values):
            spec=copy.deepcopy(grid['base']);spec[param]=value
            result.append({'id':safe_id(grid.get('prefix','candidate')+'_'+str(j)),'changes':[spec]})
    elif a.command=='quantize':
        from .lumen_bridge import quantize_linear
        ar=load_npz(a.npz);w=np.asarray(ar['W'],dtype=np.float32)
        if w.ndim!=2:raise ValueError('W must be a matrix')
        mod=nn.Linear(w.shape[1],w.shape[0],bias='bias' in ar)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(w))
            if 'bias' in ar:mod.bias.copy_(torch.from_numpy(ar['bias']))
        _,result=quantize_linear(mod,ar['X'],a.out,step_multiplier=a.step,rate_lambda=a.rate_lambda)
        print(json.dumps(result,indent=2));return 0
    elif a.command=='recover':
        from .recovery import recover_local
        from safetensors.torch import save_file
        ar=load_npz(a.npz);w=ar['W'];mod=nn.Linear(w.shape[1],w.shape[0],bias='bias' in ar)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(w))
            if 'bias' in ar:mod.bias.copy_(torch.from_numpy(ar['bias']))
        new,result=recover_local(mod,ar['X_cal'],ar['Y_cal'],ar['X_select'],ar['Y_select'],steps=a.steps)
        a.out.mkdir(parents=True,exist_ok=False);save_file({k:v.detach().cpu().contiguous() for k,v in new.state_dict().items()},str(a.out/'recovered.safetensors'))
        write_json(a.out/'recovery.json',result);print(json.dumps(result,indent=2));return 0
    elif a.command=='evaluate':
        from .metrics import logits_metrics
        ar=load_npz(a.npz);result=logits_metrics(ar['teacher_logits'],ar['student_logits'])
    elif a.command=='run':
        from .runner import run_experiment
        result=run_experiment(a.config,a.out,resume=a.resume);print(json.dumps(result,indent=2));return 0
    elif a.command=='compare':
        from .runner import result_map
        from .metrics import paired_difference
        runs=result_map(a.run);control=runs[a.control];candidate=runs[a.candidate]
        result={'paired':paired_difference(control['documents'],candidate['documents']),
                'control_local_encoded_bytes':control['encoded_local_tensor_bytes'],
                'candidate_local_encoded_bytes':candidate['encoded_local_tensor_bytes'],
                'same_local_encoded_budget':control['encoded_local_tensor_bytes']==candidate['encoded_local_tensor_bytes'],
                'note':'Local encoded scopes may differ. Check tensors, biases, transformations and full-package ledger before inferring iso-size gains.'}
    elif a.command=='monitor':
        from .supervisor import supervise
        argv=a.argv[1:] if a.argv[:1]==['--'] else a.argv
        result=supervise(argv,a.cwd,a.out,read_json(a.budget),poll_seconds=a.poll,grace_seconds=a.grace)
        print(json.dumps(result,indent=2));return result['exit_code'] if result['exit_code'] is not None and result['exit_code']>=0 else 2
    elif a.command=='archive':
        from .archive import pack_results
        result=pack_results(a.run,a.dest,a.max_file_bytes,a.include_arrays);print(json.dumps(result,indent=2));return 0
    elif a.command=='report':
        from .runner import report_run
        report_run(a.run);print(a.run/'RAPPORT_AUTOMATIQUE.md');return 0
    elif a.command=='freeze':
        from .runner import freeze
        print(json.dumps(freeze(a.run,a.ids),indent=2));return 0
    elif a.command=='confirm':
        from .runner import confirm
        print(json.dumps(confirm(a.run),indent=2));return 0
    elif a.command=='check-release':
        from .release import check_release
        result=check_release(ROOT);print(json.dumps(result,indent=2));return 0 if result['ok'] else 1
    if result is not None:
        write_json(a.out,result);print(json.dumps(result,indent=2,allow_nan=False))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,RuntimeError,MemoryError) as e:
        print(f'ERROR: {e}',file=sys.stderr);raise SystemExit(2)
