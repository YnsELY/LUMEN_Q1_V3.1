"""One loaded model; reversible candidates; immutable resumable result artifacts."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json, os, signal, sys, time, traceback, uuid
import numpy as np
import torch
from safetensors.torch import save_file
from .common import (ROOT,read_json,write_json,save_npz,load_npz,sha256,digest,code_fingerprint,
                     safe_id,separate_output,ensure_space,utc_now,integer,finite)
from .models import load_bundle,inspect_model
from .data import load_records,encode_records,fixture_records
from .capture import capture_module,save_capture,pair_captures
from .candidates import substitutions,replace_module
from .lumen_bridge import apply_quantizer
from .recovery import recover_local
from .environment import inspect_environment
from .metrics import evaluate_documents,document_summary,paired_difference,synchronize

@contextmanager
def run_lock(folder):
    path=Path(folder)/'.run.lock'
    f=path.open('a+')
    try:
        try:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except ImportError:raise RuntimeError('Runner lock requires Linux/macOS. Implement a platform lock before Windows use.')
        f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush();yield
    finally:f.close()

@contextmanager
def replacements(model):
    backups=[]
    def replace(path,new):
        backups.append((path,model.get_submodule(path)));replace_module(model,path,new)
    try:yield replace
    finally:
        for path,old in reversed(backups):replace_module(model,path,old)

def _plugin_hashes(value):
    result={}
    if isinstance(value,dict):
        for k,v in value.items():
            if (k=='factory' or k.endswith('_factory')) and isinstance(v,str):
                p=Path(v.rpartition(':')[0]);result[str(p.resolve())]=sha256(p)
            else:result.update(_plugin_hashes(v))
    elif isinstance(value,list):
        for v in value:result.update(_plugin_hashes(v))
    return result

def validate_config(cfg):
    if cfg.get('schema_version')!=1:raise ValueError('Require experiment schema_version=1')
    if cfg.get('evaluation_split','select')!='select':raise ValueError('Exploration uses select only; freeze then confirm with a separate command.')
    seen=set()
    for v in cfg['variants']:
        name=safe_id(v['id'])
        if name in seen:raise ValueError('Duplicate variant id')
        seen.add(name)
        for change in v.get('changes',[]):
            if not isinstance(change.get('target'),str) or not change['target']:raise ValueError('Candidate target required')
    if not seen:raise ValueError('No variants')
    integer(cfg.get('keep_last',8),'keep_last',1)
    return cfg

def load_config(path):
    cfg=validate_config(read_json(path));base=Path(path).resolve().parent
    if cfg.get('data_file') and not Path(cfg['data_file']).is_absolute():cfg['data_file']=str((base/cfg['data_file']).resolve())
    if cfg.get('model',{}).get('path') and not Path(cfg['model']['path']).is_absolute():
        cfg['model']['path']=str((base/cfg['model']['path']).resolve())
    if cfg.get('model',{}).get('identity_manifest') and not Path(cfg['model']['identity_manifest']).is_absolute():
        cfg['model']['identity_manifest']=str((base/cfg['model']['identity_manifest']).resolve())
    return cfg

def records_for(cfg,bundle):
    if cfg.get('data_file'):raw,data_report=load_records(cfg['data_file'])
    elif bundle.metadata['synthetic']:
        from .data import validate_records
        raw=fixture_records(**cfg.get('fixture_data',{}));data_report=validate_records(raw)
    else:raise ValueError('Real experiments require an explicit JSON/JSONL corpus')
    records=encode_records(raw,bundle.tokenizer,cfg.get('max_length',256),bundle.metadata['vocab_size'])
    return records,data_report

def file_manifest(folder):
    return {p.relative_to(folder).as_posix():sha256(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}

def verify_files(folder,manifest):
    for rel,h in manifest.items():
        from .common import contained
        p=contained(folder,rel)
        if not p.is_file() or sha256(p)!=h:raise ValueError('Result artifact changed: '+rel)


def run_experiment(config_path,output,*,resume=False):
    cfg=load_config(config_path);out=separate_output(output,cfg.get('model',{}).get('path'))
    if out.exists() and any(out.iterdir()) and not resume:raise FileExistsError('Use a fresh output or explicit --resume')
    out.mkdir(parents=True,exist_ok=True)
    ensure_space(out,reserve=integer(cfg.get('disk_reserve_bytes',2_000_000_000),'disk_reserve_bytes'))
    with run_lock(out):
        return _run(cfg,out,resume)

def _run(cfg,out,resume):
    t0=time.perf_counter();deadline=t0+finite(cfg.get('max_seconds',600),'max_seconds',.001)
    torch.set_num_threads(integer(cfg.get('cpu_threads',1),'cpu_threads',1))
    torch.backends.cuda.matmul.allow_tf32=bool(cfg.get('allow_tf32',False))
    torch.backends.cudnn.allow_tf32=bool(cfg.get('allow_tf32',False))
    fingerprint=digest({'config':cfg,'code':code_fingerprint(),'plugins':_plugin_hashes(cfg)})
    state_path=out/'state.json'
    state=read_json(state_path) if state_path.exists() else {'fingerprint':fingerprint,'completed':{},'created_utc':utc_now()}
    if state['fingerprint']!=fingerprint:raise ValueError('Config/code changed. Start a new versioned run, do not reuse old results.')
    for r in state['completed'].values():verify_files(out/r['directory'],r['manifest'])
    write_json(out/'config.resolved.json',cfg);write_json(out/'environment.json',inspect_environment(out))
    # Preserve executed source versions, including researcher plugins, for audit.
    from .common import atomic_bytes
    for base in ('formic_lab','scripts'):
        for file in (ROOT/base).glob('*.py'):
            atomic_bytes(out/'code_snapshot'/base/file.name,file.read_bytes())
    plugin_snapshot={}
    for file,h in _plugin_hashes(cfg).items():
        target=out/'code_snapshot'/'plugins'/(h+'.py');atomic_bytes(target,Path(file).read_bytes())
        plugin_snapshot[file]={'sha256':h,'snapshot':target.relative_to(out).as_posix()}
    write_json(out/'code_snapshot'/'plugins.json',plugin_snapshot)
    state['status']='loading';write_json(state_path,state)
    def check_time():
        if time.perf_counter()>deadline:raise TimeoutError('Experiment wall-time budget reached')
    # Signal leaves a checkpoint/error record. Supervisor handles hard timeout.
    old_handler=None
    if hasattr(signal,'SIGTERM'):
        old_handler=signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Termination requested')))
    try:
        bundle=load_bundle(cfg.get('model',{'backend':'toy'}));model=bundle.model
        meta_hash=digest(bundle.metadata)
        if state.get('model_metadata_hash',meta_hash)!=meta_hash:raise ValueError('Model/runtime identity changed on resume')
        state['model_metadata_hash']=meta_hash;write_json(out/'model.json',bundle.metadata)
        write_json(out/'architecture.json',inspect_model(model))
        records,data_report=records_for(cfg,bundle)
        data_hash=digest(records)
        if state.get('data_hash',data_hash)!=data_hash:raise ValueError('Data/tokenization changed on resume')
        state['data_hash']=data_hash
        write_json(out/'data_manifest.json',{**data_report,'tokenized_fingerprint':data_hash,
                    'documents':[{'id':r['id'],'source_id':r['source_id'],'split':r['split'],'token_hash':r['token_hash'],'used_tokens':r['used_tokens']} for r in records]})
        split={s:[r for r in records if r['split']==s] for s in ('cal','select','confirm')}
        if not split['cal'] or not split['select']:raise ValueError('cal and select documents required')
        eval_rows=split['select'];keep=cfg.get('keep_last',8);ref={}
        # One reference pass, saved on CPU/disk: never a second full GPU model.
        with torch.inference_mode():
            for r in eval_rows:
                check_time();ref[r['id']]=bundle.forward(r['input_ids'],keep_last=min(keep,len(r['input_ids']))).detach().float().cpu().numpy()
        ref_path=out/'teacher_logits.npz';save_npz(ref_path,**{f'doc_{j}':ref[r['id']] for j,r in enumerate(eval_rows)})
        write_json(out/'teacher_logits.json',{'ids':[r['id'] for r in eval_rows],'sha256':sha256(ref_path),
                                            'positions':'last keep_last predictors in each complete input','keep_last':keep})
        capture_kw={'max_positions':cfg.get('capture_positions',128),'per_document':cfg.get('positions_per_document',32),'seed':cfg.get('capture_seed',3)}
        # Reference captures for local recovery, before any structural change.
        recovery_targets={v['recovery']['target'] for v in cfg['variants'] if v.get('recovery')}
        reference_captures={}
        for path in sorted(recovery_targets):
            for s in ('cal','select'):
                check_time();reference_captures[path,s]=capture_module(bundle,path,split[s],**capture_kw)
        state['status']='running';write_json(state_path,state)
        for spec in cfg['variants']:
            name=spec['id']
            if name in state['completed']:continue
            check_time();folder=out/'variants'/name/('attempt_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
            started=time.perf_counter();status={'id':name,'spec':spec,'status':'running'};write_json(folder/'status.json',status)
            try:
                with substitutions(model,spec.get('changes',[])) as changes, replacements(model) as replace:
                    recovery_log=None
                    if spec.get('recovery'):
                        rc=spec['recovery'];path=rc['target'];actual={}
                        for s in ('cal','select'):
                            check_time();actual[s]=capture_module(bundle,path,split[s],**capture_kw)
                            pair_captures(reference_captures[path,s],actual[s])
                        new,recovery_log=recover_local(model.get_submodule(path),actual['cal'].x,reference_captures[path,'cal'].y,
                                    actual['select'].x,reference_captures[path,'select'].y,**rc.get('options',{}))
                        replace(path,new);write_json(folder/'recovery.json',recovery_log)
                    quant_reports=[]
                    for qspec in spec.get('quantization',[]):
                        check_time();path=qspec['target'];linear=model.get_submodule(path)
                        # Guard BEFORE allocating a large dense Gram or running long calibration.
                        if qspec.get('kind','lumen_cpu')=='lumen_cpu':
                            opts=qspec.get('options',{});shape=getattr(linear,'weight',torch.empty(0)).shape
                            if len(shape)!=2 or shape[0]*shape[1]>opts.get('max_elements',65536) or shape[1]>opts.get('max_input_width',512) or shape[0]>opts.get('max_output_width',256):
                                raise MemoryError('Lumen CPU reference cap. Develop/test a GPU adapter or choose a scientifically valid smaller pilot.')
                        cap=capture_module(bundle,path,split['cal'],**capture_kw)
                        qdir=folder/'quantization'/digest(path)[:12];save_capture(qdir/'capture',cap)
                        new,report=apply_quantizer(linear,cap.x,qdir,qspec)
                        if qspec.get('kind')=='plugin' and not qspec.get('restore_factory'):
                            if type(new) is not torch.nn.Linear or not isinstance(linear,torch.nn.Linear) or new.weight.shape!=linear.weight.shape:
                                raise ValueError('Custom quantized module needs a restore_factory for cold confirmation')
                        replace(path,new.eval());quant_reports.append({'path':path,**report})
                    check_time()
                    # Warm-up excluded from timings; does not update model weights.
                    with torch.inference_mode():bundle.forward(eval_rows[0]['input_ids'],keep_last=keep)
                    if model.get_input_embeddings().weight.device.type=='cuda':torch.cuda.reset_peak_memory_stats(model.get_input_embeddings().weight.device)
                    measurements=evaluate_documents(bundle,eval_rows,ref,keep_last=keep)
                    arch=inspect_model(model)
                    changed_paths=sorted(set([c['target'] for c in spec.get('changes',[]) if c.get('kind')!='identity']+
                                             [q['target'] for q in spec.get('quantization',[])]+([spec['recovery']['target']] if spec.get('recovery') else [])))
                    # Persist only changed root modules. No whole 55GB checkpoint copied.
                    roots=[p for p in changed_paths if not any(p.startswith(a+'.') for a in changed_paths if a!=p)]
                    saved={}
                    for path in roots:
                        local=model.get_submodule(path);needed=sum(t.numel()*t.element_size() for t in local.state_dict().values())
                        if needed>cfg.get('max_candidate_save_bytes',1_000_000_000):raise MemoryError('Candidate checkpoint save cap')
                        ensure_space(folder,needed,reserve=cfg.get('disk_reserve_bytes',2_000_000_000))
                        fname='module_'+digest(path)[:12]+'.safetensors'
                        # clone removes aliases, and saves explicit local state only.
                        save_file({k:v.detach().cpu().contiguous().clone() for k,v in local.state_dict().items()},str(folder/fname))
                        saved[path]=fname
                    result={'id':name,'status':'completed','source_kind':'synthetic_fixture' if bundle.metadata['synthetic'] else 'real_local_model',
                            'scope':'text_teacher_forced_selected_positions; not global capability evaluation',
                            'changes':changes,'quantization':quant_reports,'recovery':recovery_log,
                            'documents':measurements,'summary':document_summary(measurements),
                            'candidate_unique_params':arch['unique_parameters'],'runtime_parameter_bytes':arch['parameter_storage_bytes'],
                            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(model.get_input_embeddings().weight.device) if model.get_input_embeddings().weight.device.type=='cuda' else None,
                            'elapsed_seconds':time.perf_counter()-started,'saved_modules':saved,
                            'model_10gb_produced':False,'saved_state_is_dense_not_compressed_package':True,
                            'encoded_local_tensor_bytes':sum(q.get('encoded_tensor_files_bytes',0) for q in quant_reports)}
                    write_json(folder/'result.json',result);status['status']='completed';write_json(folder/'status.json',status)
                state['completed'][name]={'directory':folder.relative_to(out).as_posix(),'manifest':file_manifest(folder)}
                write_json(state_path,state)
            except Exception as e:
                status.update(status='failed',error=repr(e));write_json(folder/'status.json',status)
                write_json(folder/'error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
        state['status']='completed';state['elapsed_this_invocation_seconds']=time.perf_counter()-t0;write_json(state_path,state)
        report_run(out);return state
    except Exception as e:
        state['status']='interrupted' if isinstance(e,TimeoutError) else 'failed';state['error']=repr(e);write_json(state_path,state)
        raise
    finally:
        if old_handler is not None:signal.signal(signal.SIGTERM,old_handler)


def result_map(run):
    run=Path(run);state=read_json(run/'state.json');out={}
    for name,info in state['completed'].items():
        folder=run/info['directory'];verify_files(folder,info['manifest']);out[name]=read_json(folder/'result.json')
    return out


def freeze(run,ids):
    run=Path(run);state=read_json(run/'state.json');results=result_map(run)
    if not ids or len(set(ids))!=len(ids) or any(i not in results for i in ids):raise ValueError('Unknown/duplicate final candidates')
    lock={'schema':1,'created_utc':utc_now(),'run_fingerprint':state['fingerprint'],
          'data_hash':state['data_hash'],'candidate_ids':ids,'artifact_manifests':{i:state['completed'][i] for i in ids},
          'note':'Freeze before reading confirmation. Any subsequent selection needs a new holdout.'}
    write_json(run/'selection.lock.json',lock,overwrite=False);return lock


def confirm(run):
    """Reload saved candidate local weights, do NOT re-fit using confirmation."""
    from safetensors.torch import load_file
    run=Path(run);cfg=read_json(run/'config.resolved.json');lock=read_json(run/'selection.lock.json');state=read_json(run/'state.json')
    if (run/'confirmation.json').exists():raise FileExistsError('Confirmation already consumed; do not rerun to cherry-pick')
    if lock['run_fingerprint']!=state['fingerprint'] or digest({'config':cfg,'code':code_fingerprint(),'plugins':_plugin_hashes(cfg)})!=state['fingerprint']:
        raise ValueError('Frozen code/config changed')
    for name,info in lock['artifact_manifests'].items():verify_files(run/info['directory'],info['manifest'])
    b=load_bundle(cfg['model']);records,_=records_for(cfg,b)
    if digest(records)!=lock['data_hash'] or digest(b.metadata)!=state['model_metadata_hash']:raise ValueError('Frozen data/model changed')
    docs=[r for r in records if r['split']=='confirm']
    if not docs:raise ValueError('No confirmation documents')
    keep=cfg.get('keep_last',8);ref={}
    with torch.inference_mode():
        for r in docs:ref[r['id']]=b.forward(r['input_ids'],keep).float().cpu().numpy()
    results={};specs={s['id']:s for s in cfg['variants']}
    for name in lock['candidate_ids']:
        folder=run/lock['artifact_manifests'][name]['directory'];old=read_json(folder/'result.json')
        with substitutions(b.model,specs[name].get('changes',[])), replacements(b.model) as replace:
            import copy
            # Reconstruct custom quantizer module types before loading frozen state.
            for qs in specs[name].get('quantization',[]):
                if qs.get('restore_factory'):
                    from .common import load_plugin
                    fn,_=load_plugin(qs['restore_factory']);path=qs['target']
                    qr=next(v for v in old['quantization'] if v['path']==path)
                    new=fn(copy.deepcopy(b.model.get_submodule(path)),folder/'quantization'/digest(path)[:12],qs.get('options',{}),qr)
                    if not isinstance(new,torch.nn.Module):raise TypeError('restore_factory must return nn.Module')
                    replace(path,new.eval())
            # Loaded state may include local recovery / decoded compressed weights.
            for path,file in old['saved_modules'].items():
                new=copy.deepcopy(b.model.get_submodule(path));new.load_state_dict(load_file(str(folder/file)),strict=True);replace(path,new.eval())
            rows=evaluate_documents(b,docs,ref,keep)
            results[name]={'documents':rows,'summary':document_summary(rows),'trained_on_confirmation':False}
    report={'status':'confirmation_consumed','frozen_candidates':lock['candidate_ids'],'results':results,
            'synthetic':b.metadata['synthetic'],'no_selection_based_on_this_report':True}
    write_json(run/'confirmation.json',report,overwrite=False);return report


def report_run(run):
    run=Path(run);data=result_map(run);rows=[]
    for name,r in data.items():
        rows.append(f"| {name} | {r['summary']['document_mean_kl']:.6g} | {r['encoded_local_tensor_bytes']} | {r['candidate_unique_params']} | {r['elapsed_seconds']:.2f} |")
    text='# Rapport automatisé — mesures, pas conclusions scientifiques\n\n'
    text+='Périmètre : texte, préfixes imposés, positions échantillonnées. Les octets encodés concernent uniquement les matrices traitées. Les snapshots sont denses. Aucun package complet de 10 Go ni conservation des capacités générales n’est démontré.\n\n'
    text+='| Variante | KL moyenne par document | Octets des tenseurs locaux encodés | Paramètres uniques chargés | Secondes |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    text+='Sélection à faire par le chercheur sur `select`. Geler les identifiants avec `freeze`, puis seulement lancer `confirm`. Interpréter les différences à tailles et efforts comparables ; ne pas additionner les gains.\n'
    (run/'RAPPORT_AUTOMATIQUE.md').write_text(text,encoding='utf-8');return data
