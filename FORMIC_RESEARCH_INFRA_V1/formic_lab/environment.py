from __future__ import annotations
import importlib.metadata, os, platform, shutil, subprocess, sys
from pathlib import Path
from .common import utc_now

def inspect_environment(workdir):
    p=Path(workdir);p.mkdir(parents=True,exist_ok=True)
    versions={}
    for name in ('torch','numpy','safetensors','transformers','accelerate','psutil','pytest'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    try:
        import psutil
        ram=psutil.virtual_memory()._asdict()
    except ImportError: ram={'available':None}
    quota=None
    try:
        q,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if q!='max': quota=int(q)/int(period)
    except (OSError,ValueError): pass
    cuda={'available':False,'devices':[]}
    try:
        import torch
        cuda={'available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'devices':[]}
        if cuda['available']:
            for i in range(torch.cuda.device_count()):
                d=torch.cuda.get_device_properties(i)
                free,total=torch.cuda.mem_get_info(i)
                cuda['devices'].append({'index':i,'name':d.name,'total_bytes':total,'free_bytes':free,
                                       'compute_capability':[d.major,d.minor]})
    except ImportError: pass
    return {'created_utc':utc_now(),'python':sys.version,'platform':platform.platform(),
            'versions':versions,'logical_cpus':os.cpu_count(),'cpu_quota_equivalent':quota,
            'ram':ram,'disk':shutil.disk_usage(p)._asdict(),'cuda':cuda,
            'no_gpu_test_performed':not cuda['available']}

def resource_sample():
    data={'utc':utc_now()}
    try:
        import psutil
        data['system_ram_available']=psutil.virtual_memory().available
        data['system_cpu_percent']=psutil.cpu_percent(interval=None)
    except ImportError: pass
    if shutil.which('nvidia-smi'):
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
            data['gpu_csv']=r.stdout.strip()
        except (OSError,subprocess.SubprocessError) as e: data['gpu_sample_error']=str(e)
    return data
