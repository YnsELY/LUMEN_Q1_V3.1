from __future__ import annotations
import hashlib,tarfile
from pathlib import Path
from .common import sha256,write_json,integer

def pack_results(run,destination,max_file_bytes=20_000_000,include_arrays=False):
    run=Path(run).resolve();destination=Path(destination).resolve()
    if destination.is_relative_to(run):raise ValueError('Archive destination must be outside the run')
    if destination.exists():raise FileExistsError(destination)
    integer(max_file_bytes,'max_file_bytes',1);included=[];excluded=[]
    heavy={'.safetensors','.npz','.npy','.lmat','.bin'}
    for p in sorted(run.rglob('*')):
        if p.is_symlink():excluded.append({'path':p.relative_to(run).as_posix(),'reason':'symlink'});continue
        if not p.is_file() or p.name=='.run.lock':continue
        if p.stat().st_size>max_file_bytes or (p.suffix in heavy and not include_arrays):
            excluded.append({'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,'reason':'size_or_array_exclusion'});continue
        included.append(p)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(destination,'w:gz') as tar:
        for p in included:tar.add(p,arcname=p.relative_to(run).as_posix(),recursive=False)
    report={'archive':destination.name,'sha256':sha256(destination),'included_files':len(included),
            'excluded':excluded,'private_activations_included':include_arrays,
            'complete_for_reproduction':not excluded}
    write_json(destination.with_suffix(destination.suffix+'.manifest.json'),report);return report
