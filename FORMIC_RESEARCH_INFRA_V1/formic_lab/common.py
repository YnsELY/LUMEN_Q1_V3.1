"""Small explicit contracts, atomic results and reproducible identifiers."""
from __future__ import annotations
import hashlib, importlib.util, json, math, os, re, shutil, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPO = 'Qwen/Qwen3.8-27B'
EXPECTED_REVISION = '37561816e66b5b9153b1b64c10968c18c1e6f4a0'

def integer(x, name, minimum=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(x)

def finite(x, name, minimum=None):
    if isinstance(x, (bool, np.bool_)):
        raise ValueError(f'{name} cannot be boolean')
    v = float(x)
    if not math.isfinite(v) or (minimum is not None and v < minimum):
        raise ValueError(f'Invalid {name}')
    return v

def utc_now(): return datetime.now(timezone.utc).isoformat()

def read_json(path):
    def bad(x): raise ValueError(f'Nonfinite JSON value: {x}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=bad)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def ensure_space(path, extra=0, reserve=2_000_000_000):
    path=Path(path)
    while not path.exists(): path=path.parent
    free=shutil.disk_usage(path).free
    if free < integer(extra,'extra')+integer(reserve,'reserve'):
        raise OSError(f'Insufficient disk: {free} bytes free; need {extra}+{reserve}')

def atomic_bytes(path, content: bytes, *, overwrite=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not overwrite: raise FileExistsError(path)
    ensure_space(path.parent,len(content),reserve=0)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def write_json(path, value, **kw):
    atomic_bytes(path,(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode(),**kw)

def save_npz(path, **arrays):
    import io
    for k,v in arrays.items():
        if np.asarray(v).dtype.hasobject: raise ValueError('Object arrays prohibited: '+k)
    b=io.BytesIO();np.savez(b,**arrays);atomic_bytes(path,b.getvalue())

def load_npz(path):
    with np.load(path,allow_pickle=False) as f: return {k:np.array(f[k]) for k in f.files}

def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',value):
        raise ValueError('Invalid identifier')
    return value

def contained(root,relative):
    root=Path(root).resolve(); rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('Path traversal')
    path=(root/rel).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes root')
    return path

def separate_output(output, model_path=None):
    out=Path(output).resolve()
    if model_path:
        model=Path(model_path).resolve()
        if out==model or out.is_relative_to(model) or model.is_relative_to(out):
            raise ValueError('Working directory must not contain or be inside the checkpoint')
    return out

def code_fingerprint():
    return digest({p.relative_to(ROOT).as_posix():sha256(p) for base in ('formic_lab','vendor/lumen_ref','scripts')
                   for p in sorted((ROOT/base).glob('*.py'))})

def load_plugin(spec):
    """Executes explicit local Python. This is NOT a security sandbox."""
    file, sep, name = spec.rpartition(':')
    if not sep or not name.isidentifier(): raise ValueError('Plugin syntax: /local/file.py:function')
    path=Path(file).resolve()
    if not path.is_file() or path.suffix!='.py': raise ValueError('Plugin must be a local .py file')
    modname='formic_plugin_'+sha256(path)[:16]
    module_spec=importlib.util.spec_from_file_location(modname,path)
    module=importlib.util.module_from_spec(module_spec);sys.modules[modname]=module
    module_spec.loader.exec_module(module)
    fn=getattr(module,name)
    if not callable(fn): raise ValueError('Plugin entry is not callable')
    return fn, {'path':str(path),'sha256':sha256(path),'function':name}
