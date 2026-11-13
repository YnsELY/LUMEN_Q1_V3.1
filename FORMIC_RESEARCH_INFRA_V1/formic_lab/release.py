from __future__ import annotations
from pathlib import Path
from .common import sha256,read_json,contained

def check_release(root):
    root=Path(root);m=read_json(root/'MANIFEST_SHA256.json');errors=[];seen=set()
    for entry in m['files']:
        rel=entry['path']
        if rel in seen:errors.append('duplicate:'+rel)
        seen.add(rel)
        try:
            p=contained(root,rel)
            if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha256(p)!=entry['sha256']:errors.append('mismatch:'+rel)
        except (ValueError,OSError):errors.append('invalid:'+rel)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    extras=actual-seen-{'MANIFEST_SHA256.json'}
    errors.extend('extra:'+x for x in sorted(extras))
    return {'ok':not errors,'checked_files':len(seen),'errors':errors,'note':'Integrity only, not scientific validation.'}
