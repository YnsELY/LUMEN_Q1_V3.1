#!/usr/bin/env python3
"""Verify exact manifest coverage, hashes, canonical paths; never a signature."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, re, sys
ROOT=Path(__file__).resolve().parents[1]


def verify(root:Path):
    root=Path(root).resolve();errors=[];seen=set()
    try:
        manifest=json.loads((root/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
        entries=manifest['files']
        if not isinstance(entries,list):raise ValueError('files must be a list')
    except (ValueError,KeyError,OSError,TypeError) as e:
        return {'verified':False,'files_checked':0,'errors':[{'error':'invalid_manifest','detail':str(e)}]}
    for entry in entries:
        if not isinstance(entry,dict):errors.append({'error':'invalid_entry'});continue
        name=entry.get('path')
        if (not isinstance(name,str) or not name or '\\' in name or
                PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or
                PurePosixPath(name).as_posix()!=name or name=='MANIFEST_SHA256.json'):
            errors.append({'path':str(name),'error':'unsafe_or_noncanonical_path'});continue
        if name in seen:errors.append({'path':name,'error':'duplicate_manifest_entry'});continue
        seen.add(name); p=root/name
        if (p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file() or
                any(parent.is_symlink() for parent in p.parents if parent!=root and parent.is_relative_to(root))):
            errors.append({'path':name,'error':'missing_symlink_or_invalid'});continue
        if type(entry.get('bytes')) is not int or entry['bytes']<0 or not re.fullmatch('[0-9a-f]{64}',str(entry.get('sha256'))):
            errors.append({'path':name,'error':'invalid_size_or_hash'});continue
        digest=hashlib.sha256();length=0
        with p.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):
                digest.update(chunk);length+=len(chunk)
        if length!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
            errors.append({'path':name,'error':'content_changed'})
    actual={p.relative_to(root).as_posix() for p in root.rglob('*')
            if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts
            and p.relative_to(root).as_posix()!='MANIFEST_SHA256.json'}
    for name in sorted(actual-seen):errors.append({'path':name,'error':'undeclared_file'})
    return {'verified':not errors,'files_checked':len(entries),'actual_files':len(actual),'errors':errors}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args()
    result=verify(args.root);print(json.dumps(result,indent=2))
    if not result['verified']:sys.exit(1)
if __name__=='__main__':main()
