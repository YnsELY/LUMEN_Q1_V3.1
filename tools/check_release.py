#!/usr/bin/env python3
"""Read-only release check; all tests run on a disposable copy of the bundle."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from verify_manifest import verify
ROOT=Path(__file__).resolve().parents[1]


def snapshot(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root.resolve()
    before=snapshot(root);manifest=verify(root)
    if not manifest['verified']:
        print(json.dumps({'ok':False,'manifest':manifest},indent=2));raise SystemExit(1)
    with tempfile.TemporaryDirectory(prefix='lumen_v31_check_') as td:
        tmp=Path(td)/'bundle';shutil.copytree(root,tmp,ignore=shutil.ignore_patterns('__pycache__'))
        code="""import sys,unittest,io,json\nfrom pathlib import Path\nr=Path(sys.argv[1]);sys.path.insert(0,str(r));s=unittest.defaultTestLoader.discover(str(r/'tests'));l=io.StringIO();a=unittest.TextTestRunner(stream=l,verbosity=2).run(s);print(json.dumps({'tests_run':a.testsRun,'passed':a.testsRun-len(a.errors)-len(a.failures),'failures':len(a.failures),'errors':len(a.errors),'log':l.getvalue()}));sys.exit(0 if a.wasSuccessful() else 1)"""
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        run=subprocess.run([sys.executable,'-B','-c',code,str(tmp)],cwd=td,env=env,text=True,encoding='utf-8',capture_output=True,timeout=180)
        try:tests=json.loads(run.stdout)
        except json.JSONDecodeError:tests={'error':'invalid_test_output','stdout':run.stdout,'stderr':run.stderr}
    unchanged=snapshot(root)==before
    result={'ok':run.returncode==0 and unchanged,'scope':'CPU regression/codec tests, NOT Qwen quality validation','manifest':manifest,'tests':{k:v for k,v in tests.items() if k!='log'},'source_unchanged':unchanged}
    if run.returncode!=0:result['test_log']=tests.get('log',run.stdout);result['stderr']=run.stderr
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if not result['ok']:raise SystemExit(1)
if __name__=='__main__':main()
