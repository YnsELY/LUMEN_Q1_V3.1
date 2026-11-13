#!/usr/bin/env python3
"""Self-tests run on CPU fixtures; no weights, API calls or pod billing."""
from pathlib import Path
import sys,unittest,argparse,json,time,platform
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--json-out',type=Path);a=p.parse_args()
t=time.perf_counter();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
r=unittest.TextTestRunner(verbosity=2).run(suite)
if a.json_out:
 from formic_lab.common import write_json,code_fingerprint
 import numpy,torch
 write_json(a.json_out,{'success':r.wasSuccessful(),'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),
  'seconds':time.perf_counter()-t,'python':platform.python_version(),'numpy':numpy.__version__,'torch':torch.__version__,
  'code_fingerprint':code_fingerprint(),'scope':'CPU synthetic fixtures + small real files; NOT Qwen nor CUDA validation'})
raise SystemExit(0 if r.wasSuccessful() else 1)
