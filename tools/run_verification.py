#!/usr/bin/env python3
"""Run regression suite and explicit old/new numerical counterexamples."""
from pathlib import Path
import sys,json,unittest,io,time,platform,importlib.util,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from lumen_ref.linalg import inverse_unit_upper,block_ldl
from lumen_ref.budget import lowrank_cost

def main():
 t=time.perf_counter();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'));log=io.StringIO();res=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 (ROOT/'results'/'tests_final.log').write_text(log.getvalue())
 spec=importlib.util.spec_from_file_location('historical_v2',ROOT/'originals'/'budget_lumen_max.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 u=np.eye(8);u[0,1]=.5;u[0,2]=.2;u[1,2]=.3;d=np.arange(1,9,dtype=float);h=(u*d)@u.T
 u_old,b_old=old.prepare_block8_factors(h.tolist());u_new,b_new=block_ldl(h)
 old_res=float(np.max(abs(np.asarray(u_old)@np.asarray(b_old[0][0])-np.eye(8))))
 new_res=float(np.max(abs(u_new@b_new[0][0]-np.eye(8))))
 report={'status':'executed_CPU_reference_tests_not_Qwen_quality_validation','python':platform.python_version(),'numpy':np.__version__,'tests_run':res.testsRun,'tests_passed':res.testsRun-len(res.errors)-len(res.failures),'failures':len(res.failures),'errors':len(res.errors),'old_triangular_inverse_residual':old_res,'corrected_inverse_residual':new_res,'qera_rank64_int8':lowrank_cost(248320,5120,64),'source_sample_missing':not (ROOT/'sources'/'qwen38_real_slices_sample.json').exists(),'elapsed_seconds_cpu':time.perf_counter()-t}
 (ROOT/'results'/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 print(log.getvalue());print(json.dumps(report,indent=2))
 if not res.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
