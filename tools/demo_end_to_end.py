#!/usr/bin/env python3
"""Small SYNTHETIC CPU smoke test: calibrate, encode, decode, score, accept/reject."""
from pathlib import Path
import argparse,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from lumen_ref.calibration import asymmetric_target
from lumen_ref.quantization import quantize_ldlq
from lumen_ref.entropy import decode_matrix
from lumen_ref.metrics import compare_matrix_artifacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(7001);w=rng.normal(size=(8,16));xf=rng.normal(size=(16,128));xq=xf+.1*rng.normal(size=xf.shape)
    h=xq@xq.T/128+.01*np.eye(16);target,_=asymmetric_target(w,xf,xq,.01)
    prev=quantize_ldlq(w,h,.85,.0001);cand=quantize_ldlq(target,h,.85,.0001)
    vf=rng.normal(size=(16,128));vq=vf+.1*rng.normal(size=vf.shape);y=w@vf
    def evaluator(weight):return (float(np.mean((weight@vq-y)**2)),{})
    chosen,decision=compare_matrix_artifacts(prev.artifact,cand.artifact,evaluator,len(prev.artifact))
    for name,blob in (('control.lmat',prev.artifact),('candidate.lmat',cand.artifact),('retained.lmat',chosen)):
        (out/name).write_bytes(blob);decode_matrix((out/name).read_bytes())
    report={'status':'executed_synthetic_smoke_test_not_Qwen','decision':decision,'ceiling_bytes':len(prev.artifact),'candidate_bytes':len(cand.artifact),'retained_bytes':len(chosen),'retained_sha256':hashlib.sha256(chosen).hexdigest()}
    (out/'demo.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
