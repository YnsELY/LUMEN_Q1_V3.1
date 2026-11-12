#!/usr/bin/env python3
"""Reopen the six saved streams in a fresh process and verify source codes.

No encoder is called here. The known synthetic sources are regenerated, not
recovered from a hidden in-memory object used by the encoder.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from lumen_ref import e8
from lumen_ref.entropy import decode_e8

def main():
    rng=np.random.default_rng(20261112)
    sources={'uniform_shell1':e8.ROOTS[rng.integers(240,size=8192)],
             'integer_coset_shell1':e8.ROOTS[rng.integers(112,size=8192)],
             'gaussian_E8':np.array([e8.nearest(v) for v in rng.normal(size=(8192,8))])}
    records=[]
    for label,source in sources.items():
        for mode in (0,1):
            p=ROOT/'results'/f'codec_{label}_mode{mode}.bin';data=p.read_bytes()
            actual,model=decode_e8(data)
            if model.mode!=mode or not np.array_equal(actual,source):
                raise ArithmeticError('Cold stream validation failed: '+p.name)
            records.append({'file':p.name,'vectors':len(actual),'bytes':len(data),
                            'sha256':hashlib.sha256(data).hexdigest(),'errors':0})
    result={'fresh_process_decode':True,'no_encoder_called':True,
            'source':'synthetic, regenerated with recorded seed; not Qwen',
            'total_vector_reconstructions':sum(x['vectors'] for x in records),'files':records}
    (ROOT/'results'/'cold_decode_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
