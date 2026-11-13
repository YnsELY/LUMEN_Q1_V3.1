"""Exercise the user-facing commands without a GPU or external network."""
from pathlib import Path
import contextlib,io,json,sys,tempfile,unittest
from datetime import datetime,timezone
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from formic_lab.cli import main
from formic_lab.common import write_json,save_npz,EXPECTED_REVISION
from formic_lab.data import fixture_records

class CliSmokeTests(unittest.TestCase):
    def test_working_commands(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=ROOT/'configs/demo_cpu.json';model=p/'model';model.mkdir()
            write_json(model/'config.json',{'_commit_hash':EXPECTED_REVISION});save_file({'w':torch.zeros(2,8)},str(model/'model.safetensors'))
            records=fixture_records();write_json(p/'corpus.json',records)
            rng=np.random.default_rng(42);w=rng.normal(size=(4,8)).astype('float32');x=rng.normal(size=(32,8)).astype('float32');y=x@w.T
            save_npz(p/'inputs.npz',W=w,Q=w*.9,X=x,X_cal=x[:20],Y_cal=y[:20],X_select=x[20:],Y_select=y[20:],teacher_logits=y,student_logits=y*.9)
            write_json(p/'budget.json',{'pod_billing_start_utc':datetime.now(timezone.utc).isoformat(),
                         'hourly_total_usd':1.,'remaining_budget_at_start_usd':9.,'reserve_usd':1.,'max_pod_seconds':600})
            commands=[
                ['environment','--workdir',str(p),'--out',str(p/'env.json')],
                ['checkpoint','--path',str(model),'--out',str(p/'ckpt.json')],
                ['inspect','--config',str(cfg),'--out',str(p/'arch.json')],
                ['load','--config',str(cfg),'--out',str(p/'load.json')],
                ['prepare-data','--input',str(p/'corpus.json'),'--out',str(p/'checked.json')],
                ['capture','--config',str(cfg),'--target','blocks.0.mlp.up_proj','--out',str(p/'capture')],
                ['redundancy','--npz',str(p/'inputs.npz'),'--out',str(p/'redundancy.json')],
                ['sensitivity','--npz',str(p/'inputs.npz'),'--out',str(p/'sensitivity.json')],
                ['candidates','--grid',str(ROOT/'examples/candidate_grid.json'),'--out',str(p/'candidates.json')],
                ['apply','--config',str(cfg),'--variant','A1_example_rank8','--out',str(p/'candidate')],
                ['quantize','--npz',str(p/'inputs.npz'),'--out',str(p/'quant')],
                ['recover','--npz',str(p/'inputs.npz'),'--steps','2','--out',str(p/'recovery')],
                ['evaluate','--npz',str(p/'inputs.npz'),'--out',str(p/'fidelity.json')],
                ['run','--config',str(cfg),'--out',str(p/'run')],
                ['compare','--run',str(p/'run'),'--control','A0_local_E8','--candidate','A1_example_rank8_E8','--out',str(p/'comparison.json')],
                ['report','--run',str(p/'run')],
                ['freeze','--run',str(p/'run'),'--ids','A0_local_E8','A1_example_rank8_E8'],
                ['confirm','--run',str(p/'run')],
                ['archive','--run',str(p/'run'),'--dest',str(p/'results.tar.gz')],
                ['monitor','--budget',str(p/'budget.json'),'--cwd',str(p),'--poll','.02','--out',str(p/'logs'),'--',sys.executable,'-c','print("local smoke, no pod")'],
            ]
            for cmd in commands:
                with self.subTest(command=cmd[0]),contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(cmd),0)
            self.assertTrue((p/'run/confirmation.json').exists());self.assertTrue((p/'quant/weight.lmat').exists())
if __name__=='__main__':unittest.main()
