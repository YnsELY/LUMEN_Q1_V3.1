from __future__ import annotations
import copy,hashlib,json,os,subprocess,sys,tempfile,time,unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from formic_lab.common import *
from formic_lab.checkpoint import header,inspect_checkpoint
from formic_lab.data import *
from formic_lab.models import *
from formic_lab.capture import *
from formic_lab.candidates import *
from formic_lab.lumen_bridge import quantize_linear,scalar_proxy
from formic_lab.metrics import *
from formic_lab.analysis import *
from formic_lab.recovery import recover_local
from formic_lab.runner import run_experiment,freeze,confirm,result_map
from formic_lab.supervisor import budget_status,supervise
from formic_lab.archive import pack_results
from formic_lab.release import check_release

torch.set_num_threads(1)
class TempTest(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

class CommonTests(TempTest):
    def test_integer_bool(self):
        with self.assertRaises(ValueError):integer(True,'x')
    def test_integer_numpy(self):self.assertEqual(integer(np.int64(5),'x'),5)
    def test_finite_nan(self):
        with self.assertRaises(ValueError):finite(float('nan'),'x')
    def test_atomic_json(self):
        write_json(self.p/'x.json',{'é':1});self.assertEqual(read_json(self.p/'x.json'),{'é':1})
    def test_atomic_no_overwrite(self):
        write_json(self.p/'x.json',{});
        with self.assertRaises(FileExistsError):write_json(self.p/'x.json',{},overwrite=False)
    def test_json_nonfinite_reject(self):
        with self.assertRaises(ValueError):write_json(self.p/'x.json',{'n':float('nan')})
        self.assertFalse((self.p/'x.json').exists())
    def test_path_traversal(self):
        with self.assertRaises(ValueError):contained(self.p,'../outside')
    def test_model_output_separation(self):
        with self.assertRaises(ValueError):separate_output(self.p/'model/work',self.p/'model')
    def test_safe_npz(self):
        with self.assertRaises(ValueError):save_npz(self.p/'x.npz',x=np.array([object()],dtype=object))
    def test_space_guard(self):
        with self.assertRaises(OSError):ensure_space(self.p,10**30,0)

class CheckpointTests(TempTest):
    def make(self,revision=EXPECTED_REVISION):
        write_json(self.p/'config.json',{'model_type':'test','_commit_hash':revision})
        save_file({'a':torch.zeros(2,3),'b':torch.zeros(2,dtype=torch.bfloat16)},str(self.p/'model.safetensors'))
    def test_headers_no_weight_load(self):
        self.make();r=inspect_checkpoint(self.p);self.assertEqual(r['parameter_count'],8);self.assertEqual(r['weight_bytes'],28)
        self.assertFalse(r['publisher_provenance_verified'])
    def test_revision_mismatch(self):
        self.make('bad')
        with self.assertRaises(ValueError):inspect_checkpoint(self.p)
    def test_no_revision_evidence(self):
        self.make();write_json(self.p/'config.json',{'model_type':'test'})
        self.assertIsNone(inspect_checkpoint(self.p)['revision_declared'])
    def test_index_checks(self):
        self.make();write_json(self.p/'model.safetensors.index.json',{'weight_map':{'a':'model.safetensors','b':'model.safetensors'}})
        self.assertEqual(inspect_checkpoint(self.p)['tensor_count'],2)
    def test_index_unsafe(self):
        self.make();write_json(self.p/'model.safetensors.index.json',{'weight_map':{'a':'../x.safetensors'}})
        with self.assertRaises(ValueError):inspect_checkpoint(self.p)
    def test_header_truncated(self):
        (self.p/'x.safetensors').write_bytes(b'abc')
        with self.assertRaises(ValueError):header(self.p/'x.safetensors')
    def test_identity_hashes(self):
        self.make();m={'repo_id':EXPECTED_REPO,'revision':EXPECTED_REVISION,'sha256':{n:sha256(self.p/n) for n in ('config.json','model.safetensors')}}
        write_json(self.p/'identity.json',m);r=inspect_checkpoint(self.p,self.p/'identity.json');self.assertTrue(r['trusted_manifest_hashes_checked'])
    def test_identity_tamper(self):
        self.make();write_json(self.p/'identity.json',{'repo_id':EXPECTED_REPO,'revision':EXPECTED_REVISION,'sha256':{'config.json':'bad','model.safetensors':'bad'}})
        with self.assertRaises(ValueError):inspect_checkpoint(self.p,self.p/'identity.json')
    def test_unknown_dtype(self):
        import struct
        h=json.dumps({'a':{'dtype':'X','shape':[1],'data_offsets':[0,1]}}).encode();(self.p/'x').write_bytes(struct.pack('<Q',len(h))+h+b'x')
        with self.assertRaises(ValueError):header(self.p/'x')

class DataTests(TempTest):
    def test_fixture_splits(self):self.assertEqual(validate_records(fixture_records())['documents'],12)
    def test_shared_source(self):
        d=fixture_records();d[4]['source_id']=d[0]['source_id']
        with self.assertRaises(ValueError):validate_records(d)
    def test_duplicate_content(self):
        d=fixture_records();d[4]['input_ids']=d[0]['input_ids']
        with self.assertRaises(ValueError):validate_records(d)
    def test_token_bounds(self):
        with self.assertRaises(ValueError):encode_records(fixture_records(),vocab_size=2)
    def test_text_no_tokenizer(self):
        d={'id':'a','source_id':'a','split':'cal','license':'user','text':'bonjour'}
        with self.assertRaises(ValueError):encode_records([d])
    def test_empty(self):
        with self.assertRaises(ValueError):validate_records([])
    def test_tokenized_duplicate(self):
        d=fixture_records();d[4]['input_ids'][:2]=d[0]['input_ids'][:2]
        with self.assertRaises(ValueError):encode_records(d,max_length=2)
    def test_license_required(self):
        d=fixture_records();del d[0]['license']
        with self.assertRaises(ValueError):validate_records(d)

class ModelCaptureTests(TempTest):
    def setUp(self):
        super().setUp();self.b=load_bundle({'backend':'toy','seed':5});self.docs=encode_records(fixture_records(docs_per_split=2),max_length=32,vocab_size=64)
        self.cal=[d for d in self.docs if d['split']=='cal']
    def test_seed_reproducibility(self):
        c=load_bundle({'backend':'toy','seed':5})
        with torch.no_grad():torch.testing.assert_close(c.forward([1,2,3],2),self.b.forward([1,2,3],2),rtol=0,atol=0)
    def test_scope_is_synthetic(self):self.assertTrue(self.b.metadata['synthetic'])
    def test_logits_cap(self):
        self.b.max_logit_elements=1
        with self.assertRaises(MemoryError):self.b.forward([1,2,3],1)
    def test_capture_width(self):
        cap=capture_module(self.b,'blocks.0.mlp.up_proj',self.cal,max_positions=20,per_document=10)
        self.assertEqual(cap.x.shape,(20,16));self.assertEqual(cap.y.shape,(20,32))
        w=self.b.model.blocks[0].mlp.up_proj.weight.detach().numpy()
        np.testing.assert_allclose(cap.y,cap.x@w.T,atol=1e-6)
    def test_capture_reproducible_positions(self):
        a=capture_module(self.b,'blocks.0.mlp',self.cal,max_positions=10);b=capture_module(self.b,'blocks.0.mlp',self.cal,max_positions=10)
        self.assertEqual(a.meta['sample_ids'],b.meta['sample_ids']);np.testing.assert_array_equal(a.x,b.x)
    def test_capture_cleanup(self):
        mod=self.b.model.blocks[0].mlp
        capture_module(self.b,'blocks.0.mlp',self.cal,max_positions=2)
        self.assertEqual(len(mod._forward_hooks),0)
    def test_capture_hash(self):
        a=capture_module(self.b,'blocks.0.mlp',self.cal,max_positions=2);save_capture(self.p,a);r=read_capture(self.p);np.testing.assert_array_equal(r.x,a.x)
        with (self.p/'arrays.npz').open('ab') as f:f.write(b'x')
        with self.assertRaises(ValueError):read_capture(self.p)
    def test_pair_alignment(self):
        a=capture_module(self.b,'blocks.0.mlp',self.cal,max_positions=2);b=copy.deepcopy(a);b.meta['sample_ids'][0]='bad'
        with self.assertRaises(ValueError):pair_captures(a,b)
    def test_surgery_restore_exception(self):
        old=self.b.model.blocks[0].mlp.up_proj;w=old.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with substitutions(self.b.model,[{'target':'blocks.0.mlp.up_proj','kind':'lowrank_linear','rank':8}]):raise RuntimeError('stop')
        self.assertIs(self.b.model.blocks[0].mlp.up_proj,old);torch.testing.assert_close(old.weight,w)
    def test_fullrank_linear_equivalence(self):
        torch.manual_seed(4);mod=nn.Linear(8,12);fac=lowrank_linear(mod,8);x=torch.randn(6,8)
        torch.testing.assert_close(fac(x),mod(x),rtol=1e-5,atol=1e-6)
    def test_prune_keepall_equivalent(self):
        mod=self.b.model.blocks[0].mlp;new=prune_swiglu(mod,list(range(32)));x=torch.randn(5,16)
        torch.testing.assert_close(new(x),mod(x),rtol=0,atol=0)
    def test_prune_coupled_dimensions(self):
        mod=prune_swiglu(self.b.model.blocks[0].mlp,[0,3,6,9]);self.assertEqual(mod.up_proj.out_features,4);self.assertEqual(mod.down_proj.in_features,4)
    def test_nested_targets_rejected(self):
        with self.assertRaises(ValueError):
            with substitutions(self.b.model,[{'target':'blocks.0.mlp','kind':'identity'},{'target':'blocks.0.mlp.up_proj','kind':'identity'}]):pass
    def test_plugin_candidate(self):
        old=self.b.model.blocks[0].mlp;spec={'kind':'plugin','factory':str(ROOT/'examples/custom_candidate.py')+':build','gain':1.}
        new,_=build_candidate(old,spec);x=torch.randn(5,16);torch.testing.assert_close(new(x),old(x))
    def test_plugin_backend(self):
        b=load_bundle({'backend':'plugin','factory':str(ROOT/'examples/custom_backend.py')+':build'});self.assertTrue(b.metadata['synthetic'])
    def test_svd_cap(self):
        with self.assertRaises(MemoryError):lowrank_linear(nn.Linear(16,32),8,max_elements=10)

class NumericTests(TempTest):
    def test_kl_identity(self):self.assertEqual(logits_metrics([[1.,2.,3.]],[[1.,2.,3.]])['kl_nats_mean'],0.)
    def test_kl_extreme(self):self.assertAlmostEqual(logits_metrics([[0.,0.,-50.]],[[0.,-800.,-800.]])['kl_nats_mean'],399.30685281944,places=8)
    def test_kl_badshape(self):
        with self.assertRaises(ValueError):logits_metrics([[1.,2.]],[[1.]])
    def test_kl_nonfinite(self):
        with self.assertRaises(ValueError):logits_metrics([[float('nan'),2]],[[1,2]])
    def test_document_not_tokens_bootstrap(self):
        r=document_summary([{'kl_nats_mean':1.,'positions':100},{'kl_nats_mean':3.,'positions':1}]);self.assertEqual(r['document_mean_kl'],2.);self.assertEqual(r['positions'],101)
    def test_paired_reject(self):
        a=[{'id':'a','token_hash':'x','positions':1,'kl_nats_mean':1.}];b=[{**a[0],'token_hash':'y'}]
        with self.assertRaises(ValueError):paired_difference(a,b)
    def test_redundancy_rankone(self):
        d=redundancy(np.ones((4,8)));self.assertAlmostEqual(d['energy_effective_rank'],1.)
    def test_redundancy_cap(self):
        with self.assertRaises(MemoryError):redundancy(np.ones((4,8)),max_dimension=2)
    def test_sensitivity_identity(self):self.assertEqual(sensitivity(np.eye(3),np.eye(3),np.ones((5,3)))['nmse'],0.)
    def test_lumen_encode_decode(self):
        torch.manual_seed(2);mod=nn.Linear(8,4,bias=True);x=np.random.default_rng(5).normal(size=(32,8))
        q,r=quantize_linear(mod,x,self.p/'q');self.assertEqual(r['matrix_blob_bytes'],(self.p/'q/weight.lmat').stat().st_size)
        self.assertGreater(r['bias_file_bytes'],0);self.assertFalse(r['gpu_codec_implemented'])
        from lumen_ref.entropy import decode_matrix
        w,*_=decode_matrix((self.p/'q/weight.lmat').read_bytes());np.testing.assert_allclose(q.weight.detach().numpy(),w,rtol=1e-6)
    def test_lumen_cap(self):
        with self.assertRaises(MemoryError):quantize_linear(nn.Linear(8,4),np.ones((8,8)),self.p,max_elements=2)
    def test_lumen_width_reject(self):
        with self.assertRaises(ValueError):quantize_linear(nn.Linear(9,4),np.ones((8,9)),self.p)
    def test_proxy_label(self):
        _,r=scalar_proxy(nn.Linear(8,4),np.ones((8,8)),self.p);self.assertEqual(r['method'],'UNIFORM_SCALAR_PROXY_NOT_LUMEN')
    def test_recovery_local(self):
        torch.manual_seed(6);m=nn.Linear(4,3,bias=False);old=m.weight.detach().clone();rng=np.random.default_rng(4)
        x=rng.normal(size=(80,4)).astype('float32');y=x@np.ones((4,3),dtype='float32')
        new,r=recover_local(m,x[:60],y[:60],x[60:],y[60:],steps=8,lr=.02)
        self.assertLess(r['best_float32_select_mse'],r['initial_select_mse']);torch.testing.assert_close(m.weight,old)
    def test_recovery_cap(self):
        with self.assertRaises(MemoryError):recover_local(nn.Linear(4,3),np.ones((4,4)),np.ones((4,3)),np.ones((4,4)),np.ones((4,3)),max_trainable_params=1)

class BudgetLifecycleTests(TempTest):
    def cfg(self):return {'pod_billing_start_utc':(datetime.now(timezone.utc)-timedelta(hours=2)).isoformat(),'hourly_total_usd':1.,'remaining_budget_at_start_usd':9.,'reserve_usd':1.,'max_pod_seconds':21600}
    def test_cost_counts_idle_since_start(self):self.assertAlmostEqual(budget_status(self.cfg())['estimated_spent_usd'],2.,places=4)
    def test_cost_nonfinite(self):
        c=self.cfg();c['hourly_total_usd']=float('nan')
        with self.assertRaises(ValueError):budget_status(c)
    def test_cost_future(self):
        c=self.cfg();c['pod_billing_start_utc']=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
        with self.assertRaises(ValueError):budget_status(c)
    def test_supervisor_normal(self):
        r=supervise([sys.executable,'-c','print("done")'],self.p,self.p/'logs',self.cfg(),poll_seconds=.02)
        self.assertEqual(r['exit_code'],0);self.assertFalse(r['pod_stopped']);self.assertFalse(r['billing_stopped'])
    def test_supervisor_timeout(self):
        c=self.cfg();c['pod_billing_start_utc']=datetime.now(timezone.utc).isoformat();c['max_pod_seconds']=.25
        r=supervise([sys.executable,'-c','import time;time.sleep(30)'],self.p,self.p/'logs',c,poll_seconds=.02,grace_seconds=.1)
        self.assertEqual(r['reason'],'budget_or_walltime_guard');self.assertLess(r['elapsed_seconds'],4)
    def test_budget_expired_no_launch(self):
        c=self.cfg();c['max_pod_seconds']=1
        with self.assertRaises(TimeoutError):supervise([sys.executable,'-c','print(1)'],self.p,self.p/'logs',c)
    def test_archive_exclusions(self):
        run=self.p/'run';run.mkdir();write_json(run/'a.json',{});(run/'x.npz').write_bytes(b'abc')
        r=pack_results(run,self.p/'r.tar.gz');self.assertEqual(r['included_files'],1);self.assertFalse(r['complete_for_reproduction'])
    def test_archive_no_inside_output(self):
        with self.assertRaises(ValueError):pack_results(self.p,self.p/'x.tar.gz')

class RunnerTests(TempTest):
    def cfg(self):
        c=read_json(ROOT/'configs/demo_cpu.json');c['variants']=c['variants'][:2];c['fixture_data']['docs_per_split']=2
        p=self.p/'config.json';write_json(p,c);return p
    def test_end_to_end_and_resume(self):
        p=self.cfg();r=self.p/'run';s=run_experiment(p,r);self.assertEqual(s['status'],'completed')
        before=read_json(r/'state.json')['completed'];run_experiment(p,r,resume=True);self.assertEqual(read_json(r/'state.json')['completed'],before)
    def test_no_overwrite(self):
        p=self.cfg();r=self.p/'run';run_experiment(p,r)
        with self.assertRaises(FileExistsError):run_experiment(p,r)
    def test_resume_config_change(self):
        p=self.cfg();r=self.p/'run';run_experiment(p,r);c=read_json(p);c['keep_last']=2;write_json(p,c)
        with self.assertRaises(ValueError):run_experiment(p,r,resume=True)
    def test_tamper_detected(self):
        p=self.cfg();r=self.p/'run';s=run_experiment(p,r);folder=r/s['completed']['A0_full']['directory'];(folder/'result.json').write_text('{}')
        with self.assertRaises(ValueError):result_map(r)
    def test_confirm_requires_freeze(self):
        p=self.cfg();r=self.p/'run';run_experiment(p,r)
        with self.assertRaises(FileNotFoundError):confirm(r)
    def test_confirm_locked(self):
        p=self.cfg();r=self.p/'run';run_experiment(p,r);freeze(r,['A0_local_E8']);a=confirm(r);self.assertTrue(a['synthetic'])
        with self.assertRaises(FileExistsError):confirm(r)
    def test_capture_export_schema(self):
        b=load_bundle({'backend':'toy'});docs=encode_records(fixture_records(docs_per_split=1));pairs={}
        for s in SPLITS:
            a=capture_module(b,'blocks.0.mlp.up_proj',[r for r in docs if r['split']==s],max_positions=5);pairs[s]=(a,a)
        m=export_lumen_schema2(pairs,self.p/'bridge',runtime_commit='abcdef12',projection_kind='mlp_up',weight=b.model.blocks[0].mlp.up_proj.weight.detach().numpy())
        self.assertEqual(m['schema_version'],2);self.assertEqual(m['source_kind'],'synthetic_fixture')
    def test_loader_cli_from_other_directory(self):
        p=self.cfg();out=self.p/'load.json';r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/load_model.py'),'--config',str(p),'--out',str(out)],cwd=self.p,capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr);self.assertTrue(read_json(out)['finite'])

if __name__=='__main__':unittest.main()
