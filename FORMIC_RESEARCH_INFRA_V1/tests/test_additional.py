from __future__ import annotations
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from formic_lab.common import *
from formic_lab.models import *
from formic_lab.candidates import *
from formic_lab.capture import *
from formic_lab.data import *
from formic_lab.metrics import *
from formic_lab.runner import *

class AdditionalTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_source_cluster_ci(self):
        rows=[{'id':'a','source_id':'one','kl_nats_mean':1,'positions':2},{'id':'b','source_id':'one','kl_nats_mean':9,'positions':3}]
        r=document_summary(rows);self.assertIsNone(r['bootstrap_document_mean_95pct']);self.assertEqual(r['source_clusters'],1)
    def test_plugin_receives_copy(self):
        f=self.p/'plugin.py';f.write_text('import torch\ndef build(m,s):\n with torch.no_grad(): m.weight.zero_()\n return m\n')
        original=torch.nn.Linear(4,3);before=original.weight.detach().clone();new,_=build_candidate(original,{'kind':'plugin','factory':str(f)+':build'})
        torch.testing.assert_close(original.weight,before);self.assertEqual(torch.count_nonzero(new.weight),0)
    def test_capture_hook_cleanup_on_forward_error(self):
        b=load_bundle({'backend':'toy'});b.max_logit_elements=1;docs=encode_records(fixture_records())
        with self.assertRaises(MemoryError):capture_module(b,'blocks.0.mlp',docs)
        self.assertEqual(len(b.model.blocks[0].mlp._forward_hooks),0)
    def test_state_restored_after_all_variants(self):
        c=read_json(ROOT/'configs/demo_cpu.json');c['fixture_data']['docs_per_split']=1;c['variants'].append({'id':'A0_after','changes':[]})
        write_json(self.p/'c.json',c);run_experiment(self.p/'c.json',self.p/'run');r=result_map(self.p/'run')
        self.assertEqual(r['A0_after']['summary']['document_mean_kl'],0.)
    def test_local_recovery_runner(self):
        c=read_json(ROOT/'configs/demo_cpu.json');c['fixture_data']['docs_per_split']=1
        c['variants']=[{'id':'rec','changes':[{'kind':'lowrank_linear','target':'blocks.0.mlp.up_proj','rank':8}],
                        'recovery':{'target':'blocks.0.mlp.up_proj','options':{'steps':2,'max_seconds':10}}}]
        write_json(self.p/'c.json',c);run_experiment(self.p/'c.json',self.p/'run');r=result_map(self.p/'run')
        self.assertEqual(r['rec']['recovery']['steps_executed'],2)
        freeze(self.p/'run',['rec']);self.assertEqual(confirm(self.p/'run')['status'],'confirmation_consumed')
    def test_schema2_original_validator(self):
        b=load_bundle({'backend':'toy'});docs=encode_records(fixture_records(docs_per_split=1));pairs={}
        for s in SPLITS:
            cap=capture_module(b,'blocks.0.mlp.up_proj',[d for d in docs if d['split']==s],max_positions=5);pairs[s]=(cap,cap)
        export_lumen_schema2(pairs,self.p/'bridge',runtime_commit='abcdef12',weight=b.model.blocks[0].mlp.up_proj.weight.detach().numpy())
        file=ROOT/'reference/validate_capture_v3_1.py';sp=importlib.util.spec_from_file_location('reference_validator',file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
        result=m.validate(self.p/'bridge/capture.npz',self.p/'bridge/capture.json',allow_synthetic=True)
        self.assertTrue(result['schema_valid']);self.assertFalse(result['provenance_verified'])
    def test_hf_offline_contract_mock(self):
        # Tests only our arguments/guards. Does NOT test Transformers or Qwen.
        write_json(self.p/'config.json',{'_commit_hash':EXPECTED_REVISION})
        save_file({'a':torch.zeros(8)},str(self.p/'model.safetensors'))
        calls={}
        class ModelClass:
            @staticmethod
            def from_pretrained(*args,**kwargs):calls['model']=kwargs;return ToyLM()
        class Tok:
            @staticmethod
            def from_pretrained(*args,**kwargs):calls['tokenizer']=kwargs;return object()
        fake=SimpleNamespace(__version__='MOCK_NOT_VALIDATED',AutoModelForImageTextToText=ModelClass,AutoTokenizer=Tok)
        with patch.dict(sys.modules,{'transformers':fake}):
            b=load_bundle({'backend':'hf_local','path':str(self.p),'device':'cpu','expected_parameter_count':8})
        self.assertTrue(calls['model']['local_files_only']);self.assertFalse(calls['model']['trust_remote_code']);self.assertTrue(calls['model']['use_safetensors'])
        self.assertEqual(b.metadata['transformers_version'],'MOCK_NOT_VALIDATED')
    def test_hf_no_identity_failclosed(self):
        write_json(self.p/'config.json',{});save_file({'a':torch.zeros(8)},str(self.p/'model.safetensors'))
        with self.assertRaisesRegex(ValueError,'No pinned revision'):load_bundle({'backend':'hf_local','path':str(self.p),'device':'cpu'})
    def test_hf_wrong_inventory_failclosed(self):
        write_json(self.p/'config.json',{'_commit_hash':EXPECTED_REVISION});save_file({'a':torch.zeros(8)},str(self.p/'model.safetensors'))
        with self.assertRaisesRegex(ValueError,'inventory differs'):load_bundle({'backend':'hf_local','path':str(self.p),'device':'cpu'})
    def test_parameter_shared_storage(self):
        m=torch.nn.Module();m.a=torch.nn.Parameter(torch.ones(8));m.b=torch.nn.Parameter(m.a.data)
        d=inspect_model(m);self.assertEqual(d['parameter_storage_bytes'],32)
    def test_fullrank_candidate_confirmation(self):
        c=read_json(ROOT/'configs/demo_cpu.json');c['fixture_data']['docs_per_split']=1
        c['variants']=[{'id':'equivalence','changes':[{'kind':'lowrank_linear','target':'blocks.0.mlp.up_proj','rank':16}]}]
        write_json(self.p/'c.json',c);run_experiment(self.p/'c.json',self.p/'run');freeze(self.p/'run',['equivalence'])
        r=confirm(self.p/'run');self.assertLess(r['results']['equivalence']['summary']['document_mean_kl'],1e-10)
    def test_corpus_bytes_guard(self):
        write_json(self.p/'c.json',fixture_records())
        with self.assertRaises(MemoryError):load_records(self.p/'c.json',max_bytes=1)

if __name__=='__main__':unittest.main()
