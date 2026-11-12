"""Positive regression tests: bugs must be absent, not merely reproducible."""
from pathlib import Path
import sys, tempfile, json, hashlib, itertools, unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from lumen_ref import linalg as la, calibration as ca, metrics as me, entropy as en, e8, budget as bu, quantization as qu
import validate_capture as vc
import verify_manifest as vm
from independent_decoder import independent_decode_le8cpu3, independent_decode_lmatv3

class MetricsRegressions(unittest.TestCase):
    def test_duplicate_ids_rejected_both_metrics(self):
        z=np.array([[1.,0.,0.]])
        cache=me.topk_cache(z,2);cache['ids']=np.array([[0,0]])
        with self.assertRaises(ValueError):me.grouped_kl_from_cache(cache,z)
        with self.assertRaises(ValueError):me.exact_grouped_kl(z,z,cache['ids'])
    def test_noninteger_or_bool_ids_rejected(self):
        for ids in (np.array([[0.,1.]]),np.array([[False,True]])):
            with self.assertRaises(ValueError):me.exact_grouped_kl([[0.,0.,0.]],[[0.,0.,0.]],ids)
    def test_logspace_minus800(self):
        ans=me.exact_grouped_kl([[0.,0.,-50.]],[[0.,-800.,-800.]],np.array([[0,1]]))[0]
        self.assertAlmostEqual(ans,400-np.log(2),places=10)
    def test_masked_teacher_and_zero_tail(self):
        z=np.array([[0.,-np.inf,-np.inf]])
        self.assertEqual(me.kl_logits(z,z)[0],0.)
        self.assertEqual(me.grouped_kl_from_cache(me.topk_cache(z,2),z)[0],0.)
        self.assertEqual(me.exact_grouped_kl(z,z,np.array([[0,1]]))[0],0.)
    def test_positive_teacher_zero_student_support(self):
        self.assertTrue(np.isposinf(me.kl_logits([[0.,0.,-np.inf]],[[0.,-np.inf,-np.inf]])[0]))
    def test_all_masked_nan_plusinf_invalid(self):
        for z in ([[np.nan,0.]],[[np.inf,0.]],[[-np.inf,-np.inf]],np.zeros((0,3))):
            with self.assertRaises(ValueError):me.log_softmax(z)
    def test_cache_overflow_promotes_does_not_clip(self):
        z=np.array([[0.,-70000.,-80000.]])
        c=me.topk_cache(z,2)
        self.assertEqual(c['log_probs'].dtype,np.dtype('float32'))
        self.assertEqual(float(c['log_probs'][0,0]),-70000.)
        self.assertEqual(c['actual_array_bytes'],20)
        self.assertEqual(me.grouped_kl_from_cache(c,z)[0],0.)
    def test_cache_overflow_raise_mode(self):
        with self.assertRaises(ValueError):me.topk_cache([[0.,-70000.,-80000.]],2,overflow_policy='raise')
    def test_fp16_bias_remains_reported_not_promised_away(self):
        z=np.random.default_rng(3827).normal(scale=2.5,size=(1,248320))
        c=me.topk_cache(z,128)
        self.assertGreater(me.grouped_kl_from_cache(c,z)[0],1e-7)
        exact=me.topk_cache(z,128,value_dtype=np.float64)
        self.assertLess(me.grouped_kl_from_cache(exact,z)[0],1e-14)
    def test_cache_invalid_mass_rejected(self):
        c=me.topk_cache([[1.,0.,0.]],2);c['log_probs'][:]=0.;c['log_tail_mass'][:]=0.
        with self.assertRaises(ValueError):me.grouped_kl_from_cache(c,[[1.,0.,0.]])
    def test_grouped_kl_is_lower_bound_on_random_cases(self):
        rng=np.random.default_rng(501)
        for _ in range(10):
            p=rng.normal(size=(5,17));q=rng.normal(size=(5,17));ids=np.argsort(p,axis=1)[:,-4:]
            full=me.kl_logits(p,q);small=me.exact_grouped_kl(p,q,ids)
            self.assertTrue(np.all(small>=0));self.assertTrue(np.all(small<=full+1e-12))
    def test_guard_invalid_budget_and_empty(self):
        a=me.ArtifactScore(b'abc',1.,{'s':1.});b=me.ArtifactScore(b'',.9,{'s':1.})
        for limit in (float('nan'),True,3.5):
            with self.assertRaises(ValueError):me.accept_decoded(a,b,limit)
        self.assertFalse(me.accept_decoded(a,b,10)[1]['accepted'])
    def test_guard_nonfinite_even_nonrequired_sentinel(self):
        a=me.ArtifactScore(b'a',1.,{'s':1.});b=me.ArtifactScore(b'b',.9,{'s':1.,'new':float('nan')})
        self.assertFalse(me.accept_decoded(a,b,10,required_sentinels=[])[1]['accepted'])
    def test_equal_score_smaller_can_be_kept(self):
        a=me.ArtifactScore(b'ab',1.,{});b=me.ArtifactScore(b'a',1.,{})
        self.assertTrue(me.accept_decoded(a,b,3)[1]['accepted'])
    def test_scored_state_cannot_mutate(self):
        d={'s':1.};raw=bytearray(b'ab');a=me.ArtifactScore(raw,1.,d)
        d['s']=float('nan');raw[0]=0
        self.assertEqual(a.sentinels['s'],1.);self.assertEqual(a.data,b'ab')
        with self.assertRaises(TypeError):a.sentinels['s']=0
    def test_decode_and_evaluate_actual_artifacts(self):
        c=np.zeros((1,1,8),dtype=np.int64);a=en.encode_matrix(c,[1.]);b=en.encode_matrix(e8.ROOTS[:1].reshape(1,1,8),[1.])
        scorer=lambda w:(float(np.sum(w*w)),{})
        kept,info=me.compare_matrix_artifacts(a,b,scorer,max(len(a),len(b)))
        self.assertEqual(kept,a);self.assertFalse(info['accepted']);self.assertTrue(info['decoder_checked'])
        self.assertFalse(me.compare_matrix_artifacts(a,b'bad',scorer,len(a))[1]['accepted'])

class NumericalRegressions(unittest.TestCase):
    def test_symmetric_part_is_used(self):
        a=np.array([[2.,1.+1e-12],[1.-1e-12,2.]])
        out=la.checked_spd(a);self.assertTrue(np.array_equal(out,out.T))
        u,d=la.upper_ldl(a);np.testing.assert_allclose((u*d)@u.T,(a+a.T)/2,atol=1e-14)
    def test_indefinite_symmetric_part_is_rejected(self):
        a=np.array([[1.,1.+4.5e-12],[1.-4.5e-12,1.-8.5e-12]])
        with self.assertRaises((ValueError,np.linalg.LinAlgError)):la.checked_spd(a)
    def test_tiny_rhs_beyond_proposed_patch(self):
        for scale in (1e-160,1e-200,1e-300):
            b=np.full((2,2),scale);ans=la.pcg(lambda v:1e8*v,b,np.full((2,2),1e8))
            self.assertTrue(ans.converged);self.assertLess(ans.relative_residual,1e-10)
            np.testing.assert_allclose(ans.solution/(scale/1e8),np.ones((2,2)),atol=1e-14)
    def test_huge_rhs_avoids_naive_norm_overflow(self):
        ans=la.pcg(lambda v:2*v,np.array([1e200,-1e200]),np.array([2.,2.]))
        self.assertTrue(ans.converged);np.testing.assert_array_equal(ans.solution,np.array([5e199,-5e199]))
    def test_no_premature_false_return(self):
        d=np.array([[1.,-1.,.5,-.5]]);g=np.array([[1.,.9,1.1],[1.,.9,1.1],[.8,1.2,.7],[.8,1.2,.7]])
        x=np.array([[1.,-.5,.3],[.2,1.1,-.7]]);op,diag=ca.bgcc_output_operator(x,g,d,1e-11)
        b=np.array([[1.,1.],[.5,-.3],[-.7,-.7],[.2,.4]])
        ans=la.pcg(op,b,diag,rtol=1e-9,maxiter=200)
        self.assertTrue(ans.converged or ans.iterations==200)
        rel=la.stable_norm(b-op(ans.solution))/la.stable_norm(b)
        self.assertAlmostEqual(ans.relative_residual,rel,places=8)
        if ans.converged:self.assertLessEqual(rel,1.1e-9)
    def test_invalid_cg_shapes_and_criteria(self):
        for kwargs in ({'maxiter':True},{'maxiter':2.5},{'rtol':float('nan')}):
            with self.assertRaises(ValueError):la.pcg(lambda x:x,np.ones(2),**kwargs)
        with self.assertRaises(ValueError):la.pcg(lambda x:np.ones((2,1)),np.ones(2))
    def test_nan_regularizers(self):
        with self.assertRaises(ValueError):ca.asymmetric_target(np.ones((2,8)),np.ones((8,3)),np.ones((8,3)),np.nan)
        with self.assertRaises(ValueError):ca.bgcc_row_target(np.ones(8),np.ones((8,3)),np.ones(3),np.ones(3),np.nan)
        with self.assertRaises(ValueError):ca.fresh_quadratic_target(np.ones((2,8)),np.ones((2,8)),np.eye(8),np.eye(2),np.nan)
    def test_ridge_target_stationarity(self):
        rng=np.random.default_rng(120);w=rng.normal(size=(3,8));xf=rng.normal(size=(8,32));xq=xf+.1*rng.normal(size=xf.shape)
        t,h=ca.asymmetric_target(w,xf,xq,.2)
        grad=((t@xq-w@xf)@xq.T)/32+.2*(t-w)
        self.assertLess(np.max(np.abs(grad)),1e-14)
    def test_bgcc_independent_dense_rankdeficient(self):
        rng=np.random.default_rng(77);m,n,N=3,8,20;x=rng.normal(size=(n,1))@rng.normal(size=(1,N));g=la.silu(rng.normal(size=(m,N)));d=rng.normal(size=(5,m));w=rng.normal(size=(m,n));y=rng.normal(size=(5,N));rho=.05
        target,cg=ca.bgcc_output_cg(w,x,g,d,y,ridge=rho,rtol=1e-13,maxiter=400)
        A=rho*np.eye(m*n)
        for t in range(N):A+=np.kron((d.T@d)*np.outer(g[:,t],g[:,t]),np.outer(x[:,t],x[:,t]))/N
        rhs=((g*(d.T@(y-d@(g*(w@x)))))@x.T/N).ravel()
        self.assertTrue(cg.converged);np.testing.assert_allclose(target,w+np.linalg.solve(A,rhs).reshape(m,n),atol=1e-8)
    def test_wavefront_dense_kronecker(self):
        rng=np.random.default_rng(441);w=rng.normal(size=(3,16));a=rng.normal(size=(16,16));h=a@a.T+np.eye(16);a=rng.normal(size=(3,3));g=a@a.T+np.eye(3);model=en.ShellModel.gaussian();lam=.03
        q=qu.quantize_ldlq(w,h,.8,lam,model,g);uk,fac=la.block_ldl(np.kron(g,h));flat=w.ravel();z=np.zeros_like(flat)
        for i in range(0,len(flat),8):
            inv,M=fac[i//8];center=flat[i:i+8]+((flat[:i]-z[:i])@uk[:i,i:i+8])@inv
            code,_=qu.select_code(center,q.steps[i//16],M,lam,model);z[i:i+8]=code*q.steps[i//16]/2
        np.testing.assert_array_equal(q.weights.ravel(),z)

class CodecRegressions(unittest.TestCase):
    def test_independent_reader_on_all_saved_streams(self):
        files=list((ROOT/'results').glob('codec_*.bin'));self.assertEqual(len(files),6)
        for p in files:
            independent,_=independent_decode_le8cpu3(p.read_bytes());actual,_=en.decode_e8(p.read_bytes());np.testing.assert_array_equal(actual,independent)
    def test_independent_reader_on_new_extremes(self):
        codes=list(e8.ROOTS)+[[0]*8,[-2**31,0,0,0,0,0,0,0],[40,20,12,4,0,0,0,0]]
        for mode in (0,1):
            blob=en.encode_e8(codes,en.ShellModel.fit(codes,mode=mode),group_size=17)
            np.testing.assert_array_equal(independent_decode_le8cpu3(blob)[0],codes)
    def test_matrix_empty_rejected_by_encoder(self):
        for shape in ((0,2,8),(2,0,8)):
            with self.assertRaises(ValueError):en.encode_matrix(np.zeros(shape,dtype=int),np.ones(shape[0],dtype=np.float16))
    def test_scale_not_silently_rounded(self):
        codes=e8.ROOTS[:2].reshape(1,2,8)
        with self.assertRaises(ValueError):en.encode_matrix(codes,[3e-8])
        s=np.array([np.float16(3e-8)],dtype=float);raw=en.encode_matrix(codes,s)
        np.testing.assert_array_equal(independent_decode_lmatv3(raw)[2],s)
    def test_e8_int_types_and_signed_boundary(self):
        for sign in (-1,1):self.assertTrue(e8.valid(e8.nearest([sign*1e8]+[0]*7)))
        self.assertFalse(e8.valid([True]*8));self.assertTrue(e8.valid([1<<64]*8))
        self.assertEqual(e8.shell_size(np.int64(1)),240)
        with self.assertRaises(ValueError):e8.shell_size(True)
    def test_freq_and_limits_strict(self):
        for c in ([True,1],[.5,1],[-1,1]):
            with self.assertRaises(ValueError):en.normalized_frequencies(c)
        with self.assertRaises(ValueError):en.encode_e8([],group_size=True)
        with self.assertRaises(ValueError):en.decode_e8(en.encode_e8([]),max_vectors=np.nan)
    def test_refinement_preserves_nondefault_codec_group(self):
        rng=np.random.default_rng(500);w=rng.normal(size=(4,32));h=np.eye(32)
        q=qu.quantize_ldlq(w,h,group_size=3)
        z,log=qu.refine_quadratic(w,q,h,preserve_shell=True,sweeps=2)
        self.assertEqual(log['group_size'],3);self.assertFalse(log['table_refitted_after_sweep'])
        self.assertEqual(len(z.artifact),len(q.artifact))
        x=rng.normal(size=(32,32));g=la.silu(rng.normal(size=(4,32)));d=rng.normal(size=(3,4));y=d@(g*(w@x))
        z,log=ca.refine_swiglu_up(q,x,g,d,y,preserve_shell=True)
        self.assertEqual(log['group_size'],3);self.assertEqual(len(z.artifact),len(q.artifact))
    def test_refinement_rejects_inconsistent_initial_object(self):
        q=qu.quantize_ldlq(np.ones((2,8)),np.eye(8));q.weights[0,0]+=1
        with self.assertRaises(ValueError):qu.refine_quadratic(np.ones((2,8)),q,np.eye(8))
    def test_rank_tree_same_as_mixed_integer_reference(self):
        rng=np.random.default_rng(501);radices=list(map(int,rng.integers(1,1000,size=4096)));ranks=[int(rng.integers(r)) for r in radices]
        blob=en._pack_ranks(ranks,radices,4096);self.assertEqual(en._unpack_ranks(blob,radices,4096),ranks)

class BudgetRegressions(unittest.TestCase):
    def test_floor_budget_strict(self):
        with self.assertRaises(ValueError):bu.continuous_waterfill([100,200],[1,4],600-5e-6,floors=2,ceilings=8)
        r=bu.continuous_waterfill([100,200],[1,4],600,floors=2,ceilings=8);self.assertLessEqual(np.dot([100,200],r),600)
    def test_waterfill_overflow_empty_invalid(self):
        for args in (([1e307,1e307],[1,1],1e308),([],[],10)):
            with self.assertRaises(ValueError):bu.continuous_waterfill(*args)
    def test_strict_integer_bytes(self):
        for v in (True,np.nan,1.5):
            with self.assertRaises(ValueError):bu.allocate_options({},v)
        with self.assertRaises(ValueError):bu.allocate_options({'g':[bu.Option('a',True,1)]},5)
        self.assertEqual(bu.allocate_options({'g':[bu.Option('a',np.int64(3),1)]},5)['bytes'],3)
    def test_duplicate_names_and_bad_lowrank(self):
        with self.assertRaises(ValueError):bu.allocate_options({'g':[bu.Option('a',1,1),bu.Option('a',2,0)]},5)
        for dims in ((True,8,1),(8.5,8,1)):
            with self.assertRaises(ValueError):bu.lowrank_cost(*dims)
    def test_qera_conventions(self):
        raw=bu.lowrank_cost(248320,5120,64);row=bu.lowrank_cost(248320,5120,64,scale_bytes=(248320+64)*2,metadata_bytes=128)
        self.assertEqual(raw['raw_bytes'],16220160);self.assertEqual(row['total_bytes'],16717056)
    def test_dp_bruteforce_twenty_cases(self):
        rng=np.random.default_rng(202604)
        for _ in range(20):
            groups={str(g):[bu.Option(str(k),int(rng.integers(10)),float(rng.normal())) for k in range(3)] for g in range(4)}
            cap=20;feasible=[(sum(o.loss for o in p),sum(o.bytes for o in p)) for p in itertools.product(*groups.values()) if sum(o.bytes for o in p)<=cap]
            if feasible:
                loss,cost=min(feasible);result=bu.allocate_options(groups,cap);self.assertAlmostEqual(result['additive_loss'],loss);self.assertEqual(result['bytes'],cost)
            else:
                with self.assertRaises(ValueError):bu.allocate_options(groups,cap)

class CaptureAndManifestRegressions(unittest.TestCase):
    def fixture(self,td,kind='input_pair'):
        root=Path(td);arrays={};splitids={};mapping={}
        for split in ('cal','select','confirm'):
            splitids[split]=[split+'docA',split+'docB'];mapping[split]=[split+'docA']*2+[split+'docB']*2
            arrays['X_fp_'+split]=np.ones((8,4));arrays['X_q_'+split]=np.ones((8,4))
            if kind!='input_pair':arrays['Y_fp_'+split]=np.ones((3,4))
        if kind=='linear':arrays['W_fp']=np.ones((3,8))
        if kind=='swiglu':arrays.update(W_gate_fp=np.ones((2,8)),W_up_fp=np.ones((2,8)),W_down_fp=np.ones((3,2)))
        npz=root/'data.npz';np.savez(npz,**arrays)
        meta={'schema_version':2,'capture_kind':kind,'checkpoint':'fixture','revision':'fixture','runtime_commit':'a'*40,'activation_path':'linear.input','full_input_width':8,'splits':splitids,'sample_source_ids':mapping,'array_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'source_kind':'synthetic_fixture','architecture_id':'test','weight_status':'float64 fixture'}
        path=root/'manifest.json';path.write_text(json.dumps(meta),encoding='utf-8');return npz,path,meta,arrays
    def test_valid_source_ids_not_token_count_and_equal_inputs_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            a,b,m,_=self.fixture(td);r=vc.validate(a,b,allow_synthetic=True)
            self.assertTrue(r['schema_valid']);self.assertFalse(r['provenance_verified']);self.assertTrue(r['arrays']['cal']['inputs_identical'])
    def test_synthetic_needs_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            a,b,_,_=self.fixture(td)
            with self.assertRaises(ValueError):vc.validate(a,b)
    def test_linear_and_swiglu_targets(self):
        for kind in ('linear','swiglu'):
            with tempfile.TemporaryDirectory() as td:
                a,b,_,_=self.fixture(td,kind);self.assertTrue(vc.validate(a,b,allow_synthetic=True)['schema_valid'])
    def test_placeholder_and_noninteger_width(self):
        for field,value in (('runtime_commit','A_REMPLIR'),('full_input_width',8.5),('full_input_width',True)):
            with tempfile.TemporaryDirectory() as td:
                a,b,m,_=self.fixture(td);m[field]=value;b.write_text(json.dumps(m))
                with self.assertRaises(ValueError):vc.validate(a,b,allow_synthetic=True)
    def test_empty_or_wrong_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            a,b,m,_=self.fixture(td);m['sample_source_ids']['cal']=[];b.write_text(json.dumps(m))
            with self.assertRaises(ValueError):vc.validate(a,b,allow_synthetic=True)
    def test_empty_arrays_and_integer_arrays(self):
        for x in (np.zeros((8,0)),np.ones((8,4),dtype=np.int64)):
            with tempfile.TemporaryDirectory() as td:
                a,b,m,ar=self.fixture(td);ar['X_fp_cal']=x;np.savez(a,**ar);m['array_sha256']=hashlib.sha256(a.read_bytes()).hexdigest();b.write_text(json.dumps(m))
                with self.assertRaises(ValueError):vc.validate(a,b,allow_synthetic=True)
    def test_source_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            a,b,m,_=self.fixture(td);m['splits']['confirm']=m['splits']['cal'];b.write_text(json.dumps(m))
            with self.assertRaises(ValueError):vc.validate(a,b,allow_synthetic=True)
    def test_manifest_extra_missing_and_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'a').write_bytes(b'a');entry={'path':'a','bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()}
            mf=root/'MANIFEST_SHA256.json';mf.write_text(json.dumps({'files':[entry]}));self.assertTrue(vm.verify(root)['verified'])
            (root/'extra').write_bytes(b'e');self.assertFalse(vm.verify(root)['verified']);(root/'extra').unlink()
            mf.write_text(json.dumps({'files':[entry,entry]}));self.assertFalse(vm.verify(root)['verified'])
            mf.write_text(json.dumps({'files':[entry]}));(root/'a').unlink();self.assertFalse(vm.verify(root)['verified'])
