import unittest,itertools,tempfile,hashlib
from pathlib import Path
import numpy as np
from lumen_ref import e8
from lumen_ref.entropy import *
from lumen_ref.budget import *
from lumen_ref.metrics import *

class CodecTests(unittest.TestCase):
 def test_shell_rank_exhaustive_low(self):
  for m in range(3):
   seen=set()
   for r in range(e8.shell_size(m)):
    u=e8.unrank(m,r);self.assertEqual(e8.rank(u)[2],r);seen.add(u)
   self.assertEqual(len(seen),e8.shell_size(m))
 def test_coset_rank_random_shells(self):
  rng=np.random.default_rng(5)
  for m in (1,2,3,8,16,63):
   for p in (0,1):
    count=e8.coset_size(m,p)
    for _ in range(8):
     r=int(rng.integers(count));u=e8.unrank(m,r,p);self.assertEqual(e8.rank(u,True),(m,p,r))
 def test_rans_roundtrip_many_alphabets(self):
  rng=np.random.default_rng(9)
  for K in (1,2,8,65,129):
   for N in (0,1,17,1024):
    s=rng.integers(K,size=N).tolist();counts=np.bincount(s,minlength=K).tolist();f=normalized_frequencies(counts)
    self.assertEqual(rans_decode(rans_encode(s,f),f,N),s)
 def test_rans_truncated_rejected(self):
  f=normalized_frequencies([1,2]);b=rans_encode([0,1]*100,f)
  with self.assertRaises(ValueError):rans_decode(b[:-1],f,200)
 def test_codec_roundtrip_modes_and_escape(self):
  rng=np.random.default_rng(8);codes=[e8.nearest(v) for v in rng.normal(size=(500,8))]
  codes+=[np.zeros(8,dtype=int),np.array([200,200,0,0,0,0,0,0])]
  for mode in (0,1):
   for group in (1,32,128):
    model=ShellModel.fit(codes,mode=mode);blob=encode_e8(codes,model,group);u,_=decode_e8(blob)
    self.assertTrue(np.array_equal(u,np.asarray(codes)))
 def test_empty_codec(self):
  b=encode_e8([]);u,_=decode_e8(b);self.assertEqual(u.shape,(0,8))
 def test_corruption_rejected(self):
  b=bytearray(encode_e8(e8.ROOTS));b[-40]^=1
  with self.assertRaises(ValueError):decode_e8(bytes(b))
 def test_decode_resource_limit(self):
  b=encode_e8(e8.ROOTS)
  with self.assertRaises(ValueError):decode_e8(b,max_vectors=2)
 def test_matrix_scales_exact_after_fp16(self):
  codes=e8.ROOTS[:16].reshape(4,4,8);steps=np.array([.123,.4,.55,1.],dtype=np.float16)
  blob=encode_matrix(codes,steps);q,u,s,model=decode_matrix(blob)
  np.testing.assert_array_equal(u,codes);np.testing.assert_array_equal(q,(codes.reshape(4,32)/2)*steps.astype(float)[:,None])
 def test_model_rate_not_physical_rate(self):
  u=e8.ROOTS[:8];model=ShellModel.fit(u);blob=encode_e8(u,model)
  self.assertNotAlmostEqual(model.total_model_bits(u),len(blob)*8)

class BudgetMetricsTests(unittest.TestCase):
 def test_qera_factor8(self):
  out=lowrank_cost(248320,5120,64)
  self.assertEqual(out['raw_bytes'],16220160)
  self.assertAlmostEqual(out['equivalent_added_bpw'],.10206185567010309)
 def test_waterfill_kkt(self):
  n=np.array([100.,200.,500.]);omega=np.array([1.,2.,5.]);r=continuous_waterfill(n,omega,2200,0,8)
  self.assertLessEqual(n@r,2200+1e-8);np.testing.assert_allclose(omega*4**(-r),np.repeat(omega[0]*4**(-r[0]),3),rtol=1e-9)
 def test_waterfill_floor_infeasible(self):
  with self.assertRaises(ValueError):continuous_waterfill([100],[1],100,2,4)
 def test_discrete_allocation_vs_bruteforce(self):
  groups={'a':[Option('a0',3,4),Option('a1',7,1)],'b':[Option('b0',4,5),Option('b1',8,1)],'c':[Option('c0',2,3),Option('c1',5,.5)]}
  ans=allocate_options(groups,18,2)
  choices=[p for p in itertools.product(*groups.values()) if 2+sum(o.bytes for o in p)<=18]
  best=min(sum(o.loss for o in p) for p in choices);self.assertEqual(ans['additive_loss'],best)
 def test_physical_files_count_and_duplicate(self):
  with tempfile.TemporaryDirectory() as s:
   root=Path(s);(root/'a').write_bytes(b'abc');(root/'b').write_bytes(b'abc')
   self.assertEqual(account_files(root,['a','b'],6)['actual_bytes'],6)
   with self.assertRaises(ValueError):account_files(root,['a','a'],20)
 def test_fisher_exact_enumeration(self):
  p=np.array([.05,.15,.3,.5]);z=np.array(list(itertools.product((-1.,1.),repeat=4)))
  u=fisher_probe(p,z);np.testing.assert_allclose(u.T@u/16,np.diag(p)-np.outer(p,p),atol=1e-15)
 def test_split_overlap(self):
  with self.assertRaises(ValueError):assert_disjoint_splits(['d1'],['d2'],['d1'])
 def test_kl_grouped_lower_bound(self):
  rng=np.random.default_rng(23);a=rng.normal(size=(10,32));b=a+.3*rng.normal(size=a.shape);ids=np.argsort(a,axis=1)[:,-8:]
  self.assertTrue(np.all(exact_grouped_kl(a,b,ids)<=kl_logits(a,b)+1e-12))
 def test_topk_cache_rounding_measured(self):
  r=np.random.default_rng(42);a=r.normal(size=(12,24));b=a+.2*r.normal(size=a.shape);cache=topk_cache(a,6)
  err=np.max(abs(grouped_kl_from_cache(cache,b)-exact_grouped_kl(a,b,cache['ids'])))
  self.assertLess(err,1e-3)
  self.assertEqual(cache['ids'].dtype,np.int32)
 def test_guard_continuous_and_discrete_steps(self):
  prev=ArtifactScore(b'abc',.1,{'text':.1,'vision':.2})
  for candidate in [ArtifactScore(b'abcd',.09,{'text':.1,'vision':.2}),ArtifactScore(b'ab',.11,{'text':.1,'vision':.2}),ArtifactScore(b'ab',.09,{'text':.1,'vision':.3})]:
   kept,log=accept_decoded(prev,candidate,3);self.assertIs(kept,prev);self.assertFalse(log['accepted'])
  cand=ArtifactScore(b'ab',.09,{'text':.1,'vision':.2});kept,log=accept_decoded(prev,cand,3);self.assertIs(kept,cand)
 def test_guard_missing_sentinel(self):
  prev=ArtifactScore(b'a',1,{'vision':1});cand=ArtifactScore(b'a',.9,{})
  _,log=accept_decoded(prev,cand,3);self.assertFalse(log['accepted'])

if __name__=='__main__':unittest.main()
