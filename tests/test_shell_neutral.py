import unittest
import numpy as np
from lumen_ref import e8
from lumen_ref.entropy import ShellModel,encode_e8,decode_e8
from lumen_ref.quantization import quantize_ldlq,refine_quadratic,select_code
from lumen_ref.calibration import refine_swiglu_up
from lumen_ref.linalg import silu

class TestShellNeutral(unittest.TestCase):
 def test_equal_length_arbitrary_rank_changes_both_modes(self):
  rng=np.random.default_rng(303)
  codes=np.array([e8.nearest(v) for v in rng.normal(size=(513,8))])
  for mode in (0,1):
   model=ShellModel.fit(codes,mode=mode);other=[]
   for u in codes:
    m=e8.shell(u);p=int(u[0])&1
    count=e8.coset_size(m,p) if mode else e8.shell_size(m)
    other.append(e8.unrank(m,int(rng.integers(count)),p if mode else None))
   a=encode_e8(codes,model);b=encode_e8(other,model)
   self.assertEqual(len(a),len(b));self.assertTrue(np.array_equal(other,decode_e8(b)[0]))
 def test_quadratic_refinement_same_bytes_and_objective(self):
  rng=np.random.default_rng(305);w=rng.normal(size=(4,16));a=rng.normal(size=(16,16));h=a@a.T+.1*np.eye(16)
  for mode in (0,1):
   model=ShellModel.fit(np.array([e8.nearest(v) for v in rng.normal(size=(48,8))]),mode=mode)
   init=quantize_ldlq(w,h,.8,0.,model=model)
   q,log=refine_quadratic(w,init,h,rate_lambda=0.,preserve_shell=True)
   self.assertEqual(len(q.artifact),len(init.artifact));self.assertEqual(q.model,init.model)
   self.assertTrue(np.all(np.diff(log['objective_trace_fixed_table'])<=1e-8))
   for x,y in zip(q.codes.reshape(-1,8),init.codes.reshape(-1,8)):
    self.assertEqual(e8.shell(x),e8.shell(y))
    if mode:self.assertEqual(int(x[0])&1,int(y[0])&1)
 def test_full_output_refinement_preserves_bytes(self):
  rng=np.random.default_rng(307);x=rng.normal(size=(16,64));w=rng.normal(size=(4,16));g=silu(rng.normal(size=(4,64)));d=rng.normal(size=(5,4));target=d@(g*(w@x))
  init=quantize_ldlq(w,x@x.T/64+.01*np.eye(16),.8,0.)
  out,log=refine_swiglu_up(init,x,g,d,target,preserve_shell=True)
  self.assertEqual(len(out.artifact),len(init.artifact));self.assertEqual(out.model,init.model)
  self.assertTrue(np.all(np.diff(log['fixed_table_objective_trace'])<=1e-8))
 def test_incumbent_required(self):
  with self.assertRaises(ValueError):
   select_code(np.ones(8),1.,np.eye(8),0.,ShellModel.gaussian(),preserve_shell=True)
