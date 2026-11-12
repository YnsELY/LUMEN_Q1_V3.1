import unittest,itertools,importlib.util
from pathlib import Path
import numpy as np
from lumen_ref import e8
from lumen_ref.linalg import *
from lumen_ref.quantization import *
from lumen_ref.calibration import *
from lumen_ref.entropy import ShellModel

class LinearAlgebraTests(unittest.TestCase):
 def test_inverse_regression_v2(self):
  u=np.eye(8);u[0,1]=.5;u[0,2]=.2;u[1,2]=.3
  inv=inverse_unit_upper(u)
  np.testing.assert_allclose(u@inv,np.eye(8),atol=1e-14)
  self.assertAlmostEqual(inv[0,2],-.05)
 def test_inverse_random(self):
  for seed in range(6):
   r=np.random.default_rng(seed);u=np.eye(16)+np.triu(r.normal(scale=.15,size=(16,16)),1)
   np.testing.assert_allclose(inverse_unit_upper(u),np.linalg.inv(u),atol=1e-13)
 def test_ldl_reconstruction(self):
  r=np.random.default_rng(5);a=r.normal(size=(24,24));h=a@a.T+.1*np.eye(24)
  u,d=upper_ldl(h);np.testing.assert_allclose((u*d)@u.T,h,atol=2e-13)
 def test_block_inverse_and_metric(self):
  r=np.random.default_rng(4);a=r.normal(size=(16,16));h=a@a.T+np.eye(16)
  u,blocks=block_ldl(h);_,d=upper_ldl(h)
  for b,(v,m) in enumerate(blocks):
   us=u[8*b:8*b+8,8*b:8*b+8];np.testing.assert_allclose(us@v,np.eye(8),atol=1e-13)
   np.testing.assert_allclose(m,(us*d[8*b:8*b+8])@us.T,atol=1e-13)
 def test_spd_rejects_silent_clipping(self):
  with self.assertRaises(np.linalg.LinAlgError):upper_ldl(np.diag([1.,-1.]))
 def test_fwht_dimensions_and_inverse(self):
  h=hadamard(16);np.testing.assert_allclose(h@h,np.eye(16),atol=1e-14)
  with self.assertRaises(ValueError):hadamard(96)
 def test_silu_no_input_clipping(self):
  self.assertAlmostEqual(float(silu(np.array([100.]))[0]),100.)
  self.assertGreater(float(silu(np.array([-30.]))[0]),-1e-10)
 def test_full_rms_then_slice(self):
  x=np.arange(1,17,dtype=float).reshape(8,2);g=np.ones(3)
  a=rmsnorm_slice(x[:3],g,np.mean(x*x,axis=0))
  np.testing.assert_allclose(a,(x/np.sqrt(np.mean(x*x,axis=0)+1e-6))[:3])
  self.assertGreater(np.max(abs(a-rmsnorm_slice(x[:3],g,np.mean(x[:3]**2,axis=0)))),.1)
 def test_change_of_basis_objective(self):
  r=np.random.default_rng(7);e=r.normal(size=(4,8));x=r.normal(size=(8,20));y=r.normal(size=(4,12))
  h=x@x.T;g=y@y.T;di=np.linspace(.7,1.6,8);do=np.linspace(.6,1.4,4);u=hadamard(8)
  ep=(e/ do[:,None])*di[None,:]@u.T
  hp=u@(h/di[:,None]/di[None,:])@u.T;gp=g*do[:,None]*do[None,:]
  self.assertAlmostEqual(float(np.sum(e*(g@e@h))),float(np.sum(ep*(gp@ep@hp))),places=9)
 def test_pcg_against_dense(self):
  r=np.random.default_rng(8);x=r.normal(size=(16,16));a=x@x.T+.2*np.eye(16);b=r.normal(size=16)
  sol=pcg(lambda v:a@v,b,np.diag(a),rtol=1e-11,maxiter=80)
  self.assertTrue(sol.converged);np.testing.assert_allclose(sol.solution,np.linalg.solve(a,b),rtol=1e-8,atol=1e-8)
 def test_pcg_reports_incomplete(self):
  a=np.diag(np.arange(1,9,dtype=float));b=np.ones(8)
  sol=pcg(lambda v:a@v,b,maxiter=1,rtol=1e-14);self.assertFalse(sol.converged)

class QuantizerTests(unittest.TestCase):
 def test_roots_and_cardinalities(self):
  self.assertEqual(len(e8.ROOTS),240);self.assertEqual(len(set(map(tuple,e8.ROOTS))),240)
  for v in e8.ROOTS:self.assertTrue(e8.valid(v));self.assertEqual(e8.shell(v),1)
  for m in range(64):self.assertEqual(e8.coset_size(m,0)+e8.coset_size(m,1),e8.shell_size(m))
 def test_nearest_independent_small_ball(self):
  pool=np.array([e8.unrank(m,j) for m in range(3) for j in range(e8.shell_size(m))])/2
  r=np.random.default_rng(16)
  for v in r.normal(scale=.25,size=(12,8)):
   q=e8.nearest(v)/2
   self.assertLessEqual(np.sum((q-v)**2),np.min(np.sum((pool-v)**2,axis=1))+1e-12)
 def test_invalid_vector_rejected(self):
  self.assertFalse(e8.valid([2,0,0,0,0,0,0,0]))
  self.assertFalse(e8.valid([1,0,0,0,0,0,0,0]))
 def test_shell_symmetry_actual_codes(self):
  model=ShellModel.gaussian()
  costs=[model.bits(u) for u in e8.ROOTS]
  self.assertLess(max(costs)-min(costs),1e-14)
 def test_escape_no_clamp(self):
  model=ShellModel.gaussian();big=[200,200,0,0,0,0,0,0]
  self.assertGreater(model.bits(big),256.)
 def test_incumbent_in_candidates(self):
  inc=np.array([4,4,0,0,0,0,0,0]);c=e8.candidates(np.zeros(8),1,inc)
  self.assertIn(tuple(inc),set(map(tuple,c)))
 def test_ldlq_matches_dense_flatten_reference(self):
  r=np.random.default_rng(22);w=r.normal(size=(3,16));a=r.normal(size=(16,20));h=a@a.T/20+.2*np.eye(16)
  b=r.normal(size=(3,5));g=b@b.T+.2*np.eye(3);model=ShellModel.gaussian();lam=.03
  q=quantize_ldlq(w,h,.8,lam,model,g)
  big=np.kron(g,h);ui,fac=block_ldl(big,8);wf=w.ravel();qr=np.zeros_like(wf)
  for j in range(0,len(wf),8):
   inv,metric=fac[j//8];center=wf[j:j+8]+((wf[:j]-qr[:j])@ui[:j,j:j+8])@inv
   code,_=select_code(center,q.steps[j//16],metric,lam,model)
   qr[j:j+8]=code*q.steps[j//16]/2
  np.testing.assert_allclose(q.weights.ravel(),qr,atol=1e-12)
 def test_abcd_fixed_table_monotonic(self):
  r=np.random.default_rng(31);w=r.normal(size=(3,16));a=r.normal(size=(16,20));h=a@a.T/20+.1*np.eye(16)
  q=quantize_ldlq(w,h,.85,.02);out,log=refine_quadratic(w,q,h,rate_lambda=.02,sweeps=2)
  self.assertTrue(np.all(np.diff(log['objective_trace_fixed_table'])<=1e-9))
  self.assertGreater(out.encoded_bpw,0)
 def test_zero_rows_export(self):
  q=quantize_ldlq(np.zeros((2,8)),np.eye(8));self.assertTrue(np.array_equal(q.weights,np.zeros((2,8))))

class CalibrationTests(unittest.TestCase):
 def setUp(self):self.r=np.random.default_rng(75)
 def test_asym_normal_equations(self):
  x=self.r.normal(size=(8,40));xq=x+.15*self.r.normal(size=x.shape);w=self.r.normal(size=(3,8));rho=.1
  b,h=asymmetric_target(w,x,xq,rho)
  grad=((b@xq-w@x)@xq.T)/40+rho*(b-w)
  self.assertLess(np.max(abs(grad)),1e-12)
 def test_asym_identity_control(self):
  x=self.r.normal(size=(8,40));w=self.r.normal(size=(3,8));b,h=asymmetric_target(w,x,x,.1)
  np.testing.assert_allclose(b,w,atol=1e-12)
 def test_bgcc_row_equations(self):
  x=self.r.normal(size=(8,40));w=self.r.normal(size=8);g=silu(self.r.normal(size=40));y=self.r.normal(size=40)
  b,h=bgcc_row_target(w,x,g,y,.2);f=x*g[None,:]
  grad=f@(b@f-y)/40+.2*(b-w);self.assertLess(np.max(abs(grad)),1e-12)
 def test_bgcc_operator_symmetry_and_spd(self):
  x=self.r.normal(size=(8,32));g=self.r.normal(size=(4,32));d=self.r.normal(size=(5,4))
  mv,diag=bgcc_output_operator(x,g,d,.1);a=self.r.normal(size=(4,8));b=self.r.normal(size=(4,8))
  self.assertAlmostEqual(float(np.sum(a*mv(b))),float(np.sum(b*mv(a))),places=10)
  self.assertGreater(float(np.sum(a*mv(a))),0);self.assertTrue(np.all(diag>0))
 def test_bgcc_cg_vs_dense_full_hessian(self):
  m,n,N=3,4,50;x=self.r.normal(size=(n,N));g=self.r.normal(size=(m,N));d=self.r.normal(size=(4,m));a=self.r.normal(size=(m,n));y=self.r.normal(size=(4,N));rho=.2
  target,info=bgcc_output_cg(a,x,g,d,y,rho,rtol=1e-12,maxiter=80)
  mv,_=bgcc_output_operator(x,g,d,rho);eye=np.eye(m*n)
  H=np.stack([mv(e.reshape(m,n)).ravel() for e in eye],axis=1)
  residual=y-d@(g*(a@x));rhs=((g*(d.T@residual))@x.T/N).ravel()
  dense=a+np.linalg.solve(H,rhs).reshape(m,n)
  self.assertTrue(info.converged);np.testing.assert_allclose(target,dense,atol=1e-9)
 def test_bgcc_gradient_finite_difference(self):
  x=self.r.normal(size=(4,12));g=self.r.normal(size=(3,12));d=self.r.normal(size=(2,3));a=self.r.normal(size=(3,4));w=a+.1;y=self.r.normal(size=(2,12));v=self.r.normal(size=w.shape);rho=.3
  grad=2*(g*(d.T@(d@(g*(w@x))-y)))@x.T/12+2*rho*(w-a)
  eps=1e-6;fd=(swiglu_objective(w+eps*v,x,g,d,y,a,rho)-swiglu_objective(w-eps*v,x,g,d,y,a,rho))/(2*eps)
  self.assertAlmostEqual(fd,float(np.sum(grad*v)),places=6)
 def test_swiglu_discrete_monotonic(self):
  x=self.r.normal(size=(16,96));g=silu(self.r.normal(size=(4,96)));d=self.r.normal(size=(5,4));w=self.r.normal(scale=.2,size=(4,16));h=x@x.T/96+.01*np.eye(16)
  q=quantize_ldlq(w,h,.9);y=d@(g*(w@x))
  out,log=refine_swiglu_up(q,x,g,d,y,ridge=.01,rate_lambda=.0001,sweeps=2)
  self.assertTrue(np.all(np.diff(log['fixed_table_objective_trace'])<=1e-9))
 def test_fresh_target_completion_square(self):
  w=self.r.normal(size=(3,8));b=self.r.normal(size=w.shape);h=np.diag(np.arange(1,9.));g=np.diag([1.,2.,4.])
  target,info=fresh_quadratic_target(w,b,h,g,1e9)
  np.testing.assert_allclose(g@(target-w)@h,-b,atol=1e-13)
  t,info=fresh_quadratic_target(w,b,h,g,.1);v=t-w
  self.assertLessEqual(np.sqrt(np.sum(v*(g@v@h))),.1+1e-12)

if __name__=='__main__':unittest.main()
