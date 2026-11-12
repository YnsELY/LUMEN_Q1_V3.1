import unittest
import numpy as np
from lumen_ref.linalg import inverse_unit_upper
from lumen_ref.budget import allocate_options,Option,continuous_waterfill
from lumen_ref.metrics import ArtifactScore,accept_decoded
from lumen_ref.quantization import select_code
from lumen_ref.entropy import ShellModel
class TestGuards(unittest.TestCase):
 def test_nonunit_diagonal_is_not_silently_ignored(self):
  with self.assertRaises(ValueError):inverse_unit_upper(np.eye(8)*(1+1e-6))
 def test_fractional_bytes_rejected(self):
  with self.assertRaises(ValueError):allocate_options({'x':[Option('a',1.5,0.)]},10)
 def test_nan_budget_rejected(self):
  with self.assertRaises(ValueError):continuous_waterfill([8],[1],float('nan'))
 def test_nan_penalty_rejected(self):
  with self.assertRaises(ValueError):select_code(np.ones(8),1,np.eye(8),float('nan'),ShellModel.gaussian())
 def test_nan_sentinel_tolerance_rejected(self):
  a=ArtifactScore(b'a',1.,{'vision':1.});b=ArtifactScore(b'b',.9,{'vision':2.})
  with self.assertRaises(ValueError):accept_decoded(a,b,10,tolerances={'vision':float('nan')})
