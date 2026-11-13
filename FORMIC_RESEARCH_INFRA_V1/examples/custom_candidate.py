"""Executable example plugin, not an architecture recommendation.
Spec: {kind:'plugin', target:'...', factory:'/.../custom_candidate.py:build', gain:1.0}
"""
import copy
from torch import nn
class ResidualScale(nn.Module):
    def __init__(self,base,gain):super().__init__();self.base=base;self.gain=float(gain)
    def forward(self,x):return self.gain*self.base(x)
def build(original,spec):
    return ResidualScale(copy.deepcopy(original),spec.get('gain',1.0))
