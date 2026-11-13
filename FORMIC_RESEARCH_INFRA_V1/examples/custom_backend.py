"""Working factory contract using the tiny fixture. Replace with your real runtime.
Do not identify this fixture as Qwen. Plugin code is executed with user privileges.
"""
from formic_lab.models import Bundle,ToyLM
import torch

def build(cfg):
    with torch.random.fork_rng():
        torch.manual_seed(cfg.get('seed',17));model=ToyLM().eval()
    return Bundle(model,None,{'backend':'plugin_fixture','synthetic':True,'checkpoint':'formic/toy-fixture',
                              'revision':'example-factory','vocab_size':64})
