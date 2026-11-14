# A0 x Q0 with the real Lumen algebra: E8 + block-8 LDLQ (sequential, student inputs), reference row-RMS steps.
# Resumable: checkpoints every 4 layers in /workspace/work/ckpt/A0_Q0ldlq (re-submitting this job resumes).
import importlib, seq
importlib.reload(seq); SQ = seq
SQ.run_ldlq_variant("A0_Q0ldlq", dict(rule="lumen"), H.S["mult_Q0ldlq"], method="E8-LDLQ sequential")
