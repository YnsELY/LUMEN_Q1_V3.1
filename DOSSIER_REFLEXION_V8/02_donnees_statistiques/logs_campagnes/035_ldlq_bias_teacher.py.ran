# A_b2 x Q0: E8-LDLQ with centred Gram; FP16 bias anchors each linear's mean output to the BF16 teacher's
# mean output (b = W mu_teacher - Q mu_student): removes the accumulated drift of the mean activation.  Resumable.
import importlib, seq
importlib.reload(seq); SQ = seq
SQ.run_ldlq_variant("Ab2_Q0ldlq", dict(rule="lumen", bias_corr="teacher"), H.S["mult_Abldlq"], method="E8-LDLQ sequential, centred Gram + FP16 bias anchored on teacher mean output")
