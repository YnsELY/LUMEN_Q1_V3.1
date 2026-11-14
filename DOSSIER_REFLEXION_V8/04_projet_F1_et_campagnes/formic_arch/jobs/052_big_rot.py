SQ.run_ldlq_variant("ArotN_Q0ldlq_big", dict(rule="lumen", match_bpw=2.8158), H.S["mult_Q0ldlq"],
                    method="standard E8-LDLQ on the model natively rewritten in a rotated residual basis (re-run of ArotN_Q0ldlq)", prepare=lambda: SQ.rotate_model(0))
