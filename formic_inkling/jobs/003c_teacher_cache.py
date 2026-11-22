"""Cache du professeur CORRIGÉ (experts NVFP4 au facteur FP32 exact) sur P-I gelé (sha inchangé) + journal du correctif."""
import sys, json, torch
sys.path.insert(0, '/workspace/LUMEN_Q1_V3.1/formic_inkling/lab')
import importlib, ink_eval as E; importlib.reload(E)
E.init(model, tok); print(E.load_protocol()); print(E.teacher_cache())
import os
tc = json.load(open(E.RES + '/teacher_check.json')) if os.path.exists(E.RES + '/teacher_check.json') else {}
fx = model._formic_fix
tc['transformers_loader_defect'] = dict(
    finding="Transformers 5.24 (sans quantifieur) convertit les sources en BF16 avant Inkling2LoopDequantizeExperts : le facteur FP32 amax/dénominateur par expert NVFP4 est arrondi en BF16",
    evidence=dict(all_40_tensors_equal_bf16_factor_dequant=all(v['transformers_equals_bf16_factor_dequant'] for v in fx.values()),
                  mean_frac_weights_changed_by_fix=sum(v['frac_weights_changed'] for v in fx.values()) / len(fx), per_tensor=fx),
    kl_bad_teacher_vs_exact_fp32_dev=0.018765602260828018, top1_bad_vs_exact=0.9755226373672485,
    fix="ink_model.fix_nvfp4_experts : redéquantification depuis les codes au facteur FP32 exact, arrondi BF16 (RNE)",
    consequence="le cache du professeur, le contrôle FP32/BF16 et toutes les passes sont faits avec le professeur corrigé ; les réponses générées de P-I (agentique, raisonnement, instructions) l'ont été avant correction et sont conservées (protocole gelé)")
json.dump(tc, open(E.RES + '/teacher_check.json', 'w'), indent=1)
