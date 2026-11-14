"""Full-package byte budget from the real checkpoint headers (no weights read).
Quantizable = every 2-D+ weight tensor with >= 4096 elements per row-block that the E8 codec can take
(all linears, embeddings, conv patch embed flattened); protected = norms, biases, tiny tensors (kept FP16)."""
import json, sys
hdr = json.load(open("/workspace/LUMEN_Q1_V3.1/formic_arch/results/checkpoint_headers.json"))
T = hdr["tensors"]; items = list(T.values()) if isinstance(T, dict) else T
CEIL = 10_000_000_000
AUX = 84_066_560          # Lumen v3.1 "Max" auxiliary assumption (inherited, not measured)
GUARD = 18_500_000        # Lumen v3.1 technical guard
CODEC_OVERHEAD_BPW = 0.009  # measured: physical - ideal bits of the reference codec on sub-matrices
groups = {"text_body": 0, "embed": 0, "lm_head": 0, "vision": 0, "mtp": 0}; prot = 0; nq = 0; nrows = 0
for t in items:
    n = 1
    for d in t["shape"]: n *= d
    name = t["name"]
    quant = len(t["shape"]) >= 2 and n >= 65536 and "conv1d" not in name
    if not quant:
        prot += n; continue
    nq += n; nrows += t["shape"][0]
    if name.startswith("mtp."): groups["mtp"] += n
    elif "visual" in name: groups["vision"] += n
    elif "embed_tokens" in name: groups["embed"] += n
    elif name.startswith("lm_head"): groups["lm_head"] += n
    else: groups["text_body"] += n
payload = CEIL - 2 * prot - AUX - GUARD
bpw_phys = 8 * payload / nq
out = dict(ceiling_bytes=CEIL, protected_params=prot, protected_bytes_fp16=2 * prot, quantizable_params=nq, groups=groups,
           aux_bytes_assumed=AUX, guard_bytes=GUARD, physical_bpw_capacity=bpw_phys,
           ideal_symbol_bpw_capacity=bpw_phys - CODEC_OVERHEAD_BPW, lumen_v31_declared_capacity=2.832825,
           target_used_in_experiments_ideal_bpw=2.824,
           text_params_evaluated=groups["text_body"] + groups["embed"] + groups["lm_head"],
           not_evaluated_params=groups["vision"] + groups["mtp"])
json.dump(out, open("/workspace/LUMEN_Q1_V3.1/formic_arch/results/budget_package.json", "w"), indent=1)
print(json.dumps(out, indent=1))
