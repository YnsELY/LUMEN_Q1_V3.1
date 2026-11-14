import importlib, sys, time, json, math
import formic_q, harness
# reload code but keep state (teacher cache etc.)
_S = harness.S; importlib.reload(formic_q); importlib.reload(harness); harness.S.update(_S); H = harness
H.S["_bias_free"] = [mm for mm in list(H.S["lm"].modules()) + [model.lm_head] if isinstance(mm, torch.nn.Linear) and mm.bias is None]
print("bias-free linears:", len(H.S["_bias_free"]))
st = H.S["st"]; H.restore()
t = time.time(); st["MU"] = H.collect_means(); print("means %.0fs" % (time.time()-t))
# how much of the input second moment is the squared mean (input-independent component)?
for n in ["layers.1.mlp.gate_proj", "layers.30.mlp.gate_proj", "layers.30.mlp.down_proj", "layers.30.linear_attn.in_proj_qkv", "layers.30.linear_attn.out_proj", "layers.31.self_attn.q_proj", "layers.31.self_attn.o_proj", "layers.62.mlp.gate_proj", "lm_head"]:
    mu = st["MU"][n]; X = st["X"][n]
    print("%-36s |mu|^2/tr(H)=%.3f  top-8 dims share of tr(H)=%.3f  of which mean part=%.3f" % (n, float(mu.pow(2).sum()/X.sum()), float(X.topk(8).values.sum()/X.sum()), float(mu.pow(2)[X.topk(8).indices].sum()/X.topk(8).values.sum())))
share = {k: float(st["MU"][k].pow(2).sum()/st["X"][k].sum()) for k in st["MU"] if k in st["X"]}
json.dump(share, open(H.OUT+"/diag_mean_share.json", "w"))
mats = H.qmats(); target = H.S["target_bits"]; m0 = H.S["mult_Q0rtn"]
def run(vid, spec):
    t = time.time(); H.restore()
    mult = H.calibrate_mult(mats, spec, target, st, lo=0.25, hi=2.0, iters=11)
    info = H.quantize_all(mats, spec, mult, st)
    info.update(spec=spec, mult=mult, target_bits=target, bits_over_target=info["bits"]/target-1, bytes=info["bits"]/8)
    print(vid, "mult=%.4f bpw=%.4f bits/target-1=%+.4f%%" % (mult, info["bpw"], 100*info["bits_over_target"]), {k: round(v, 3) for k, v in info["bykind"].items()}, flush=True)
    return H.run_variant(vid, info)
# family-level allocation derived from the measured single-family KLs (diag_bykind_rules.json, select subset)
bk = json.load(open(H.OUT+"/diag_bykind_rules.json"))
fam = {"mlp_in(gate,up)": ("gate_proj", "up_proj"), "mlp_down": ("down_proj",), "gdn_in(qkv,z,a,b)": ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
       "gdn_out": ("out_proj",), "attn_in(q,k,v)": ("q_proj", "k_proj", "v_proj"), "attn_out": ("o_proj",), "embed": ("embed",), "lm_head": ("lm_head",)}
dens = {g: bk[g+"|lumen"]["kl"]/bk[g+"|lumen"]["params"] for g in fam}
ntot = sum(bk[g+"|lumen"]["params"] for g in fam)
gm = sum(bk[g+"|lumen"]["params"]*math.log2(dens[g]) for g in fam)/ntot
dbits = {}
for g, kinds in fam.items():
    for k in kinds: dbits[k] = round(0.5*(math.log2(dens[g]) - gm), 3)
print("family bit offsets:", dbits); H.S["dbits"] = dbits
run("Ab_Q0rtn", dict(rule="lumen", bias_corr=True))
run("A0_Qalloc_rtn", dict(rule="lumen", kind_dbits=dbits))
run("Ab_Qalloc_rtn", dict(rule="lumen", kind_dbits=dbits, bias_corr=True))
H.restore()
