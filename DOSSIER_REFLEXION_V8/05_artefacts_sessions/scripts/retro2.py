# Same retrodiction, with the proxy weight w_j = Var_j + lam * Mean_j^2  (lam=1: raw second moment used by LDLQ/F1; lam=0: centred variance)
import torch, json, math
ce=torch.load('colenergy.pt'); st=torch.load('/workspace/work/stats_fisher.pt', map_location='cpu'); X=st['X']; MU=st['MU']
meas=json.load(open('/workspace/LUMEN_Q1_V3.1/formic_arch/results/diag_bykind_rules.json'))
groups={"mlp_in":("gate_proj","up_proj"),"mlp_down":("down_proj",),"gdn_in":("in_proj_qkv","in_proj_z","in_proj_a","in_proj_b"),
        "gdn_out":("out_proj",),"attn_in":("q_proj","k_proj","v_proj"),"attn_out":("o_proj",),"lm_head":("lm_head",)}
mkey={"mlp_in":"mlp_in(gate,up)","mlp_down":"mlp_down","gdn_in":"gdn_in(qkv,z,a,b)","gdn_out":"gdn_out","attn_in":"attn_in(q,k,v)","attn_out":"attn_out","lm_head":"lm_head"}
ml,mc=0.5917,0.5722
print("mean^2 share of sum X per family (sum-weighted):")
for g,kinds in groups.items():
    ks=[k for k in X if k.split('.')[-1] in kinds]
    sh=sum(float((MU[k].double()**2).sum()) for k in ks)/sum(float(X[k].double().sum()) for k in ks)
    print("  %-9s %.3f"%(g,sh))
for lam in (1.0,0.5,0.25,0.1,0.0):
    line=[]; sse=0
    for g,kinds in groups.items():
        num=0.0; den=0.0
        for k in [k for k in X if k.split('.')[-1] in kinds]:
            C=ce[k]['col'].double(); x=X[k].double(); mu2=MU[k].double()**2
            v=(x-mu2).clamp_min(0); w=v+lam*mu2
            xs=x.clamp_min(float(x.mean())*1e-6)                # scale actually used by the 'col' rule: c^2 = X/GM
            nat=float(C.sum()*w.sum()); a1=float((C*xs).sum()*(w/xs).sum())
            num+=(mc/ml)**2*a1; den+=nat
        m=meas[mkey[g]+'|col']['kl']/meas[mkey[g]+'|lumen']['kl']; p=num/den
        sse+=(math.log(p)-math.log(m))**2; line.append("%s %.2f/%.2f"%(g,p,m))
    print("lam=%.2f  model/measured: "%lam, " | ".join(line), " | rms log-error %.3f"%math.sqrt(sse/7))
