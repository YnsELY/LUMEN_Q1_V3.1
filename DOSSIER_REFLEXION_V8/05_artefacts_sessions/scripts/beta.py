# Calibrate, per family, the exponent beta of the column KL-weight  omega_j ~ X_j^beta  from the recorded alpha=1 RTN family test,
# then predict the best exponent alpha* and the corresponding gain.  Family KL model (RTN, diagonal):
#   K(alpha) ~ sum_matrices (G mult^2/n) (sum_j C_j X_j^alpha) (sum_j X_j^(beta-alpha))
import torch, json, math
ce=torch.load('colenergy.pt'); st=torch.load('/workspace/work/stats_fisher.pt', map_location='cpu'); X=st['X']
meas=json.load(open('/workspace/LUMEN_Q1_V3.1/formic_arch/results/diag_bykind_rules.json'))
groups={"mlp_in":("gate_proj","up_proj"),"mlp_down":("down_proj",),"gdn_in":("in_proj_qkv","in_proj_z","in_proj_a","in_proj_b"),
        "gdn_out":("out_proj",),"attn_in":("q_proj","k_proj","v_proj"),"attn_out":("o_proj",),"lm_head":("lm_head",)}
mkey={"mlp_in":"mlp_in(gate,up)","mlp_down":"mlp_down","gdn_in":"gdn_in(qkv,z,a,b)","gdn_out":"gdn_out","attn_in":"attn_in(q,k,v)","attn_out":"attn_out","lm_head":"lm_head"}
ml,mc=0.5917,0.5722
def K(ks,alpha,beta):
    tot=0.0
    for k in ks:
        C=ce[k]['col'].double(); x=X[k].double(); x=x.clamp_min(float(x.mean())*1e-6); lx=x.log()
        lx=lx-lx.mean()                                   # geometric-mean normalisation (scale-free)
        tot+=float((C*(alpha*lx).exp()).sum()*(((beta-alpha)*lx).exp()).sum())/len(x)*float(x.log().mean().exp())**beta
    return tot
print("family     measured  beta_fit | predicted K(alpha)/K(0) at alpha=0.25, 0.5, 0.75, 1 (same mult) | alpha* , K(alpha*)/K(0)")
for g,kinds in groups.items():
    ks=[k for k in X if k.split('.')[-1] in kinds]
    m=meas[mkey[g]+'|col']['kl']/meas[mkey[g]+'|lumen']['kl']/(mc/ml)**2      # ratio at equal multiplier
    f=lambda b: K(ks,1.0,b)/K(ks,0.0,b)-m
    lo,hi=-3.0,3.0
    if f(lo)*f(hi)>0:
        print("%-9s %.3f  no root in [-3,3]: f(-3)=%.3f f(3)=%.3f"%(g,m,f(lo)+m,f(hi)+m)); continue
    for _ in range(60):
        mid=(lo+hi)/2
        if f(lo)*f(mid)<=0: hi=mid
        else: lo=mid
    b=(lo+hi)/2; k0=K(ks,0.0,b)
    grid=[i/40-1 for i in range(0,101)]; vals=[K(ks,a,b)/k0 for a in grid]; ia=min(range(len(grid)),key=lambda i:vals[i])
    print("%-9s %.3f    %+.2f   | %.3f %.3f %.3f %.3f | %+.2f  %.3f"%(g,m,b,K(ks,.25,b)/k0,K(ks,.5,b)/k0,K(ks,.75,b)/k0,K(ks,1.0,b)/k0,grid[ia],vals[ia]))
