# Retrodiction of the recorded per-family RTN tests (diag_bykind_rules.json) with the F1 noise model.
# RTN proxy error of a matrix with column scales c:  (G mult^2 / n) * (sum_j C_j c_j^2) * (sum_j X_j / c_j^2)
import torch, json, math, collections
ce=torch.load('colenergy.pt'); st=torch.load('/workspace/work/stats_fisher.pt', map_location='cpu'); X=st['X']
meas=json.load(open('/workspace/LUMEN_Q1_V3.1/formic_arch/results/diag_bykind_rules.json'))
groups={"mlp_in(gate,up)":("gate_proj","up_proj"),"mlp_down":("down_proj",),"gdn_in(qkv,z,a,b)":("in_proj_qkv","in_proj_z","in_proj_a","in_proj_b"),
        "gdn_out":("out_proj",),"attn_in(q,k,v)":("q_proj","k_proj","v_proj"),"attn_out":("o_proj",),"lm_head":("lm_head",)}
ml,mc=0.5917,0.5722
def fam(kinds):
    return [k for k in X if k.split('.')[-1] in kinds]
print("family | measured KL ratio col/lumen | model proxy ratio (alpha=1 incl. mult) : sum-weighted, median | half-power gain gamma (theta=1) : sum-weighted, median")
for g,kinds in groups.items():
    num=0.0; den=0.0; rs=[]; gs=[]; gnum=0.0
    for k in fam(kinds):
        C=ce[k]['col'].double(); x=X[k].double().clamp_min(float(X[k].double().mean())*1e-6); n=len(x)
        nat=float(C.sum()*x.sum())                      # native proxy (up to G mult^2/n)
        a1=float((C*x).sum())*n                         # alpha=1 : (sum C x / GM)(sum x / (x/GM)) = n * sum C x
        r=(mc/ml)**2*a1/nat
        gam=float((C*x).sqrt().sum()**2)/nat            # Cauchy-Schwarz optimum (half-power gauge)
        num+=(mc/ml)**2*a1; den+=nat; gnum+=gam*nat; rs.append(r); gs.append(gam)
    rs.sort(); gs.sort()
    m=meas[g+'|col']['kl']/meas[g+'|lumen']['kl']
    print(f"{g:20s} | {m:5.3f} | {num/den:6.3f}  {rs[len(rs)//2]:6.3f} | {gnum/den:5.3f}  {gs[len(gs)//2]:5.3f}")
