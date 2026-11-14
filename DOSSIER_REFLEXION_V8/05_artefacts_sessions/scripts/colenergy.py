# Static, CPU-only: column / row energies (sum of squares) of the text-path linears of the BF16 checkpoint.
# No forward, no quantization. Output: scratchpad/colenergy.pt
import json, torch, time, sys
from safetensors import safe_open
torch.set_num_threads(32)
root='/root/models/qwen3.8-27b'
idx=json.load(open(root+'/model.safetensors.index.json'))['weight_map']
out={}; t=time.time()
names=[k for k in idx if k.endswith('.weight') and (('mlp.' in k) or ('linear_attn.in_proj' in k) or ('linear_attn.out_proj' in k) or ('self_attn.' in k and '_proj' in k)) and k.startswith('model.language_model.layers.')]
names+= ['lm_head.weight','model.language_model.embed_tokens.weight']
byshard={}
for k in names: byshard.setdefault(idx[k],[]).append(k)
for shard,ks in sorted(byshard.items()):
    with safe_open(root+'/'+shard,'pt',device='cpu') as f:
        for k in ks:
            W=f.get_tensor(k)
            col=torch.zeros(W.shape[1],dtype=torch.float64); row=torch.zeros(W.shape[0],dtype=torch.float64)
            for i in range(0,W.shape[0],8192):
                w=W[i:i+8192].float(); w2=w*w
                col+=w2.sum(0).double(); row[i:i+8192]=w2.sum(1).double()
            key=k.replace('model.language_model.','').replace('.weight','')
            out[key]=dict(col=col.float(), row=row.float(), shape=tuple(W.shape))
            del W
    print(shard, len(ks), round(time.time()-t,1), flush=True)
torch.save(out,'/tmp/claude-0/-workspace/78f73527-ae7c-4365-8749-6095616e0753/scratchpad/colenergy.pt')
print('done', len(out), round(time.time()-t,1))
