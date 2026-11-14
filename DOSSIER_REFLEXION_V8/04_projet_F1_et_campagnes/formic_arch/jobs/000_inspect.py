import time
lm = model.model.language_model
print(type(model).__name__, type(model.model).__name__, type(lm).__name__)
print([n for n,_ in model.named_children()], [n for n,_ in model.model.named_children()])
print(lm.layers[0]); print(lm.layers[3])
print(sum(p.numel() for p in model.parameters()))
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model(input_ids=ids)
print(out.logits.shape, tok.decode(out.logits[0,-1].argmax()))
x = torch.randint(0, 200000, (8, 1024)).cuda()
for bs in (8,):
    torch.cuda.synchronize(); t=time.time()
    with torch.no_grad(): o = model(input_ids=x[:bs]).logits
    torch.cuda.synchronize(); print("bs", bs, "T 1024", time.time()-t, o.dtype, torch.cuda.max_memory_allocated()/2**30)
    t=time.time()
    with torch.no_grad(): o = model(input_ids=x[:bs]).logits
    torch.cuda.synchronize(); print("bs", bs, "T 1024 2nd", time.time()-t)
del o
import inspect, transformers.models.qwen3_5.modeling_qwen3_5 as M
print(M.__file__)
