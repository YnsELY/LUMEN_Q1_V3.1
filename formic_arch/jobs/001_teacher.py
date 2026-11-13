import importlib, sys, time, json
sys.path.insert(0, "/workspace/LUMEN_Q1_V3.1/formic_arch/lab")
import formic_q, harness
importlib.reload(formic_q); importlib.reload(harness)
H = harness; H.init(model, tok)
t=time.time(); print(H.build_teacher(), time.time()-t)
# parity: language_model path vs full multimodal wrapper forward
x = H.S["data"]["select"][:2].cuda()
with torch.no_grad():
    l1 = model(input_ids=x).logits.float()
    l2 = model.lm_head(H.S["lm"](input_ids=x).last_hidden_state).float()
print("wrapper-vs-text-path max abs logit diff:", (l1-l2).abs().max().item())
del l1, l2
t=time.time(); r = H.evaluate("select"); print("teacher self-eval:", H.short(r), time.time()-t)
json.dump(r, open(H.OUT+"/teacher_select.json","w"))
print({k:(round(v["ce_t"],4)) for k,v in r["by_domain"].items()})
print("mats:", len(H.qmats()), sum(m["mod"].weight.numel() for m in H.qmats())/1e9)
print(torch.cuda.memory_allocated()/2**30, torch.cuda.max_memory_allocated()/2**30)
