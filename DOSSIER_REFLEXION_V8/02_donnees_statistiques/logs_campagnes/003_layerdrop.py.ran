import time, json
docs = list(range(0, 72, 3))   # 24 docs, 4 per domain
t=time.time(); res={}
base = H.evaluate("select", docs=docs)
for i in range(64):
    with H.skip_layers([i]):
        r = H.evaluate("select", docs=docs)
    res[i] = dict(kl=r["kl"], agree=r["agree"], ce=r["ce_s"], by_domain={k:v["kl"] for k,v in r["by_domain"].items()})
    print(i, H.S["lm"].layers[i].block_type[:4], "KL=%.4f agree=%.3f dCE=%.4f" % (r["kl"], r["agree"], r["ce_s"]-base["ce_t"]), flush=True)
json.dump(res, open(H.OUT+"/diag_layerdrop_single.json","w"))
print("time", time.time()-t)
