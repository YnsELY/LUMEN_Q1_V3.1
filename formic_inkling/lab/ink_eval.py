"""FORMIC-Ink — protocole P-I, cache du professeur, évaluation (dérivé de formic_v7/lab/v7.py arki_teacher/arki_eval et
formic_arch/lab/harness.py evaluate). Exécuté dans le serveur de laboratoire (S = état partagé).

KL(professeur ‖ candidat) sur les 200 058 entrées utiles (l'unembed est chargé non rembourré), en nats, moyenne par token,
logits = 30·tanh(z/30) pour les deux, log-softmax par morceaux de 1024 positions. Le professeur est rejoué depuis ses
états finaux mis en cache aux positions scorées (2048–4095) avec SA tête d'origine (copie BF16 gelée).
Accord de routage : pour chaque couche et chaque application (48 par token), part des 8 experts communs élève/professeur et
part des tokens dont l'expert top-1 change, aux positions scorées."""
import os, json, time, math, random
import numpy as np, torch, torch.nn.functional as F

S = globals().get("S") or {}          # persistant à travers importlib.reload (copie du professeur)
D = "/workspace/work_ink/data"; SEALED = "/workspace/work_ink/sealed"
RES = "/workspace/LUMEN_Q1_V3.1/formic_inkling/results"
os.makedirs(SEALED, exist_ok=True)
CAP = 30.0
P0, P1 = 2048, 4096


def init(model, tok):
    S["model"] = model; S["tok"] = tok; S["lm"] = model.model.language_model
    if "T_head" not in S:
        S["T_head"] = model.lm_head.weight.detach().clone()          # tête du professeur, gelée
    return S


def softcap(z): return torch.tanh(z / CAP) * CAP


# ------------------------------------------------------------------------------------------------ routage
class Router:
    """Crochets sur les 24 routeurs : enregistre les indices top-8 triés par score (top-1 en tête) à chaque application."""
    def __init__(self, lm, positions=None):
        self.h = []; self.rec = {}; self.pos = positions
        for i, L in enumerate(lm.layers):
            self.h.append(L.mlp.gate.register_forward_hook(self._mk(i)))
    def _mk(self, i):
        def hook(mod, inp, out):
            logits = out[0]
            sc = mod.score_fn(logits) + mod.e_score_correction_bias
            idx = torch.topk(sc, mod.top_k, dim=-1, sorted=True).indices            # même ensemble que le routeur
            if self.pos is not None:
                idx = idx.view(-1, P1, mod.top_k)[:, self.pos].reshape(-1, mod.top_k)
            self.rec.setdefault(i, []).append(idx.to(torch.uint8).cpu())
        return hook
    def take(self):
        r = self.rec; self.rec = {}; return r
    def remove(self):
        for h in self.h: h.remove()


def routing_agreement(rs, rt):
    """rs, rt : {couche: [tenseurs [n,8] uint8 par application]} -> dict par couche/application."""
    out = {}
    for i in rt:
        for a, (s, t) in enumerate(zip(rs[i], rt[i])):
            s = s.long(); t = t.long()
            ms = torch.zeros(s.shape[0], 128, dtype=torch.bool); ms.scatter_(1, s, True)
            mt = torch.zeros(t.shape[0], 128, dtype=torch.bool); mt.scatter_(1, t, True)
            ov = (ms & mt).sum(1).float() / 8
            out[f"{i}.{a}"] = (float(ov.sum()), float((s[:, 0] != t[:, 0]).float().sum()), int(s.shape[0]))
    return out


# ------------------------------------------------------------------------------------------------ protocole
def load_protocol():
    d = torch.load(f"{D}/pi_tokens.pt")
    S["pi"] = dict(tok=d["tokens"], dom=d["domain"], part=d["part"], sid=d["source_ids"])
    return S["pi"]["tok"].shape


@torch.no_grad()
def teacher_cache(path=f"{D}/pi_teacher.pt"):
    """États finaux du professeur aux positions scorées + routage du professeur (appeler avec les poids du professeur)."""
    if os.path.exists(path):
        c = torch.load(path); S["teach"] = c["h"]; S["troute"] = c["route"]; return "loaded"
    lm = S["lm"]; A = S["pi"]; hs = []; routes = []
    R = Router(lm, positions=slice(P0, P1)); t0 = time.time()
    for i in range(A["tok"].shape[0]):
        h = lm(input_ids=A["tok"][i:i + 1].cuda().long()).last_hidden_state[0, P0:P1]
        hs.append(h.cpu()); routes.append(R.take())
    R.remove()
    S["teach"] = hs; S["troute"] = routes
    torch.save(dict(h=hs, route=routes), path)
    return "built in %.0fs" % (time.time() - t0)


@torch.no_grad()
def evaluate(rows=None, chunk=1024, route=True, keep_pos=False):
    """Évalue le modèle courant sur les fenêtres `rows` (défaut : toutes). Retourne la liste par fenêtre."""
    lm = S["lm"]; model = S["model"]; A = S["pi"]
    rows = list(range(A["tok"].shape[0])) if rows is None else list(rows)
    R = Router(lm, positions=slice(P0, P1)) if route else None
    per = []
    try:
        for i in rows:
            h = lm(input_ids=A["tok"][i:i + 1].cuda().long()).last_hidden_state[0, P0:P1]
            ht = S["teach"][i].cuda(); kw = []; aw = []
            for j in range(0, h.shape[0], chunk):
                lps = F.log_softmax(softcap(model.lm_head(h[j:j + chunk]).float()), -1)
                lpt = F.log_softmax(softcap(F.linear(ht[j:j + chunk], S["T_head"]).float()), -1)
                kw.append((lpt.exp() * (lpt - lps)).sum(-1).cpu()); aw.append((lps.argmax(-1) == lpt.argmax(-1)).cpu())
                del lps, lpt
            kw = torch.cat(kw); aw = torch.cat(aw)
            w = dict(row=i, domain=A["dom"][i], part=A["part"][i], n=int(kw.numel()), kl=float(kw.mean()), agree=float(aw.float().mean()),
                     kl_vec=kw.half())
            if route: w["route"] = routing_agreement(R.take(), S["troute"][i])
            per.append(w)
    finally:
        if R: R.remove()
    return per


def summarize(per, name):
    allk = torch.cat([w["kl_vec"].float() for w in per])
    agree = sum(w["agree"] * w["n"] for w in per) / sum(w["n"] for w in per)
    bydom = {}
    for d in sorted(set(w["domain"] for w in per)):
        ww = [w for w in per if w["domain"] == d]
        k = torch.cat([w["kl_vec"].float() for w in ww])
        bydom[d] = dict(kl=float(k.mean()), kl_p99=float(torch.quantile(k.double(), 0.99)),
                        agree=sum(w["agree"] * w["n"] for w in ww) / sum(w["n"] for w in ww), n=int(k.numel()), windows=len(ww))
    out = dict(protocol=name, windows=len(per), positions=int(allk.numel()), kl=float(allk.mean()),
               kl_p99=float(torch.quantile(allk.double(), 0.99)), agree=agree, by_domain=bydom,
               per_window=[{k: v for k, v in w.items() if k not in ("kl_vec", "route")} for w in per])
    if per and "route" in per[0]:
        tot = {}
        for w in per:
            for k, (ov, ch, n) in w["route"].items():
                t = tot.setdefault(k, [0.0, 0.0, 0]); t[0] += ov; t[1] += ch; t[2] += n
        by = {k: dict(overlap=v[0] / v[2], top1_changed=v[1] / v[2]) for k, v in tot.items()}
        out["routing"] = dict(mean_overlap=sum(v["overlap"] for v in by.values()) / len(by),
                              mean_top1_changed=sum(v["top1_changed"] for v in by.values()) / len(by), by_layer_application=by)
    return out


def report(per, vid, info=None):
    """Moitié dev imprimée et enregistrée dans results/ ; moitié conf et ensemble scellés."""
    dev = [w for w in per if w["part"] == "dev"]; conf = [w for w in per if w["part"] == "conf"]
    rd = summarize(dev, "P-I-dev"); rd["info"] = info or {}
    json.dump(rd, open(f"{RES}/pi_dev_{vid}.json", "w"), indent=1)
    if conf:
        json.dump(dict(conf=summarize(conf, "P-I-conf"), full=summarize(per, "P-I"), info=info or {}), open(f"{SEALED}/pi_{vid}.json", "w"))
    print("[%s] %s" % (vid, short(rd)) + ("  (moitié conf scellée)" if conf else ""), flush=True)
    return rd


def short(r):
    s = "%s KL=%.5f p99=%.4f top1=%.2f%%" % (r["protocol"], r["kl"], r["kl_p99"], 100 * r["agree"])
    if "routing" in r: s += " route=%.2f%% top1exp_chg=%.2f%%" % (100 * r["routing"]["mean_overlap"], 100 * r["routing"]["mean_top1_changed"])
    return s + " | " + " ".join("%s:%.4f" % (k[:6], v["kl"]) for k, v in r["by_domain"].items())


def boot_ratio(a, b, n=4000, seed=0):
    """Méthode V7 (summary7.py) : rapport des KL moyennes par token a/b − 1, bootstrap des fenêtres appariées."""
    A = {w["row"]: w for w in a}; B = {w["row"]: w for w in b}; rows = [r for r in A if r in B and A[r]["n"] > 0]
    tm = lambda X, rr: sum(X[r]["kl"] * X[r]["n"] for r in rr) / sum(X[r]["n"] for r in rr)
    rnd = random.Random(seed); bs = []
    for _ in range(n):
        rr = [rows[rnd.randrange(len(rows))] for _ in rows]
        bs.append(tm(A, rr) / tm(B, rr) - 1)
    bs.sort()
    return dict(gain=tm(A, rows) / tm(B, rows) - 1, ci95=[bs[int(.025 * n)], bs[int(.975 * n)]], windows_won=sum(A[r]["kl"] < B[r]["kl"] for r in rows), n=len(rows))


# ------------------------------------------------------------------------------------------------ mesures secondaires
@torch.no_grad()
def pw_teacher(path=f"{D}/pw_teacher.pt"):
    if os.path.exists(path):
        S["pw"] = torch.load(path); return "loaded"
    X = torch.load(f"{D}/pw_tokens.pt")["tokens"]; lm = S["lm"]; hs = []
    for b in range(0, X.shape[0], 8):
        hs.append(lm(input_ids=X[b:b + 8].cuda().long()).last_hidden_state[:, 256:].cpu())
    S["pw"] = dict(tok=X, h=torch.cat(hs)); torch.save(S["pw"], path); return "built"


@torch.no_grad()
def pw_eval():
    """P-W : KL et accord top-1 aux positions 256–511 de 40 fenêtres de Wikitext-2 (512 tokens)."""
    lm = S["lm"]; model = S["model"]; X = S["pw"]["tok"]; kls = []; ag = []
    for b in range(0, X.shape[0], 8):
        h = lm(input_ids=X[b:b + 8].cuda().long()).last_hidden_state[:, 256:]
        for k in range(h.shape[0]):
            lps = F.log_softmax(softcap(model.lm_head(h[k]).float()), -1)
            lpt = F.log_softmax(softcap(F.linear(S["pw"]["h"][b + k].cuda(), S["T_head"]).float()), -1)
            kls.append((lpt.exp() * (lpt - lps)).sum(-1).cpu()); ag.append((lps.argmax(-1) == lpt.argmax(-1)).float().cpu())
    k = torch.cat(kls); a = torch.cat(ag)
    return dict(protocol="P-W (Wikitext-2 test, 40 x 512, positions 256-511)", positions=int(k.numel()), kl=float(k.mean()),
                kl_p99=float(torch.quantile(k.double(), 0.99)), agree=float(a.mean()))


@torch.no_grad()
def _greedy32(texts, bs=25):
    tok = S["tok"]; model = S["model"]; tok.padding_side = "left"; outs = []
    for b in range(0, len(texts), bs):
        x = tok(texts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        o = model.generate(**x, max_new_tokens=32, min_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += o[:, x["input_ids"].shape[1]:].cpu().tolist()
    return outs


def pd32_teacher(path=f"{D}/pd32_teacher.json"):
    if os.path.exists(path):
        S["pd32"] = json.load(open(path)); return "loaded"
    P = json.load(open(f"{D}/pd32_prompts.json"))
    gen = _greedy32([p["text"] for p in P])
    S["pd32"] = dict(prompts=P, teacher=gen); json.dump(S["pd32"], open(path, "w")); return "built"


def pd32_eval():
    P = S["pd32"]["prompts"]; T = S["pd32"]["teacher"]; G = _greedy32([p["text"] for p in P])
    same = []; first = []; dom = {}
    for p, t, g in zip(P, T, G):
        fd = next((i for i, (a, b) in enumerate(zip(t, g)) if a != b), 32)
        same.append(fd == 32); first.append(fd)
        d = dom.setdefault(p["domain"], [0, 0, 0]); d[0] += fd == 32; d[1] += fd; d[2] += 1
    return dict(protocol="P-D32 (300 invites, glouton 32 tokens)", n=len(P), identical_share=sum(same) / len(same),
                mean_first_divergence=sum(first) / len(first),
                by_domain={k: dict(identical_share=v[0] / v[2], mean_first_divergence=v[1] / v[2], n=v[2]) for k, v in dom.items()})


# ------------------------------------------------------------------------------------------------ Vision
def _vis_inputs(s):
    return {k: (v.cuda() if k != "pixel_values" else v.cuda().to(torch.bfloat16)) for k, v in s.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}


@torch.no_grad()
def vision_teacher(path=f"{D}/vision/vision_teacher.pt", max_new=64):
    """Réponse gloutonne du professeur (64 tokens) pour 96 invites image + texte, puis états finaux du professeur sur ces
    64 positions (réponse imposée)."""
    if os.path.exists(path):
        S["vis"] = torch.load(path); return "loaded"
    data = torch.load(f"{D}/vision/vision_inputs.pt"); model = S["model"]; out = []
    for s in data:
        x = _vis_inputs(s)
        g = model.generate(**x, max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False)[0, x["input_ids"].shape[1]:]
        ids = torch.cat([x["input_ids"][0], g])[None]
        x2 = dict(x, input_ids=ids, attention_mask=torch.ones_like(ids))
        h = model.model(**x2).last_hidden_state[0, x["input_ids"].shape[1] - 1:-1]
        out.append(dict(domain=s["domain"], source=s["source"], gen=g.cpu(), h=h.cpu(), plen=x["input_ids"].shape[1]))
    S["vis"] = out; torch.save(out, path); return "built"


@torch.no_grad()
def vision_eval():
    data = torch.load(f"{D}/vision/vision_inputs.pt"); model = S["model"]; kls = []; ag = []; dom = {}
    for s, t in zip(data, S["vis"]):
        x = _vis_inputs(s); ids = torch.cat([x["input_ids"][0], t["gen"].cuda()])[None]
        x2 = dict(x, input_ids=ids, attention_mask=torch.ones_like(ids))
        h = model.model(**x2).last_hidden_state[0, t["plen"] - 1:-1]
        lps = F.log_softmax(softcap(model.lm_head(h).float()), -1)
        lpt = F.log_softmax(softcap(F.linear(t["h"].cuda(), S["T_head"]).float()), -1)
        k = (lpt.exp() * (lpt - lps)).sum(-1).cpu(); a = (lps.argmax(-1) == lpt.argmax(-1)).float().cpu()
        kls.append(k); ag.append(a); d = dom.setdefault(t["domain"], [[], []]); d[0].append(k); d[1].append(a)
    k = torch.cat(kls); a = torch.cat(ag)
    return dict(prompts=len(data), positions=int(k.numel()), kl=float(k.mean()), agree=float(a.mean()),
                by_domain={d: dict(kl=float(torch.cat(v[0]).mean()), agree=float(torch.cat(v[1]).mean())) for d, v in dom.items()})
