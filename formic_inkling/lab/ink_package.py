"""FORMIC-Ink — paquet complet (dérivé de formic_v7/lab/v7.py : package_vision, package_mtp, package_aux_files, load_package).
Nommage : llm.layers.{i}.mlp.experts.w13_weight.{e} / .w2_weight.{e} (une matrice E8 par expert), llm.layers.{i}.self_attn.*,
llm.embed, llm.unembed ; raw.bf16.<param> / raw.f32.<param> : petits tenseurs du texte, bit à bit dans le type du checkpoint ;
visual.<param> (E8, 5 bpw) et raw.visual.<param> ; mtp.<clé> (E8, 4 bpw, non évalué) et raw.mtp.<clé> ; aux.* : fichiers copiés."""
import os, sys, json, time, math, glob, shutil, subprocess
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formic_q as fq
import ink_quant as Q
CODEC = Q.CODEC; MODEL_DIR = "/root/models/inkling-2-mini"
sys.path.insert(0, CODEC)
import package


@torch.no_grad()
def e8_rtn_to_pkg(W2, bpw, pkg, name):
    fq.set_limit(16384)            # comme V7 (extension) : au-delà de ~3,6 bpw, les coquilles > 127 ne doivent pas être des échappements
    W2 = W2.float().cuda(); m, n = W2.shape; lo, hi = 0.002, 3.0
    for _ in range(22):
        mid = math.sqrt(lo * hi); r = fq.row_steps(W2, mid); _, hist = fq.quantize_rtn(W2, r)
        if fq.matrix_bits(hist, m, n) > bpw * m * n: lo = mid
        else: hi = mid
    r = fq.row_steps(W2, math.sqrt(lo * hi)); Qm, hist = fq.quantize_rtn(W2, r)
    U = torch.round(2.0 * Qm / r[:, None])
    assert torch.equal(U * (r[:, None] / 2.0), Qm)
    a = float(U.abs().max()); dt = np.int8 if a <= 127 else (np.int16 if a <= 32767 else np.int32)
    nb = package.write_tensor(pkg, name, U.cpu().numpy().astype(dt), r.cpu().numpy().astype(np.float16), None)
    return Qm, nb


def _raw(pkg, name, t):
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16: return package.write_raw(pkg, f"raw.bf16.{name}", t.view(torch.int16).numpy().view(np.uint16))
    return package.write_raw(pkg, f"raw.f32.{name}", t.float().numpy())


@torch.no_grad()
def package_vision(pkg, bpw=5.0):
    vis = Q.S["model"].model.vision_tower; nb = nq = nraw = rb = 0
    for name, p in vis.named_parameters():
        W2 = p.reshape(p.shape[0], -1) if p.dim() >= 2 else None
        if W2 is not None and W2.numel() >= 65536 and W2.shape[1] % 8 == 0:
            Qm, b = e8_rtn_to_pkg(W2, bpw, pkg, "visual." + name); p.copy_(Qm.reshape(p.shape).to(p.dtype)); nb += b; nq += p.numel()
        else:
            rb += _raw(pkg, "visual." + name, p); nraw += p.numel()
    return dict(e8_bytes=nb, e8_params=nq, raw_params=nraw, raw_bytes=rb, bpw=8 * nb / max(nq, 1))


@torch.no_grad()
def package_mtp(pkg, bpw=4.0):
    from safetensors import safe_open
    nb = nq = nraw = rb = 0
    with safe_open(os.path.join(MODEL_DIR, "mtp.safetensors"), "pt") as f:
        for k in sorted(f.keys()):
            p = f.get_tensor(k)
            W2 = p.reshape(p.shape[0], -1) if p.dim() >= 2 else None
            if W2 is not None and W2.numel() >= 65536 and W2.shape[1] % 8 == 0 and p.dtype in (torch.bfloat16, torch.float16, torch.float32):
                _, b = e8_rtn_to_pkg(W2, bpw, pkg, "mtp." + k); nb += b; nq += p.numel()
            else:
                rb += _raw(pkg, "mtp." + k, p); nraw += p.numel()
    return dict(e8_bytes=nb, e8_params=nq, raw_params=nraw, raw_bytes=rb, bpw=8 * nb / max(nq, 1))


def package_aux(pkg):
    n = 0
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "processor_config.json"):
        src = os.path.join(MODEL_DIR, f)
        if os.path.exists(src): shutil.copy(src, os.path.join(pkg, "aux." + f)); n += os.path.getsize(src)
    td = os.path.join(MODEL_DIR, "tiktoken")
    if os.path.isdir(td):
        os.makedirs(os.path.join(pkg, "aux.tiktoken"), exist_ok=True)
        for f in os.listdir(td): shutil.copy(os.path.join(td, f), os.path.join(pkg, "aux.tiktoken", f)); n += os.path.getsize(os.path.join(td, f))
    json.dump(dict(format="FORMIC-Ink E8 package v1", model="thinkingmachines/Inkling-2-Mini@b85c3cfa615fc413cc9af3379f491b963f3b586a",
                   decode="W = bf16_rne(f32(U) * (f32(step)/2)) ; experts : tranche [e] des tenseurs fusionnés gate_up_proj [E,2304,2304] ([gate;up]) / down_proj [E,2304,1152]",
                   unembed="200 058 lignes utiles (vocab_size 201 024 : lignes de remplissage absentes)"),
              open(os.path.join(pkg, "aux.formic_ink_format.json"), "w"), indent=1)
    return n


def param_for(name):
    """nom du paquet -> (paramètre, index expert ou None)."""
    lm = Q.S["lm"]; model = Q.S["model"]
    if name == "llm.embed": return lm.embed_tokens.weight, None
    if name == "llm.unembed": return model.lm_head.weight, None
    if name.startswith("llm.layers.") and ".mlp.experts." in name:
        p = name.split("."); i = int(p[2]); w = p[5]; e = int(p[6])
        ex = lm.layers[i].mlp.experts
        return (ex.gate_up_proj if w == "w13_weight" else ex.down_proj), e
    if name.startswith("llm."):
        m = lm
        for a in name[4:].split("."): m = getattr(m, a)
        return m.weight, None
    if name.startswith("visual."):
        return dict(model.model.vision_tower.named_parameters())[name[7:]], None
    return None, None


@torch.no_grad()
def load_package(pkg, tmp="/dev/shm/ink_pkgdec", workers=48, blank=True):
    """Reconstruit texte + Vision DEPUIS LES FICHIERS (décodeur CPU séparé). blank=True : tous les paramètres concernés sont
    d'abord mis à NaN, puis on vérifie qu'aucun NaN ne subsiste (le paquet couvre tout)."""
    model = Q.S["model"]; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    if blank:
        for n, p in model.named_parameters(): p.fill_(float("nan"))
        for n, b in model.named_buffers():
            if "e_score_correction_bias" in n: b.fill_(float("nan"))
    proc = subprocess.Popen([sys.executable, "-I", "-B", CODEC + "/decode_stream.py", pkg, tmp, str(workers), "20"],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=open(tmp + "_stdout.log", "w"), stderr=subprocess.STDOUT)
    n = 0; t0 = time.time()
    while True:
        files = sorted(glob.glob(tmp + "/*.npy")); done = os.path.exists(tmp + "/_DONE")
        if not files:
            if done: break
            if proc.poll() is not None and not os.path.exists(tmp + "/_DONE"): raise RuntimeError("décodeur : " + open(tmp + "_stdout.log").read()[-1500:])
            time.sleep(0.2); continue
        for f in files:
            name = os.path.basename(f)[:-4]; a = np.load(f); os.remove(f)
            if name.startswith("mtp."): continue
            P, e = param_for(name)
            if P is None: raise RuntimeError("nom inconnu " + name)
            t = torch.from_numpy(a).cuda()
            if e is None: P.copy_(t.reshape(P.shape).to(P.dtype))
            else: P[e].copy_(t.to(P.dtype))
            n += 1
    man = package.load_manifest(pkg); params = dict(model.named_parameters()); bufs = dict(model.named_buffers()); nr = 0
    for k, v in man.items():
        if v["kind"] != "raw": continue
        _, dt, name = k.split(".", 2)
        if name.startswith("mtp."): continue                  # drafter : non chargé dans ce graphe (non évalué)
        a = package.read_raw(pkg, k)
        t = torch.from_numpy(a.view(np.int16)).view(torch.bfloat16) if dt == "bf16" else torch.from_numpy(a)
        P = params.get(name) if name in params else (bufs.get(name) if name in bufs else dict(model.model.vision_tower.named_parameters()).get(name[7:]) if name.startswith("visual.") else None)
        if P is None: raise RuntimeError("petit tenseur inconnu " + k)
        P.copy_(t.reshape(P.shape).to(P.device, P.dtype)); nr += 1
    shutil.rmtree(tmp, ignore_errors=True)
    nan = [n_ for n_, p in model.named_parameters() if torch.isnan(p).any()]
    nan += [n_ for n_, b in model.named_buffers() if "e_score_correction_bias" in n_ and torch.isnan(b).any()]
    if nan: raise RuntimeError("paramètres non couverts par le paquet : %s" % nan[:10])
    return dict(e8_loaded=n, raw_loaded=nr, seconds=time.time() - t0)
