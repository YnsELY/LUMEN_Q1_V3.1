"""Calculs de l'étude de faisabilité LUMEN × Inkling 2 Mini (22 novembre 2026).

Aucun poids n'est lu : tout part des champs de config.json publiés, des tailles de fichiers publiées et de la
courbe KL publiée par Unsloth. Les résultats sont des ESTIMATIONS. Lancer avec un Python qui a numpy :
    python -B calculs_inkling.py
"""
import numpy as np

G = 1e9

# ---------------------------------------------------------------- 1. Inventaire des poids (config.json)
H, L, E, I, V = 2304, 24, 128, 1152, 201024
NQ, NKV, HD = 24, 8, 128
expert = 3 * H * I                                   # gate + up + down (w13 fusionné + w2)
experts_layer = E * expert
experts_all = L * experts_layer
attn_layer = H * NQ * HD + 2 * H * NKV * HD + NQ * HD * H   # wq, wk, wv, wo seuls
attn_all = L * attn_layer
router_all = L * E * H
vocab = 2 * V * H                                    # embed + unembed (non liés)
vis_layers = 23 * 12 * 1024 ** 2                     # 23 blocs ViT largeur 1024, mlp_ratio 4
vis_proj = (4 * 1024) * 4096 + 4096 * 2304           # fusion 2x2 puis projection vers 2304
vis_patch = 3 * 2 * 20 * 20 * 1024
vision = vis_layers + vis_proj + vis_patch
drafter = 636e6 / 2                                  # mtp.safetensors, 636 Mo supposés en BF16

ident = experts_all + attn_all + router_all + vocab + vision
print("== 1. Inventaire (poids)")
for k, v in [("expert unique", expert), ("experts / couche", experts_layer), ("experts, 24 couches", experts_all),
             ("  prélude 0-5", 6 * experts_layer), ("  coeur bouclé 6-17", 12 * experts_layer),
             ("  coda 18-23", 6 * experts_layer), ("  dont NVFP4 0-19", 20 * experts_layer),
             ("  dont MXFP8 20-23", 4 * experts_layer), ("attention wq/wk/wv/wo", attn_all),
             ("routeurs", router_all), ("embed + unembed", vocab), ("Vision (estimée)", vision),
             ("drafter DSpark", drafter), ("TOTAL identifié hors drafter", ident)]:
    print(f"  {k:30s} {v / G:9.4f} G")

# Reconstitution de la taille du checkpoint pour borner les tenseurs non identifiés
nvfp4_bpw = 4 + 8 / 16          # code E2M1 + échelle FP8 E4M3 par bloc de 16
mxfp8_bpw = 8 + 8 / 32          # code E4M3 + échelle E8M0 par bloc de 32
ck = (20 * experts_layer * nvfp4_bpw + 4 * experts_layer * mxfp8_bpw) / 8 \
     + (attn_all + router_all + vocab + vision) * 2
print(f"  checkpoint reconstitué        {ck / G:9.3f} Go (publié : 19,5 Go)")
missing = (19.5 * G - ck) / 2
print(f"  écart -> tenseurs BF16 non identifiés ≈ {missing / G:.3f} G poids ; total ≈ {(ident + missing) / G:.2f} G (publié 26,2-26,4)")
annex = attn_all + router_all + missing           # attention + petits tenseurs non experts
text = experts_all + annex + vocab
print(f"  chemin texte retenu           {text / G:9.3f} G (experts {experts_all / text:.1%})")

# ---------------------------------------------------------------- 2. Entropie des codes NVFP4 (simulation)
print("\n== 2. NVFP4 : entropie des codes (simulation, blocs de 16, échelle absmax/6 arrondie en E4M3)")
rng = np.random.default_rng(0)
grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
def e4m3(x):
    e = np.floor(np.log2(x)); return np.round(x / 2 ** e * 8) / 8 * 2 ** e
Hs = []
for name, gen in [("gauss", lambda n: rng.standard_normal(n)), ("laplace", lambda n: rng.laplace(size=n)),
                  ("student5", lambda n: rng.standard_t(5, n))]:
    w = gen(16 * 400000).reshape(-1, 16)
    s = e4m3(np.abs(w).max(1, keepdims=True) / 6)
    idx = np.abs((np.abs(w) / s)[..., None] - grid).argmin(-1)
    code = np.where(idx == 0, 0, idx * np.sign(w).astype(int))
    _, c = np.unique(code, return_counts=True); p = c / c.sum()
    h = -(p * np.log2(p)).sum(); Hs.append(h)
    q = np.sign(w) * grid[idx] * s
    snr = 10 * np.log10((w ** 2).mean() / ((w - q) ** 2).mean())
    print(f"  {name:9s} H(code) = {h:.3f} bit/poids ; SNR du grillage NVFP4 = {snr:.1f} dB")
nvfp4_lossless = max(Hs) + 0.4    # échelles FP8 : 0,5 bit brut, ~0,4 après codage (hypothèse)
print(f"  NVFP4 conservé bit pour bit + codage entropique ≈ {nvfp4_lossless:.2f} bpw (brut 4,50)")

# ---------------------------------------------------------------- 3. Courbe publiée Unsloth (KL vs UD-Q8_K_XL)
print("\n== 3. Courbe Unsloth (texte seul ; bpw = 8 × octets / poids du chemin texte)")
uns = [("UD-IQ1_S", 7.76, 0.0911, 89.4), ("UD-IQ1_M", 8.02, 0.0867, 90.2), ("UD-IQ2_XXS", 8.49, 0.0726, 90.6),
       ("UD-Q2_K_XL", 9.14, 0.0595, 91.6), ("UD-IQ3_XXS", 11.66, 0.0269, 94.7), ("UD-Q3_K_XL", 13.63, 0.0174, 96.3),
       ("UD-IQ4_XS", 15.73, 0.0109, 97.7), ("UD-Q4_K_M", 16.61, 0.0108, 97.9), ("UD-Q4_K_XL", 17.63, 0.0107, 98.0)]
for n, gb, kl, t1 in uns:
    print(f"  {n:11s} {gb:6.2f} Go  {8 * gb * G / text:5.2f} bpw  KL {kl:.4f}  top-1 {t1:.1f} %")
fit = [(8 * gb * G / text, np.log(kl)) for _, gb, kl, _ in uns if gb < 14]
b = np.array(fit); slope, icpt = np.polyfit(b[:, 0], b[:, 1], 1)
print(f"  ajustement ln KL = {icpt:.3f} {slope:+.3f}·bpw sur 2,4-4,2 bpw -> KL × {np.exp(slope):.3f} par bit")
def kl_uns(r): return np.exp(icpt + slope * r)
lk = np.log([u[2] for u in uns[::-1]]); t1s = [u[3] for u in uns[::-1]]
def top1(kl): return np.interp(np.log(kl), lk, t1s)

# ---------------------------------------------------------------- 4. Gain LUMEN transposé (hypothèse)
f1_vs_arki = 0.0613 / 0.0915            # moitié scellée du protocole public, RAPPORT_V7 §4
arki_vs_uns = 0.0823 / 0.0983           # publié par Arki05, même protocole, Qwen3.8-27B
g_lo, g_hi = f1_vs_arki * arki_vs_uns, 0.75
print(f"\n== 4. Gain LUMEN à octets égaux : F1/Arki05 = {f1_vs_arki:.3f} ; Arki05/Unsloth (publié) = {arki_vs_uns:.3f}")
print(f"  hypothèse de transposition : KL_LUMEN = g × KL_Unsloth, g ∈ [{g_lo:.2f} ; {g_hi:.2f}]")
print(f"  équivalent en débit : {np.log(g_hi) / slope:.2f} à {np.log(g_lo) / slope:.2f} bpw économisés à KL égale")

# ---------------------------------------------------------------- 5. Paliers (paquet complet)
print("\n== 5. Paliers du paquet complet (texte + Vision 5 bpw + drafter 4 bpw + annexes 0,06 Go)")
extra = (vision * 5 + drafter * 4) / 8 + 0.06 * G
print(f"  Vision + drafter + annexes = {extra / G:.3f} Go")
for r in [2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.6]:
    t = (experts_all * r + annex * 4.5 + vocab * (r + 0.5)) / 8
    r_eff = 8 * t / text
    k1, k2 = g_lo * kl_uns(r_eff), g_hi * kl_uns(r_eff)
    print(f"  experts {r:.1f} bpw : texte {t / G:5.2f} Go ({r_eff:.2f} bpw moyen), complet {(t + extra) / G:5.2f} Go ;"
          f" KL estimée {k1:.3f}-{k2:.3f} ; top-1 ~{top1(k2):.1f}-{top1(k1):.1f} % ; Unsloth à ce débit {kl_uns(r_eff):.3f}")
qe = (20 * experts_layer * nvfp4_lossless + 4 * experts_layer * 4.0 + annex * 5 + vocab * 5) / 8 + extra
print(f"  quasi exact (NVFP4 conservé, MXFP8 -> E8 4 bpw, reste 5 bpw) : {qe / G:.2f} Go")
qe2 = (20 * experts_layer * nvfp4_lossless + 4 * experts_layer * 4.0) / 8 + (annex + vocab) * 2 + extra
print(f"  variante avec attention/vocabulaire en BF16                 : {qe2 / G:.2f} Go")

# ---------------------------------------------------------------- 6. Calibration MoE et lecture par token
print("\n== 6. Calibration et lecture mémoire")
for ntok in [409_600, 2_000_000, 4_000_000]:
    per = ntok * 8 / E
    print(f"  {ntok:>9,d} tokens -> {per:>9,.0f} tokens/expert en moyenne (coeur bouclé : ×3 = {3 * per:,.0f}) ; ratio/dim 2304 = {per / H:.1f}")
act_expert = 8 * expert
per_tok = (48 * act_expert * 2.8 + (6 + 36 + 6) * (attn_layer) * 4.5 + V * H * 3.3) / 8
print(f"  octets lus par token à 2,8 bpw (borne haute, poids re-lus à chaque passe) ≈ {per_tok / G:.2f} Go"
      f" (Qwen F1 : 9,45 Go)")
