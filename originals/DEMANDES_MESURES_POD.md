# Demandes Prioritaires de Mesures Pod (File A → B) — Chantier LUMEN / FORMIC-Max

**Émetteur :** Agent A (Recette LUMEN / FORMIC-Max — Version 2 révisée) — **Destinataire :** Agent B (Architecture & Pod)  
**Date :** 12 novembre 2026 — **Référence d'interface :** `COORDINATION_CHANTIERS_LUMEN.md` & `MANIFESTE_INTERFACE_A_B.json`  
**Statut du canal :** **À TRANSMETTRE MANUELLEMENT** (aucun canal partagé actif ; aucune acceptation ni exécution par B n'est présumée).  
**Périmètre réel et gouvernance :** Conformément à `COORDINATION_CHANTIERS_LUMEN.md`, cette file est plafonnée à **trois demandes simultanées au statut `proposed`**, conçues pour ne pas détourner le pod du sprint architectural de B. `REQ-A2B-01` cible un panel de 4 couches (`{0, 30, 31, 63}`), `REQ-A2B-02` cible **2 macro-blocs consécutifs (`layers 0..7`, repli minimal sur `layers 0..3`)**, et `REQ-A2B-03` cible uniquement `lm_head` et les `15` tenseurs propres du module `MTP`. Toutes les partitions (`calib_train`, `select_dev`) suivent les définitions uniques gelées dans `MANIFESTE_INTERFACE_A_B.json`.

---

## Tableau normalisé des trois demandes (`COORDINATION_CHANTIERS_LUMEN.md` §3)

```text
request_id ; hypothesis_id ; question ; required_artifacts ; baseline_id ; metric ; data_split ; expected_cost_or_timing_needed ; smallest_useful_scope ; status
REQ-A2B-01 ; H1_MLP_SWIGLU_BGCC_AND_KA_RDOQ_TRANSFER ; L'avantage de E8 sur S0 (+0.7 dB rapporte a l'Etape 1 sur 256 lignes aleatoires de in_proj_qkv/z couche 0) se transfere-t-il aux MLP SwiGLU et aux couches profondes, et quels gains apportent isolement la courbure Kronecker YAQA et le couplage bilineaire SwiGLU BGCC a octets reels encodes egaux ? ; Ratio octets_encodes/entropie du codec Lumen actuel + matrices H_in et H_O bloc-128 (4 sondes Rademacher-Fisher) sur layers.{0, 30, 31, 63} pour {in_proj_qkv (Q,K,V ventiles), mlp.gate_proj, mlp.up_proj, mlp.down_proj, self_attn.o_proj} ; L0 et S0 exacts de l'Etape 1 a debit encode egal (~2.65 - 2.85 bpw) ; (1) NMSE locale ventilee Q/K/V et sortie post-SwiGLU, (2) KL exacte vocabulaire complet 248320 apres greffe unitaire dans A0 BF16, (3) Octets reels du codec ; calib_train (sous-ensemble 256 seq x 2048 tok) pour calibration et select_dev pour mesure ; ~25 a 35 min sur 1x H100 80 Go (~1.2 Go disque) ; Uniquement layer 0 mlp.{gate,up,down_proj} + layer 30 mlp.{gate,up,down_proj} et in_proj_qkv ; proposed
REQ-A2B-02 ; H2_INTERLAYER_DRIFT_GPTAQ_VS_POSITIONAL_PROFILE ; Sur deux macro-blocs consecutifs quantifies (layers 0..7), la perte de KL provient-elle principalement de la derive d'entree inter-couches dX = X_fp - X_q (corrigee par GPTAQ) ou croit-elle avec la position dans la sequence (0-1k vs 3k-4k) a etats GDN FP32 et KV BF16 intacts ? ; Greffe des 2 macro-blocs layers 0..7 quantifies en L0 symetrique vs L0 + GPTAQ asymetrique vs L0 + KA-BGCC-RDOQ complet, avec profil de KL exacte par tranche de position ; L0 symetrique (||Wq Xq - W Xq||^2) vs L0 asymetrique GPTAQ (||Wq Xq - W X_fp||^2) sur memes prefixes ; KL exacte sur vocabulaire complet ventilee par position [0..1024, 1024..2048, 2048..4096] + norme relative ||S_t(Q) - S_t(BF16)||_F / ||S_t(BF16)||_F ; calib_train pour calibration et select_dev (256 seq x 4096 tok) pour evaluation ; ~35 a 50 min sur 1x H100 80 Go (1 seul bloc en VRAM selon Alg. 2 GPTAQ) ; 1 seul macro-bloc complet (layers 0..3 = 3 GDN + 1 Full Attn + 4 MLP) ; proposed
REQ-A2B-03 ; H3_TOP128_KL_BIAS_MTP_BRANCH_AND_QERA_HEAD ; L'estimateur KL Top-128 + log-masse restante classe-t-il sans biais les allocations face a la KL complete, quel debit minimal preserve le taux d'acceptation speculatif du module MTP, et un correcteur Kronecker-QERA r=64 sur lm_head bat-il le reinvestissement de ses 13 Mo en bits E8 ? ; Comparaison KL exacte 248320 vs Top-128+tail sur select_dev + test d'acceptation speculative MTP (15 tenseurs propres a 2.4 vs 2.8 vs 3.2 bpw) + duel lm_head (L0 + QERA r=64 INT8 vs L0 avec +0.08 bpw E8) ; L0 sur lm_head et MTP BF16 de reference ; Spearman rho et erreur relative du proxy KL Top-128+tail ; taux d'acceptation alpha_MTP sur 64 prompts ; KL exacte finale pour le duel lm_head ; select_dev (256 seq x 4096 tok incluant code, maths, XML outils, image et video) ; ~25 a 40 min sur 1x H100 80 Go ; Uniquement lm_head + module MTP (15 tenseurs) avec tronc A0 BF16 fixe ; proposed
```

---

## Justification scientifique et exploitation conditionnelle par l'Agent A

### 1. `REQ-A2B-01` — Vrai transfert au-delà de la couche 0 (MLP SwiGLU, couches profondes et ventilation Q/K/V)
- **Contexte factuel rectifié :** L'échantillon de 256 lignes de l'Étape 1 ayant été tiré au hasard (`47` lignes Q, `45` lignes K, `164` lignes V — `rapporté par le projet`), Q, K et V étaient déjà représentés dans la moyenne globale de `in_proj_qkv` en couche 0. La véritable limite de l'Étape 1 est qu'elle ne couvre **que la couche 0 et aucune matrice MLP SwiGLU** (`61,6 %` des paramètres du modèle). En outre, notre lecture directe par `HTTP Range` de tranches de `layers.0.linear_attn.in_proj_qkv` montre une forte hétérogénéité statistique interne entre sous-tranches (`kurtosis = 3,23` sur Q et `3,14` sur K contre `11,01` sur V — `mesuré/reproduit par toi`), et notre test sur vraies paires `(gate_proj, up_proj)` des couches 0 et 31 montre un gain hors-échantillon de `+0,40` à `+0,61 dB` pour le couplage bilinéaire `BGCC` lorsque $N_{\text{cal}} / n = 16$.
- **Précaution d'implémentation pour $H_O$ :** Ne pas exécuter un backward naïf de $D_{\mathrm{KL}}(p_{\text{BF16}} \| p_{\text{BF16}})$ (dont le gradient est identiquement nul), mais propager `4` sondes Rademacher-Fisher $u_t = \mathrm{diag}(\sqrt{p_t})\zeta_t - p_t(\sqrt{p_t}^\top \zeta_t)$.
- **Règle de décision autonome de A :** `BGCC` et `KA-RDOQ` ne sont activés en campagne complète que sur les familles où `REQ-A2B-01` (ou l'étape pilote `S1`) confirme une baisse de la KL de greffe sur `select_dev` à octets réels égaux.

### 2. `REQ-A2B-02` — Séparation Dérive Inter-Couches (`GPTAQ`) vs Dérive Positionnelle (*Gated DeltaNet*)
- **Question tranchée :** `Minima` (`arXiv:2609.04098`, `publié`) observe en `NVFP4 W4A4` (`17,5 Gio`) que l'erreur récurrente de *Gated DeltaNet* plafonne rapidement et que l'écart de perplexité diminue avec la position, mais ce résultat n'est pas établi à `~2,65–2,89 bpw` en $E_8$. Tester `layers 0..7` sur `4 096` tokens (sans remise à zéro artificielle des états intra-séquence, avec états GDN `FP32` et KV `BF16` de référence) sépare proprement :
  - une erreur dominée par l'empilement des couches $\Delta X = X_{\text{fp}} - X_q$ dès les premiers `1 024` tokens (traitée directement par `GPTAQ` et une récupération courte à `4k` tokens) ;
  - d'une éventuelle dérive croissant entre `0–1k` et `2k–4k` (qui justifierait alors de monter graduellement à `16k` tokens pré-chauffés en Passe 3b).

### 3. `REQ-A2B-03` — Validation du proxy `KL Top-128`, Protection de la branche `MTP` et Test à coût égal sur `lm_head`
- **Trois vérifications rapides à fort impact budgétaire :**
  1. Vérifier que le cache `Top-128 logits FP16 + indices int32 + log-masse restante FP32` (`772` octets/token, soit `3,24 Go` pour `4,19M` tokens au lieu de `2,08 To` en vocabulaire plein `248 320`) classe les candidats exactement comme la KL complète.
  2. Comme `Qwen3_5ForConditionalGeneration.forward` ne traverse pas `mtp.*` lors du calcul des logits principaux, vérifier la sensibilité du taux d'acceptation spéculatif $\alpha_{\text{MTP}}$ entre `2,4`, `2,8` et `3,2 bpw` afin de fixer le plancher disjoint du MTP dans `EB-RCO-Slack`.
  3. Tester sur la vraie matrice `lm_head` `[248320, 5120]` si un correcteur `Kronecker-QERA` de rang `64` en `INT8` (`~16,2 Mo`, soit seulement `+0,0128 bpw` à cette grande dimension) bat ou non l'ajout de `+0,08 bpw` dans la grille `E8` de `lm_head`.

---

## Conditions de validité des artefacts transmis entre B et A

- Tout cache d'activations ou de Hessiennes partagé par B doit préciser : la révision du modèle (`37561816e66b5b9153b1b64c10968c18c1e6f4a0`), l'architecture (`A0` vs `A1`), l'origine des entrées (`BF16` intact vs préfixe quantifié), la base (canonique vs post-rotation), et la politique d'état initial (`zéro` au début du document uniquement).
- Si B modifie les dimensions ou insère de nouvelles rotations inter-couches dans `A1`, toutes les statistiques $(H_{\text{in}}, H_O, \Delta C_X)$ collectées sur `A0` sont invalidées et devront être recalculées sur `A1` avant l'évaluation de `A1Q1`.
