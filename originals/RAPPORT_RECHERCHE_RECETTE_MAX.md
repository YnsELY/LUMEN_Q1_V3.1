# Rapport de Recherche Scientifique — Recette LUMEN pour `FORMIC-Max` (`≤ 10 000 000 000` octets)

**Chantier :** Agent A (Recette, Allocation, Récupération & Format) — **Version :** 2.1 (12 novembre 2026, révision vérifiée post-audit)  
**Modèle cible épinglé :** `Qwen/Qwen3.8-27B`, révision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` (`1 199` tenseurs physiques dont `520` matrices 2D et `679` petits tenseurs tous stockés en `BF16`, `27 781 427 952` paramètres uniques, `55 562 855 904` octets — **`mesuré/reproduit par toi`** sur `config.json`, `model.safetensors.index.json` et en-têtes binaires partiels lus par `HTTP Range`).  
**Plafond principal `FORMIC-Max` :** **`10 000 000 000` octets complets** (`Texte + Vision + MTP + tous auxiliaires`, soit $\le \mathbf{2,87962\text{ bpw}}$ effectif sur l'ensemble des paramètres).  
**Statuts des affirmations utilisés strictement :** `publié`, `rapporté par le projet`, `mesuré/reproduit par toi`, `démontré sous hypothèses`, `estimé`, `à tester`.

---

## 0. Résumé décisionnel : comment dépasser les limites actuelles sous 10 Go

Le repère concurrentiel le plus instructif sur `Qwen/Qwen3.8-27B` est le dépôt `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` (`publié`, tailles LFS `mesurées/reproduites par toi`) : son build `IQ2_S` pèse déjà `9 259 510 912` octets en texte seul (`10 539 127 648` octets avec `mmproj` Vision `BF16` et `MTP`, dépassant notre plafond de `10,00 Go`) et accuse une baisse de `-3,54 points` sur `GPQA-Diamond` (`86,36` vs `89,90` sur le harnais `llama.cpp` des auteurs) et de `-3,42 points` sur `LiveCodeBench v6` (`82,29` vs `85,71`). Le seul build `GSQ-RCO` passant sous `10,00 Go` complets est `IQ2_XS` (`9 702 458 208` octets complets, dont `8,42 Go` de texte), qui recule de `-5,05 points` sur `GPQA-Diamond` et de `-9,14 points` sur `LiveCodeBench v6`.

Pour que `FORMIC-Max` ait une chance sérieuse de préserver les capacités du modèle original sous `10 000 000 000` octets, il ne suffit donc pas d'égaler `GSQ-RCO` : **il faut cumuler un avantage structurel d'octets utiles et un avantage algorithmique de quantification.** Nos vérifications mathématiques et nos tests sur extraits réels de poids `BF16` de `Qwen/Qwen3.8-27B` (`budget_lumen_max.py`) aboutissent à trois propositions finalistes complémentaires et quatre décisions d'assainissement :

1. **Finaliste 1 (`0 octet` ajouté au package) — `KA-BGCC-RDOQ` : Arrondi $E_8$ par Onde Frontale de Kronecker, Cible Asymétrique et Couplage Bilinéaire SwiGLU :**
   - Associe la rétroaction hors-diagonale de sortie `YAQA` (`arXiv:2505.22988v3`, estimée sans biais par sondes Rademacher-Fisher), la compensation asymétrique de la dérive amont `GPTAQ` (`arXiv:2504.02692v3`), et un **nouveau mécanisme de Calibration Bilinéaire Couplée (`BGCC`)** exploitant la structure multiplicative $\mathrm{SiLU}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x)$ des 64 MLP SwiGLU (`61,6 %` des poids du modèle) pour que `up_proj` compense l'erreur réelle déjà figée dans `SiLU(gate_proj)` (complété par `1–2` balayages Gauss-Seidel `ABCD-E8`).
   - Sur vraies paires `(gate_proj, up_proj)` extraites des couches `0` et `31` de `Qwen/Qwen3.8-27B` (`EXP-V2-02`, validation hors-échantillon à `2,50 bpw` sous ratio $N_{\text{cal}}/n = 16$), `BGCC` réduit l'erreur quadratique de sortie SwiGLU de **`8,73 %` (`+0,40 dB`) en couche 0** et de **`13,27 %` (`+0,61 dB`) en couche 31** (`mesuré/reproduit par toi` sur sous-matrices réelles ; transfert au modèle complet `à tester`). Sur une chaîne honnête de plein rang à 2 couches (`EXP-V2-03`), `KA-RDOQ` réduit la KL de validation de `39,2 %` en régime power-law modéré et de `31,2 %` en régime quasi-isotrope défavorable.
2. **Finaliste 2 — `SU-EB-RCO` : Codeur Sphérique Exact `SU-E8-rANS`, Format Sobre `LUMEN-v2` (`84,07 Mo` d'auxiliaires) et Allocation Conjointe Multi-Budgets :**
   - En remplaçant un codage asymétrique par coordonnées et des auxiliaires lourds par le codeur sphérique exact sans table **`SU-E8-rANS`** (`redondance mesurée +0,0011 à +0,0025 bpw`, `3,50 Kio` de tables sur tout le modèle, préservant l'équiprobabilité exacte de $D_8$ et $D_8 + \frac{1}{2}$ et reproduisant l'avantage de **`+0,71 dB` de $E_8$ sur `S0`** sur les vrais poids de couche 0 en `EXP-V2-01`), combiné à des super-groupes `512` et super-tuiles adaptatives (`84,07 Mo` d'auxiliaires totaux) et à une allocation Water-Filling puis `RCO-Slack` utilisant `9,9815 Go` sur `10,0000 Go` (garde technique de `18,52 Mo`), le **cœur texte grimpe à `9,6329 Go` (`2,865 bpw` effectif)** — soit **`+373,4 Mo` au-dessus de `GSQ-RCO IQ2_S` texte seul (`9,2595 Go`)**, tout en logeant `Vision` (`202,55 Mo`, `3,517 bpw` effectif) et `MTP` (`146,00 Mo`, `2,750 bpw` effectif) sous le plafond (`estimé`). Inclut obligatoirement une **branche de perte et un plancher de budget dédiés pour le module `MTP`**, dont le gradient serait sinon identiquement nul.
3. **Finaliste 3 — `EAR-PV` : Récupération Hybride Continue et Voisinage Kissing-$E_8$ avec Validation après Décodage et Rollback :**
   - Encadre la Passe 3b par : (i) un duel pilote préalable à temps GPU égal entre hors-politique et `GKD` (`arXiv:2306.13649v3`), (ii) un cache enseignant `Top-128` comptabilisant exactement ses indices `int32` (`3,24 Go` pour `4,19M` tokens), et (iii) une alternance entre réglage continu des échelles et recherche locale sur les `240` voisins de Kissing de $E_8$ sous pénalité duale $-\mu_{\text{dual}}\log_2 \widehat{P}_{\text{shell}}(m)$, assortie d'un **rejet automatique (rollback)** si la KL mesurée sur `select_dev` **après rechargement depuis le fichier décodé** ne diminue pas.
4. **Décisions fermes d'assainissement :**
   - **Correction sur l'Étape 1 :** l'échantillon de 256 lignes de `in_proj_qkv` ayant été tiré au hasard (`47 Q, 45 K, 164 V`, `rapporté par le projet`), la vraie limite de l'Étape 1 est sa restriction à la couche 0 sans MLP SwiGLU ; la ventilation `Q/K/V` se justifie quant à elle par l'hétérogénéité réelle mesurée de leur kurtosis (`3,23` sur Q et `3,14` sur K contre `11,01` sur V en couche 0).
   - **Désactivation par défaut du scaling d'entrée ($D_{\text{in}} = I$)** (`-1,30 dB` reproduit sur vraies entrées de couche 0 en `EXP-V2-01`) et **maintien de la suspension** de `lm_head` $\leftarrow$ `embed_tokens`.
   - **Conversion en scores de benchmarks finaux (`LiveCodeBench v6`, `Terminal-Bench`, `DeepSWE`, `GPQA-Diamond`) :** **Non estimable avec les données disponibles.**

---

## 1. Note de diagnostic initial : état réel, correction de périmètre et taxonomie des pertes

### 1.1 Ce qui est connu, ce qui est absent et ce que nos vérifications directes ajoutent
- **Présent et lu (`rapporté par le projet`) :** `ANALYSE_ET_PRIORITES_LUMEN_MAX_V2.md` et `COORDINATION_CHANTIERS_LUMEN.md`. Les fichiers `FORMIC_RECETTE_V2.md`, `Code/results/RESUME_ETAPE1.md`, `step1_real_layer.py`, le code du codec Lumen et le manifeste de l'Agent B sont **absents** de cet espace de travail.
- **Éléments de l'Étape 1 (`rapporté par le projet`, corrigés après audit) :** essais à `2,5 bpw` sur `48 000` tokens (`7 020` IDs distincts, `4` livres) portant sur `256` lignes **tirées au hasard** (`rng.choice`, graine 0 : `47` lignes Q couvrant `14/16` têtes, `45` lignes K couvrant `15/16` têtes, `164` lignes V couvrant `48/48` têtes) de `in_proj_qkv` et `in_proj_z` de la **couche 0** (`RTN` : `-21,0/-20,9 dB`, scalaire équitable `S0` : `-26,4/-26,3 dB`, meilleur `E8` `L0` : `-27,1/-27,1 dB`, Hadamard `+0,3` à `+0,4 dB`, scaling d'entrée `-1,5` à `-2,4 dB`, prédicteur `lm_head` `0,000 bit`). La limite structurelle de l'Étape 1 est donc l'absence de test sur les **64 MLP SwiGLU**, les **16 couches Full Attention**, les **couches profondes** et les **10 domaines hors prose littéraire**.
- **Faits nouveaux établis directement (`mesuré/reproduit par toi`) :**
  1. Vérification sur la révision épinglée `37561816e66b5b9153b1b64c10968c18c1e6f4a0` (et par lecture `HTTP Range` des en-têtes binaires des shards `00001, 00003, 00009, 00018`) que tous les `1 199` tenseurs sont en `BF16` et que `mtp.layers.0.mlp.down_proj.weight` est de forme `[5120, 17408]`.
  2. Mesure directe sur les lignes complètes (`5 120` colonnes) d'extraits `BF16` de `layers.0.linear_attn.in_proj_qkv` d'un contraste marqué de kurtosis entre `Q` (`3,23`), `K` (`3,14`) et `V` (`11,01`), confirmant qu'un tenseur fusionné unique masque des régimes statistiques distincts.
  3. Identification dans `modeling_qwen3_5.py` du fait que le module `mtp.*` n'est pas appelé par `Qwen3_5ForConditionalGeneration.forward` : sans perte auxiliaire dédiée, `mtp.*` reçoit une courbure $H_O = 0$ et un gradient nul en Passes 1, 2 et 3.

### 1.2 Distinction des cinq causes possibles de perte (sans présupposer une dérive récurrente)

Les prépublications de septembre 2026 étudient deux perturbations distinctes du régime Lumen : `Minima` (`arXiv:2609.04098v1`, `publié`) montre en `NVFP4 W4A4` (`17,53 Gio`, `~4,5 bpw`) que l'erreur d'état des 48 couches *Gated DeltaNet* plafonne rapidement grâce à l'écrasement le long de $k_t$, tandis que `STEPQuant` (`arXiv:2609.38169v1`, `publié`) compresse l'état persistant $S_t$ en mémoire vive (économie nulle sur le fichier de poids). Avant d'engager des ressources lourdes, il faut donc séparer expérimentalement, en maintenant d'abord les états GDN en `FP32` et le cache KV en `BF16` :
1. **L'erreur locale de projection et de porte multiplicative** ($\Delta W_\ell X_{\text{fp}}$ et interaction bilinéaire `gate` $\times$ `up` dans SwiGLU), isolée par greffe d'une couche ou d'un MLP (`REQ-A2B-01`).
2. **La dérive cumulative inter-couches des activations** ($\Delta X_\ell = X_{\text{fp},\ell} - X_{q,\ell}$), reconnaissable à une croissance avec la profondeur $\ell$ dès les `1 024` premiers tokens mais plate selon la position temporelle $t$ (`REQ-A2B-02`).
3. **Une éventuelle dérive positionnelle ou d'exécution** (croissance de la KL entre `0–1k` et `3k–4k+` à préfixe identique, ou écart numérique entre kernel fusionné et référence FP32).
4. **Le décalage de préfixe en génération libre** (*exposure bias* lorsque l'élève s'écarte de la trajectoire du professeur sur de longues chaînes `<think>` ou séquences d'outils XML).
5. **Le défaut de couverture domaine/vocabulaire** (les 4 livres ne couvrant que `2,83 %` des `248 320` entrées de `embed_tokens`/`lm_head` et `0 %` de la Vision/Vidéo).

---

## 2. Réparation du Quantifieur $E_8$ et Explication du Scaling d'Entrée (`EXP-V2-01`)

Dans une première implémentation jouet, une pénalité de débit codée à la main attribuait plus de bits aux `128` vecteurs demi-entiers $(\pm \frac{1}{2})^8$ de $D_8 + \frac{1}{2}$ qu'aux `112` vecteurs $(\pm 1, \pm 1, 0^6)$ de $D_8$ sur la même coquille sphérique $m = 1$ ($\|z\|_2^2 = 2$), brisant l'isotropie du réseau de Gosset $E_8$ et faisant perdre $E_8$ contre le scalaire.

En rétablissant la **vraie longueur de code sphérique** $R(z) = -\log_2 \widehat{P}_{\text{shell}}(m) + \log_2 |\mathcal{S}_m|$ (`|\mathcal{S}_0|=1, |\mathcal{S}_1|=240, |\mathcal{S}_2|=2160, \dots`) mise à jour par EM dans `budget_lumen_max.py`, et en testant hors-échantillon à **`2,500 bpw` encodé égal** sur `72` vraies lignes (`24 Q + 24 K + 24 V`, sous-blocs de `128` colonnes) de `layers.0.linear_attn.in_proj_qkv` alimentées par de vrais embeddings `RMSNorm_0(embed_tokens)` pondérés selon une loi de Zipf (`N_cal = 2048`, `N_val = 1024` indépendants, `mesuré/reproduit par toi`) :
- `RTN` scalaire (`2,5003 bpw`) : `-16,12 dB` ;
- **`S0`** scalaire équitable (`Hadamard + LDLQ + RDOQ EM`, `2,4989 bpw`) : **`-17,79 dB`** ;
- **`L0` (`SU-E8-rANS` + `Hadamard + Block-8 LDLQ + RDOQ EM`, `2,5010 bpw`) : `-18,50 dB`**, soit un gain de **`+0,71 dB` sur `S0`** (concordant avec les `+0,7` à `+0,8 dB` rapportés à l'Étape 1 sur `5 120` colonnes) ;
- **`L0 + ABCD-E8` (2 passes Gauss-Seidel bidirectionnelles post-LDLQ, `2,4994 bpw`) : `-18,58 dB`** (`+0,08 dB` additionnel, `+0,79 dB` sur `S0`) ;
- **`L0` + scaling d'entrée $d_{\text{in}} = \sqrt{\mathrm{diag}(H_{\text{cal}})}$ (`2,5007 bpw`, avec changement de base exact $H'$) : `-17,20 dB`**, soit une **dégradation de `1,30 dB` face à `L0`** (proche de la fourchette de `1,5` à `2,4 dB` rapportée à l'Étape 1), car multiplier $W$ par $D_{\text{in}}$ transfère la variance des canaux chauds d'activation dans l'énergie des poids que $E_8$ doit coder sous débit fixe, alors que Hadamard seul ($D_{\text{in}} = I$) égalise déjà la diagonale de $U H U^\top$ sans changer $\det(H)$.

---

## 3. Les Trois Propositions Finalistes Approfondies (Fiches Implémentables en 9 Points)

---

### Fiche Finaliste 1 — `KA-BGCC-RDOQ` : Arrondi $E_8$ par Onde Frontale de Kronecker, Cible Asymétrique et Couplage Bilinéaire SwiGLU

1. **Limite identifiée dans Lumen `Q0` et éléments de soutien :**  
   Dans `Q0`, chaque matrice linéaire est arrondie ligne par ligne avec un Fisher purement diagonal côté sortie, et les paires multiplicatives (`gate_proj`, `up_proj`) des 64 MLP SwiGLU sont quantifiées indépendamment. Or : (a) un Fisher diagonal ne propage aucune compensation entre lignes corrélées dans l'espace de sortie (`YAQA`, `arXiv:2505.22988v3`, `publié`) ; (b) si la cible reste symétrique $\|\widehat{W}X_q - WX_q\|^2$, la dérive amont $\Delta X = X_{\text{fp}} - X_q$ s'accumule (`GPTAQ`, `arXiv:2504.02692v3`, `publié`) ; (c) dans $h = \mathrm{SiLU}(W_g x) \odot (W_u x)$, une erreur déjà figée sur $\mathrm{SiLU}(\widehat{W}_g x)$ perturbe multiplicativement le produit final si $\widehat{W}_u$ l'ignore (`EXP-V2-02`, `mesuré/reproduit par toi`).
2. **Mécanisme proposé et différence avec `Q0` :**  
   Sans modifier le format binaire ni ajouter un seul poids ou FLOP à l'inférence, `KA-BGCC-RDOQ` combine trois opérations lors de la Passe 2 :
   - **Cible asymétrique complète** $W_{\text{eff}}' = W' + W'\Delta C_X' (H_{q,q}')^{-1}$ compensant la dérive des couches précédentes ;
   - **Couplage Bilinéaire `BGCC` sur SwiGLU :** quantifier d'abord `gate_proj` en $\widehat{W}_g$, puis ajuster la cible de chaque ligne $i$ de `up_proj` en résolvant le problème des moindres carrés pondéré par les portes quantifiées réelles $g_{q,i,t} = \mathrm{SiLU}(\widehat{w}_{g,i} x_{q,t})$ ;
   - **Onde frontale anti-diagonale de Kronecker** sur $(L_O, L_I)$ par blocs de $b_r = \min(128, m - 128 r)$ lignes (couvrant les dimensions non multiples de `128` comme `vis_mlp_fc1` = `4 304`), suivie optionnellement d'une passe Gauss-Seidel `ABCD-E8`.
3. **Formulation mathématique et conditions de validité :**  
   - *Identités démontrées sous hypothèses :* la sonde $u_t = \mathrm{diag}(\sqrt{p_t})\zeta_t - p_t(\sqrt{p_t}^\top\zeta_t)$ ($\zeta_t \in \{\pm 1\}^{|\mathcal{V}|}$) est un estimateur sans biais de $\nabla_z^2 D_{\mathrm{KL}}(p_{\theta^*}\|p_\theta)\big|_{\theta^*}$ (contrairement au gradient nul de $D_{\mathrm{KL}}(p_{\theta^*}\|p_{\theta^*})$ et au carré biaisé du gradient moyen), et l'onde frontale de `YAQA` converge en exactement $b_r + n/8 - 1$ pas parallèles. En revanche, l'optimalité conjointe avec la sélection discrète non convexe RDOQ sur $E_8$ est une propriété heuristique dont le gain net sur 27B est **`à tester`**.
   - *Condition critique d'échantillonnage établie en `EXP-V2-02` :* `BGCC` et `GPTAQ` exigent un ratio d'échantillonnage **$N_{\text{cal}} / n \ge 10$** (soit $\ge 180\,000$ tokens pour `down_proj` où $n = 17\,408$) et un mélange régularisé $0,7 H_{g_q^2} + 0,3 \bar{g}_{q,i}^2 H_{q,q}$ ; en régime sous-échantillonné ($N_{\text{cal}} / n \approx 1,25$), nous avons mesuré que `BGCC` sur-ajuste la calibration (`-0,68 dB` en validation), alors qu'à $N_{\text{cal}} / n = 16$ il apporte **`+0,40 dB` (couche 0) et `+0,61 dB` (couche 31)**.
4. **Statistiques, données et ressources nécessaires :**  
   - Collecte de $H_O$ (blocs $\le 128 \times 128$ en `FP32` sur les `520` matrices 2D, avec branche dédiée `MTP`) : `8,91 Mo` pour une matrice `[17408, 5120]`, `127,14 Mo` pour `lm_head`, soit **`2,38 Go` sur disque** (`1,20 Go` en triangles supérieurs empaquetés).
   - En Passe 2, un seul bloc Transformer réside en VRAM (`< 18 Go`).
5. **Spécification algorithmique (sans erreur d'indice ni suppression de diagonale) :**
   ```python
   def ka_bgcc_quantize_mlp_up_and_wavefront(W_up, W_gate_orig, W_gate_hat, X_fp, X_q, H_O_blocks, U_had, alpha, lam):
       # 1. Cible couplée BGCC par ligne i (activée uniquement si X_q.shape[1] >= 10 * W_up.shape[1])
       G_fp = F.silu(W_gate_orig @ X_fp)                                  # [m, N_cal]
       G_q  = F.silu(W_gate_hat  @ X_q)                                   # [m, N_cal]
       U_fp = W_up @ X_fp
       H_qq_reg = X_q @ X_q.T + 0.01 * (X_q.pow(2).sum() / X_q.shape[0]) * torch.eye(X_q.shape[0], device=X_q.device)
       W_target = compute_bgcc_regularized_target(W_up, G_fp, G_q, U_fp, X_q, H_qq_reg, mix=0.7)
       # (Pour une couche non-SwiGLU, cible GPTAQ complète incluant la diagonale : W + W @ ((X_fp - X_q) @ X_q.T) @ inv(H_qq_reg))

       # 2. Passage en base de Hadamard et décomposition causale par blocs 8x8
       W_t_p = apply_hadamard_cols(W_target, U_had)
       H_p   = transform_gram_hadamard(H_qq_reg, U_had)
       L_I, U_s_inv_blks, M_blks8 = block8_upper_ldl(H_p)                 # L_I triangulaire supérieure unité

       # 3. Onde frontale anti-diagonale par tranche de br = min(128, m - r0) lignes (gère 4304 % 128 == 80)
       W_hat_p = torch.zeros_like(W_t_p)
       for r0 in range(0, W_t_p.shape[0], 128):
           br = min(128, W_t_p.shape[0] - r0)
           L_O, D_O = unit_upper_ldl(H_O_blocks[r0 // 128])               # [br, br]
           W_hat_p[r0:r0+br] = run_causal_wavefront_su_e8(
               W_t_p[r0:r0+br], L_O, D_O, L_I, U_s_inv_blks, M_blks8, alpha[r0:r0+br], lam
           )
       return inverse_hadamard_cols(W_hat_p, U_had)
   ```
6. **Bilan d'octets et coûts :**  
   - **Coût permanent dans le package et à l'inférence :** **`0 octet`, `0 FLOP`**.
   - **Coût temporaire :** `~2,38 Go` disque pour $H_O$ + surcoût de calcul de Passe 2 d'environ `+45 min` sur `1× H100` pour les matrices `up_proj` et l'onde frontale.
7. **Contrôle fort et plus petite expérience informative :**  
   - Contrôles à octets encodés strictement égaux : `L0` $\to$ `L0 + YAQA seul` $\to$ `L0 + GPTAQ seul` $\to$ `L0 + BGCC seul` $\to$ `KA-BGCC-RDOQ` sur `layers 0..7` (`REQ-A2B-01/02`).
8. **Critères de poursuite, réfutation ou repli :**  
   - Conserver chaque sous-brique (`YAQA`, `GPTAQ`, `BGCC`, `ABCD-E8`) uniquement si elle améliore individuellement la KL sur `select_dev` au-delà de l'erreur-type bootstrap inter-documents ; désactiver individuellement toute sous-brique neutre.
9. **Risques par domaine :**  
   - Sur `Vision` (`vis_mlp_fc1/fc2`, activation `gelu_pytorch_tanh` et non `SiLU`), adapter la dérivée exacte de `GELU` dans `BGCC` et vérifier l'absence de régression sur `OmniDocBench` / `CharXiv`.

---

### Fiche Finaliste 2 — `SU-EB-RCO` : Codage Sphérique Exact `SU-E8-rANS`, Format Sobre `84,07 Mo` et Allocation Multi-Budgets

1. **Limite identifiée et éléments de soutien :**  
   Un format lourd (`413 Mo` d'auxiliaires + `62 Mo` d'inefficacité rANS) combiné à `258 Mo` non alloués retire plus de `600 Mo` au cœur texte, plaçant le modèle en infériorité numérique face à `GSQ-RCO IQ2_S` (`9,26 Go` texte seul). De plus, sans branche dédiée, une allocation globale type `RCO` écrase le module `MTP` dont le gradient principal est nul.
2. **Mécanisme proposé :**  
   - **Codeur Sphérique Exact `SU-E8-rANS` sans table de vecteurs :** décompose $u = 2z \in 2E_8$ en sa coquille $m = \|u\|_2^2 / 8 \in \{0..126\} \cup \{\text{ESC}\}$ (codée par une table `rANS` 14-bit de `256` octets, soit **`3,50 Kio`** pour `14` profils sur tout le modèle) et son rang combinatoire uniforme $r_m \in \{0,\dots,|\mathcal{S}_m|-1\}$ sur $|\mathcal{S}_m| = 240\,\sigma_3(m)$ (désindexé algébriquement dans les registres ALU sans table RAM, `0` erreur vérifiée sur `314 688` vecteurs Monte-Carlo, redondance mesurée **`+0,0011 à +0,0025 bpw`**).
   - **Format sobre `LUMEN-v2-SG512-Adaptive` (`84 066 560` octets = `84,07 Mo` pour tous les auxiliaires)** : échelles `FP16` par ligne + `FP8 E4M3` par super-groupe de `512` poids (`63,89 Mo`) et super-tuiles adaptatives `262 144` / `131 072` poids à `32` lanes (`20,02 Mo`), saturant `100 %` des SMs GPU (`240` à `4 850` warps par opérateur).
   - **Allocation en deux étapes (`Water-Filling` $\to$ `EB-RCO-Slack` multi-budgets) :** initialisation lagrangienne fermée jusqu'à `9,9815 Go`, puis affinage conjoint sur la variété de `RCO` (`arXiv:2605.00649v3`) avec **variable d'écart $s \ge 0$ et contraintes disjointes protégeant `Vision` et `MTP`**, en gardant les $K=5$ variantes compressées (`~48 Go`) **en RAM hôte (`128 Go`)** pour supprimer tout goulet d'I/O disque.
3. **Formulation et conditions de validité :**  
   - Bijection combinatoire Coquille+Rang et géométrie de la variété multi-contraintes de `RCO` : **`démontrées sous hypothèses`**.
   - Compatibilité avec le décodeur existant du projet : **`à tester`** lors de l'inspection du code `LUMEN-v2` (si le décodeur actuel est déjà figé avec un autre agencement, seule l'allocation `EB-RCO-Slack` sur les coûts encodés réels de ce décodeur est conservée).
4. **Ressources nécessaires :**  
   - `1× GPU 80 Go` + **`128 Go` de RAM système** (hébergeant les `~48 Go` de variantes compressées `SU-E8-rANS` en mémoire vive pendant les `300` pas de RCO) ; pli `select_dev` (`1,05M` tokens).
5. **Spécification de l'objectif multi-budgets `EB-RCO-Slack` (incluant la branche `MTP`) :**
   ```python
   # Objectif conjoint évalué sur select_dev (évite l'effondrement à 0 du module MTP !)
   L_joint(alpha) = KL_main(p_ref_t, p_student_t(alpha)) + 0.15 * KL_mtp(p_ref_t1, p_mtp_t1(alpha))

   # Contraintes simultanées sur la variété RCO (§A.7 & §A.8 de Helcig & Alistarh 2026) :
   #   1) C_text(alpha) + C_vis(alpha) + C_mtp(alpha) + s_global == 9_985_000_000 - C_protected_fixed
   #   2) C_vis(alpha) - s_vis_surplus == 195_000_000   (plancher multimodal garanti)
   #   3) C_mtp(alpha) - s_mtp_surplus == 135_000_000   (plancher d'acceptation spéculative garanti)
   # Suivi d'une programmation dynamique entière avec coûts arrondis par excès ceil(c_{g,k} / 16384).
   ```
6. **Bilan d'octets vérifié (`budget_lumen_max.py`) :**  
   - `FORMIC-Max` atteint **`9 981 480 462` octets** (`9,9815 Go`, garde technique `18,52 Mo`), dont **`9 826 546 532` octets de pur payload d'entropie** (`+623,38 Mo` de précision gagnée sur les poids par rapport à notre v1), allouant **`9,6329 Go` (`2,8652 bpw` effectif) au Tronc Texte**, `202,55 Mo` (`3,5171 bpw` effectif) à `Vision` et `146,00 Mo` (`2,7501 bpw` effectif) au `MTP` propre.
7. **Contrôle fort et test minimal :**  
   - Comparer à taille encodée strictement égale (`±1 Mo`) : (a) `Water-Filling analytique` vs `EB-RCO-Slack`, et (b) `SG-512` (`84,07 Mo` aux.) vs `g=128` (`+162,5 Mo` aux., donc `-0,048 bpw` sur les poids) sur 2 macro-blocs.
8. **Critères de décision :**  
   - Si `SG-512` perd plus sur certaines projections sensibles (`attn_k/v`, `lm_head`) que ne rapportent les `+0,048 bpw` économisés, activer le mode hybride (`g=128` uniquement sur `attn_k/v` et `lm_head`, coût marginal `+16,8 Mo`). Si `EB-RCO-Slack` gagne `< 2 %` de KL relative sur le Water-Filling initialisé avec les traces de Kronecker, garder le Water-Filling direct.
9. **Risques par domaine :**  
   - Décodage d'un seul token dans `embed_tokens` : utiliser sur `embed_tokens` des tuiles `32 × 5120 = 163 840` poids (`1` ligne par lane, surcoût `+0,49 Mo`) pour éviter de décoder `256` lignes lors d'un lookup unitaire.

---

### Fiche Finaliste 3 — `EAR-PV` : Récupération Continue et Voisinage Kissing-$E_8$ sous Contrôle de Décodage et Rollback

1. **Limite identifiée :**  
   En Passe 3, même une petite dérive d'histogramme lors de la mise à jour des codes discrets (`+0,0057 bpw` = `~+19,2 Mo` mesurés sur un seul pas de 20 % des blocs en `EXP-V2-04`) suffit à franchir la garde serrée de `18,52 Mo` sous `10 000 000 000` octets ; en outre, modifier simultanément des millions de codes sous une approximation diagonale peut dégrader la KL globale sans mécanisme de rejet, et ajuster les échelles continues sur un corpus étroit peut dégrader les domaines sous-représentés.
2. **Mécanisme proposé :**  
   - **Duel préalable des données (`Hors-politique` vs `50 % GKD`) à temps GPU égal** avant d'engager toute la Passe 3b.
   - **Étape P continue régularisée (`Main + MTP`)** : optimisation AdamW des échelles (`FP16`/`FP8` maîtres) et `RMSNorm` sur la perte conjointe `KL Top-128 + log-masse restante` (incluant la branche `MTP` et une pénalité proximale $\lambda_{\text{prox}}\|\phi - \phi_{\text{P2}}\|_2^2$ protégeant les connaissances rares hors-calibration).
   - **Étape V discrète optionnelle avec validation après décodage réel et rollback :** recherche locale sur `{0} ∪ (λ + Kissing_240(E8))` (motivée heuristiquement par l'observation scalaire de `GSQ` §L.3, mais vérifiée directement sur $E_8$) avec pénalité $-\mu_{\text{dual}}\log_2 \widehat{P}_{\text{shell}}(m)$, mise à jour EM des tables `SU-E8-rANS`, export vers un fichier temporaire portable (`tempfile`), rechargement par le décodeur réel et **rollback immédiat à l'état post-Étape P si `size > 10^10` ou si la KL sur `select_dev` n'a pas baissé**.
3. **Formulation et conditions de validité :**  
   La non-régression par rapport à l'Étape P sur `select_dev` et le respect de `size <= 10 000 000 000` octets reposent non sur une hypothèse de convexité asymptotique, mais sur le **test explicite d'acceptation/rollback après décodage réel** (`démontré par construction sur select_dev` ; généralisation à `confirmation_holdout` **`à tester`**).
4. **Données et ressources exactes (corrigées) :**  
   - Cache NVMe enseignant `Top-128` : `128 × 2` o (`FP16`) + **`128 × 4` o (`int32` requis pour `vocab_size = 248 320`)** + `4` o (`tail FP32`) = `772` o/token, soit **`3,238 Go` pour `4,19M` tokens** (`+0,82 Go` pour le cache `Top-32` du `MTP` = **`4,06 Go` totaux**, ou `~2,9 Go` avec empaquetage 18 bits). Pic VRAM élève sur `1× 80 Go` : `~38–44 Go` (professeur BF16 absent de la VRAM en mode hors-politique).
5. **Pseudocode avec portabilité OS et rollback strict :**
   ```python
   def ear_pv_safe_recovery(student, calib_loader, select_dev_loader, max_bytes=10_000_000_000):
       # 1. Étape P continue (codes E8 figés, perte conjointe Main + 0.15 * MTP + régul. proximale)
       train_continuous_scales_and_norms(student, calib_loader, steps=800)
       with tempfile.TemporaryDirectory() as tmpdir:
           p_path = Path(tmpdir) / "ckpt_post_p.lumen"
           p_bytes = export_and_reload_cold(student, p_path)
           assert p_bytes <= max_bytes
           kl_post_p = eval_exact_kl_on_split(student, select_dev_loader)
           snapshot_p = student.clone_discrete_and_tables()

           # 2. Étape V Kissing-240 + EM SU-E8-rANS (évaluée uniquement si elle bat kl_post_p après décodage !)
           G_w, H_diag = accumulate_weight_grad_stats(student, calib_loader)
           propose_dual_kissing240_step_(student, G_w, H_diag, top_frac=0.02, max_bytes=max_bytes)
           v_path = Path(tmpdir) / "ckpt_post_v.lumen"
           v_bytes = export_and_reload_cold(student, v_path)
           kl_post_v = eval_exact_kl_on_split(student, select_dev_loader)

           if v_bytes > max_bytes or kl_post_v >= kl_post_p:
               student.restore_discrete_and_tables(snapshot_p)  # Rollback garanti
       return student
   ```
6. **Bilan d'octets :** **`0 octet`** permanent ajouté ; conformité $\le 10\,000\,000\,000$ octets vérifiée sur les octets réellement décodés.
7. **Contrôle fort et test discriminant :**  
   - Duel 1 (`2 GPU-h` vs `2 GPU-h`) : `P-step hors-politique` vs `P-step 50 % GKD` sur préfixes de l'élève évalués par le professeur sur le même historique.
   - Duel 2 : `P-step seul` vs `P-step + V-step Kissing-240` (notre `EXP-V2-04` montre qu'à échelle fixe sur un seul pas jouet, le gain pur de V-step est modeste à `-0,37 %` MSE : si V-step apporte `< 1 %` de gain KL réel après P-step sur le 27B, il sera désactivé pour simplifier le pipeline).
8. **Critères de décision :**  
   - Ne retenir `GKD` que s'il améliore les tâches de génération multi-étapes sur `select_dev` à temps GPU total identique ; ne monter de `4k` vers `16k` tokens (avec états GDN pré-chauffés sur l'historique réel) que si `REQ-A2B-02` montre une croissance positionnelle de l'erreur.
9. **Risques par domaine :**  
   - Sur `Outils` (D7, format XML `<tool_call><function=...>`), interdire toute trajectoire `GKD` sans exécution sandbox cohérente ; sur `Connaissances` (D4), surveiller une sentinelle factuelle hors-calibration à chaque cycle P pour prévenir tout sur-ajustement des échelles aux domaines fréquents.

---

## 4. Comparateurs Externes, Séparation des Harnais BF16 et Scénarios Horaires Physiquement Réalistes

### 4.1 Comparaison externe complète et incompatibilité des trois harnais BF16

**Avertissement critique sur les références BF16 :** trois jeux de scores BF16 distincts existent pour `Qwen/Qwen3.8-27B` et **ne doivent jamais être mélangés sur une même courbe** :
1. **Carte officielle `Qwen/Qwen3.8-27B` (`37561816...`, `publié`) :** `GPQA-Diamond = 89,2`, `LiveCodeBench v6 = 90,3` (mode Thinking `temp=1.0, top_p=0.95, top_k=20`, budget jusqu'à `262k`).
2. **Graphique historique Unsloth (`rapporté par le projet` dans `ANALYSE_ET_PRIORITES_LUMEN_MAX_V2.md`) :** `LiveCodeBench = 88,62`.
3. **Harnais GGUF `llama.cpp` d'`ISTA-DASLab` (`d562806d...`, `publié`) :** `AIME25 = 100,00`, `GPQA-Diamond = 89,90`, `LiveCodeBench v6 = 85,71`.

| Modèle / Checkpoint sur `Qwen/Qwen3.8-27B` | Texte seul (Go) | Complet Vision+MTP (octets / Go) | Sous `10,00 Go` complets ? | Résultats publiés sur le harnais propre de chaque auteur (non comparables entre lignes d'auteurs différents) |
|---|---:|---:|:-:|---|
| **`ISTA-DASLab GSQ-RCO IQ3_S`** *(repère 11,8 Go)* | `11,7715 Go` | `13 051 163 488` (`13,05 Go`) | Non | Harnais DASLab : `AIME25 100,0`, `LCB6 85,71`, **`GPQA-D 89,39` (`-0,51 pt` vs `89,90` = régression Formic §4 ; `1/198` q., $p_{\text{McNemar}}=1,0$)** |
| **`orcarouter/OrcaSAQ-2-27B` (EXL3)** | *(12,27 Go av. MTP)* | `13 191 895 596` (`13,19 Go` éq.) | Non | Sans Vision ; `93,2 %` concordance Top-1, `KLD = 0,031` (`WikiText-2`) |
| **`EschaLabs/Qwen3.8-27B-Escha-W2` (QAT)** | `10,1531 Go` | `11 923 948 824` (`11,92 Go` éq.) | Non | Sans Vision ; QAT lourd ; harnais SGLang : `GPQA-D = 88,38`, `LCB6 = 86,81` |
| **`ISTA-DASLab GSQ-RCO IQ3_XXS`** | `10,0944 Go` | `11 373 974 368` (`11,37 Go`) | Non | Harnais DASLab : `AIME25 100,0`, `GPQA-D 88,89` (`-1,01`), `LCB6 84,57` (`-1,14`) |
| **`Unsloth Dynamic UD-Q2_K_XL`** | `~9,80 Go` | `~11,08 Go` éq. | Non | Harnais DASLab : `AIME25 100,0`, `GPQA-D 86,87` (`-3,03`), `LCB6 82,28` (`-3,43`) |
| **`ISTA-DASLab GSQ-RCO IQ2_S`** *(+`-mtp` `Q6_K` +`mmproj`)* | `9,2595 Go` | `10 539 127 648` (`10,54 Go`) | Non | Harnais DASLab : `AIME25 100,0`, **`GPQA-D 86,36` (`-3,54`)**, **`LCB6 82,29` (`-3,42`)** |
| **`ISTA-DASLab GSQ-RCO IQ2_XS`** *(+`-mtp` `Q6_K` +`mmproj`)* | `8,4228 Go` | **`9 702 458 208` (`9,70 Go`)** | **Oui** | Harnais DASLab : `AIME25 96,67` (`-3,33`), **`GPQA-D 84,85` (`-5,05`)**, **`LCB6 76,57` (`-9,14`)** |
| **`LUMEN-Max Q1-v2` (Proposition Water-Filling)** | **`9,6329 Go`** *(est.)* | **`9 981 480 462` (`9,98 Go` est.)** | **Oui** | **`À tester`** sur harnais identique contre `GSQ-RCO IQ2_XS` (comparaison produit complet `< 10 Go`) et `IQ2_S` (comparaison algorithmique cœur texte) |

*(Note sur la structure `-mtp.gguf` de DASLab : l'inspection de `Qwen3.8-27B-GSQ-RCO-IQ2_S-mtp.rco-allocation.txt` montre que DASLab fige les 8 matrices 2D du MTP en `Q6_K` [`348,47 Mo`] et laisse `mmproj` en `BF16` [`931,15 Mo`], soit `1,280 Go` hors texte. En ramenant `Vision + MTP` à `348,55 Mo` via `SU-E8-rANS`, `LUMEN-Max Q1-v2` alloue **`+1,210 Go` de plus au cœur texte que `IQ2_XS`** [`9,633 Go` contre `8,423 Go`] et **`+0,373 Go` de plus que `IQ2_S`** [`9,633 Go` contre `9,260 Go`].)*

### 4.2 Scénarios horaires GPU transparents et physiquement cohérents (`1× NVIDIA H100 SXM5 80 Go`)

Deux copies `BF16` du modèle (`111,13 Go`) ne tenant pas sur `80 Go`, toutes les passes opèrent par streaming de macro-blocs ou avec un seul modèle en VRAM. Sur **`1× H100 SXM5 80 Go`** (pic théorique dense `BF16` = **`989,4 TFLOP/s`**, et non `1 979 TFLOP/s` qui suppose la sparsité 2:4) avec **`128 Go` RAM CPU** et **`≥ 450 Go` SSD NVMe local (`~2,5 Go/s`)**, un token traverse `~25,6B` paramètres actifs (`~51,3 GFLOPs` en forward pur, `~205–256 GFLOPs` en backward avec *gradient checkpointing* et déquantification en vol). En retenant une efficacité réaliste soutenue de `~58–62 %` du pic dense (`~11 500 tok/s` en forward streaming BF16 et `~2 800 à 3 500 tok/s` en backward checkpointé avec décompression), la décomposition $T_{\text{total}} = T_{\text{collecte}} + T_{\text{professeur}} + T_{\text{apprentissage}} + T_{\text{codage}} + T_{\text{évaluation}} + T_{\text{IO}}$ donne (`estimé`, hors enveloppe du pod de B) :

| Scénario de campagne (sur `1× H100 SXM5 80 Go` + `128 Go` RAM) | $T_{\text{collecte}}$ *($H_I$ + 4 sondes $H_O$ Main & MTP)* | $T_{\text{professeur}}$ *(Cache Top-128 `4,06 Go` + $\Delta C_X$)* | $T_{\text{codage}}$ *($K=5$ variantes RCO + passe finale + P4)* | $T_{\text{apprentissage}}$ *(`EB-RCO` en RAM + Passe 3 `EAR-PV`)* | $T_{\text{évaluation}}$ *(Rechargement à froid + KL + dev)* | $T_{\text{IO}}$ *(Shards, NVMe, exports)* | **$T_{\text{total}}$ estimé** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S1 — Pilote discriminant minimal (`layers 0..7` + `lm_head` + `MTP`, `524k` tok)** | `0,45 h` | `0,25 h` | `0,55 h` | `0,70 h` | `0,50 h` | `0,20 h` | **`~2,65 h`** |
| **S2 — Campagne `FORMIC-Max Q1-v2` complète hors-politique (`4,19M` tok `4k`)** | `3,60 h` | `1,85 h` | `5,80 h` *(inclut $K=5$ variantes + resync)* | `8,50 h` *(`EB-RCO` 3,0h + `EAR-PV` 5,5h)* | `3,80 h` | `1,25 h` | **`~24,80 h`** |
| **S3 — Extension conditionnelle avec `50 % GKD` + fenêtres `16k` pré-chauffées** | `3,60 h` | `5,90 h` *(gén. élève + scoring prof.)* | `5,80 h` | `12,10 h` | `4,40 h` | `1,60 h` | **`~33,40 h`** |

*(Sur un GPU local de `4 Go` type RTX 2050, seuls les tests unitaires du codec `SU-E8-rANS` et l'arrondi d'une matrice isolée en streaming sont réalisables ; les passes multi-couches avec backward Rademacher requièrent le pilote `S1`.)*

---

## 5. Ordre d'Expérimentation, Protocole des 11 Domaines et Inconnues Décisives

### 5.1 Séquence expérimentale progressive et protocole de confirmation (Instructions Formic Sections 4 & 11)
1. **Jalon E0 (Audit du codec existant, `< 30 min`) :** mesurer sur une vraie couche de `Qwen/Qwen3.8-27B` l'écart entre le codec actuel du projet et `SU-E8-rANS` + `LUMEN-v2-SG512-Adaptive`.
2. **Jalon E1 (`REQ-A2B-01` & `REQ-A2B-02` ou Pilote `S1`, `~2,65 h` H100) :** vérifier le transfert de `SU-E8-rANS`, `KA-RDOQ` et `BGCC` sur les MLP SwiGLU et sur `layers 0..7`, et mesurer le profil positionnel `0–1k` vs `2k–4k` à états GDN `FP32` et KV `BF16` fixes.
3. **Jalon E2 (`REQ-A2B-03`, `~40 min` H100) :** valider la corrélation de rang du proxy `KL Top-128 + tail`, fixer le plancher de débit du `MTP` préservant $\alpha_{\text{MTP}}$, et trancher le duel à coût égal `QERA r=64` vs `+0,08 bpw` sur `lm_head`.
4. **Jalon E3 (Campagne `S2` sur `A0` fixe) :** produire `A0Q1` décodé à froid vérifiant `size_bytes <= 10_000_000_000`, et exécuter le duel de `2 GPU-h` `Hors-politique vs GKD`.
5. **Jalon E4 (Confirmation protégée sur les 11 domaines & Intégration `A0/A1 × Q0/Q1`) :**
   - Évaluation à gabarit `chat_template.jinja`, `reasoning_effort` (`xhigh`), `preserve_thinking` et budgets de jetons strictement identiques au `BF16` sur le **même moteur**, couvrant les **11 domaines** : (D1) Raisonnement scientifique (`GPQA-D` en multi-tirages appariés), (D2) Maths (`AIME`, `MATH-500`), (D3) Code (`LiveCodeBench v6`, `SWE-bench`), (D4) **Connaissances factuelles pures** (`SimpleQA`, `MMLU-Pro`), (D5) Multilingue (`MMMLU`, `MGSM`), (D6) Suivi de consignes (`IFEval`), (D7) **Outils XML `<tool_call><function=...>` & JSON** (`BFCL v3`), (D8) Agents multi-tours (`TAU-bench`) + taux d'acceptation $\alpha_{\text{MTP}}$, (D9) **Longs contextes `64k–262k` natifs et `1M` sous YaRN `factor=4.0`** (`RULER`, `LongBench v2`), (D10) **Multimodal Image & Vidéo** (`MMMU`, `OmniDocBench 1.5`, `CharXiv`, séquences vidéo M-RoPE 3D), et (D11) Robustesse/calibration (`KL moyenne`, **`KL P99`**, taux de flips Top-1 stratifié par entropie).
   - **Qualification statistique stricte (Section 4 Formic) :** utilisation conjointe du **test exact apparié de McNemar** sur les items binaires et du **bootstrap par documents/épisodes** ($10\,000$ réplications). Aucune compensation inter-domaines n'est admise, aucune baisse même $< 0,1\,\%$ n'est qualifiée d'absence de perte, et l'on distingue systématiquement *non-régression empirique sur le périmètre testé*, *manque de puissance statistique* et *régression détectée*.

### 5.2 Les cinq inconnues critiques susceptibles de modifier nos choix

1. **Écart réel entre `SG-512` (`84,07 Mo` d'auxiliaires) et `g=128` (`+162,5 Mo`) sur les couches d'attention et `lm_head` :** si `SG-512` pénalise localement `attn_k/v` ou `lm_head`, un schéma hybride (`g=128` sur ces seules matrices, `SG-512` sur MLP/GDN) ne coûte que `+16,8 Mo`.
2. **Transfert de `BGCC` en profondeur sous vraies activations cachées :** nos gains de `+0,40` à `+0,61 dB` en `EXP-V2-02` utilisent des entrées d'embeddings normalisés ; si les activations d'entrée du MLP de couche 31 (`post_attention_layernorm`) présentent des queues beaucoup plus lourdes réduisant la stabilité du Gram pondéré par $g_q^2$, le paramètre de régularisation convexe devra être relevé vers `0,5` ou `BGCC` restreint aux couches où `select_dev` valide le gain.
3. **Comportement temporel de *Gated DeltaNet* à `~2,69–2,89 bpw` au-delà de `32k` tokens (`262k–1M`) :** si `REQ-A2B-02` ou `RULER @ 262k` révèle une dérive lente absente à `4 bits` dans `Minima`, `EB-RCO-Slack` devra relever le plancher des tranches `K` et `V` des couches GDN à grand horizon $1/(1-\bar{\alpha})$.
4. **Sensibilité de la modalité Vidéo (`patch_embed` 3D temporel + 27 blocs ViT) à `3,52 bpw` effectifs :** si les benchmarks vidéo ou OCR dense accusent une baisse, porter toute la `Vision Tower` à `4,50 bpw` effectifs ne requiert que `+56 Mo` (`-0,017 bpw` sur le tronc).
5. **Interaction avec l'architecture candidate `A1` de l'Agent B :** si `A1` modifie le rang ou la largeur des MLP, le débit moyen disponible par paramètre restant sous `10,00 Go` augmentera et toutes les courbures $(H_I, H_O)$ ainsi que le couplage `BGCC` devront être recalculés sur `A1` avant de trancher entre `A0Q1`, `A1Q0` et `A1Q1`.
