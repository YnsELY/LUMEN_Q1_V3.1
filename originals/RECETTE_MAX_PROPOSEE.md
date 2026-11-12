# Spécification Différentielle de la Recette Candidate `LUMEN-Max Q1-v2` (Passes 0 à 4)

**Identifiant immuable de la proposition :** `Q1 = LUMEN-Max-Q1-v2.0` (12 novembre 2026 — Chantier A, révision post-audit)  
**Recette de référence préservée :** `Q0 = LUMEN-v2-ref` (telle que décrite dans `ANALYSE_ET_PRIORITES_LUMEN_MAX_V2.md` ; `FORMIC_RECETTE_V2.md` étant absent de cet espace de travail, `Q0` est intégralement préservée comme témoin de référence).  
**Architecture de référence :** `A0` (`Qwen/Qwen3.8-27B`, révision vérifiée `37561816e66b5b9153b1b64c10968c18c1e6f4a0`, structure intacte).  
**Statut global de `Q1-v2` :** **Proposition de recette spécifiée (`estimé` / `démontré sous hypothèses` / `à tester`) — distincte d'une réalisation déjà exécutée sur le modèle 27B complet.**

---

## 1. Tableau différentiel des décisions par composant (`Q0` → `Q1-v2`)

| Composant / Mécanisme | État connu dans `Q0` / Étape 1 (`rapporté par le projet`) | Décision dans `Q1-v2` (`LUMEN-Max-Q1-v2.0`) | Justification vérifiée et conditions d'activation |
|---|---|---|---|
| **Contrôles internes `L0` et `S0`** | `L0` (`E8+LDLQ+RDOQ`, `-27,1 dB`) et `S0` (`scalaire+LDLQ/Hadamard`, `-26,4/-26,3 dB`) sur 256 lignes aléatoires (`47 Q, 45 K, 164 V`) de `in_proj_qkv` et `in_proj_z` couche 0 (`2,5 bpw`). | **MAINTENUS INTACTS** comme contrôles internes obligatoires à chaque étape. | Reproduit sur vraies tranches BF16 de couche 0 (`EXP-V2-01` : `SU-E8-rANS` devance `S0` de `+0,71 dB` à `2,50 bpw`). |
| **Scaling d'entrée $D_{\text{in}}$ testé à l'Étape 1** | Défavorable (`-1,5` à `-2,4 dB` rapportés). | **DÉSACTIVÉ PAR DÉFAUT ($D_{\text{in}} = I_n$)** en Passes 0–2. | Sur vraies entrées `RMSNorm_0(embed)` (`EXP-V2-01`), $d_{\text{in}} = \sqrt{\mathrm{diag}(H)}$ dégrade `L0` de `1,30 dB` en transférant l'anisotropie d'activation vers l'énergie des poids. |
| **Transformation de Hadamard $U$** | Petit avantage local (`+0,3` à `+0,4 dB` rapportés). | **MAINTENUE COMME CONTRÔLE**, vérifiée par famille avant généralisation. | Particulièrement utile sur les sous-matrices à forte kurtosis (telle que la tranche `V` de `layers.0.in_proj_qkv` mesurée à `11,01` contre `3,23` sur `Q`). Les rotations globales relèvent de **B**. |
| **Prédiction `lm_head` $\leftarrow$ `embed_tokens`** | `0,000 bit` de gain rapporté. | **SUSPENDUE.** | Maintenue fermée en l'absence d'hypothèse non linéaire nouvelle. |
| **Protection Passe 0 des petits tenseurs** | Prévue dans `Q0`. | **CONSERVÉE EN `BF16` NATIF (`62,20 Mo` sur `679` tenseurs)**, avec cast `.float()` à l'exécution. | 100 % des tenseurs du checkpoint officiel sont en `BF16` ; éviter une promotion statique en `FP32` économise `1,41 Mo` de mantisse nulle tant que Passe 3 ne les modifie pas. |
| **Format binaire et codage entropique $E_8$** | Auxiliaires sobres (`~83 Mo`) et $E_8$ non tronqué + rANS prévus dans `Q0`. | **PRÉCISÉ par `LUMEN-v2-SG512-Adaptive` (`84,07 Mo` d'auxiliaires) + codeur sphérique `SU-E8-rANS` (`3,50 Kio` de tables).** | Préserve exactement l'équiprobabilité des points de $D_8$ et $D_8 + \frac{1}{2}$ sur chaque coquille sphérique $m = \|z\|_2^2/2$ (`redondance +0,0011 à +0,0025 bpw`), libérant `+623 Mo` de payload poids face à un format lourd. |
| **Métrique de sortie Passe 1 (`Fisher diagonal`)** | Diagonale de sortie prévue dans `Q0`. | **ÉTENDUE (sous ablation) en `Sketch-A Kronecker` bloc-diagonal ($B_O = 128$, avec bloc résiduel $m \bmod 128$) + branche dédiée `MTP`.** | Sans branche auxiliaire dédiée, les `15` tenseurs propres du `MTP` ont un gradient et un $H_O$ identiquement nuls lors du forward principal. |
| **Cible de calibration et couplage bilinéaire Passe 2** | Comportement symétrique vs asymétrique de `Q0` non vérifiable sans le code absent. | **PROPOSÉE en `KA-BGCC-RDOQ`** : onde frontale Kronecker + cible asymétrique `GPTAQ` + couplage bilinéaire `BGCC` (`gate_proj` $\to$ `up_proj` SwiGLU) + polissage optionnel `ABCD-E8`. | Sur vraies paires `(gate_proj, up_proj)` des couches 0 et 31 (`EXP-V2-02`, $N_{\text{cal}}/n = 16$), `BGCC` réduit l'erreur MSE post-SwiGLU hors-échantillon de `8,7 %` à `13,3 %` (`+0,40` à `+0,61 dB`) sans surcoût d'octets. |
| **Allocation des octets sous `10 000 000 000` o** | Formule analytique de référence dans `Q0`. | **INITIALISÉE PAR WATER-FILLING LAGRANGIEN (`9,9815 Go` calculés, garde `18,52 Mo`)**, puis **AFFINÉE PAR `EB-RCO-Slack`** avec planchers par module. | Utilise utilement le plafond (`9,633 Go` pour le cœur texte, `+373 Mo` au-dessus de `GSQ-RCO IQ2_S` texte seul) tout en protégeant Vision et MTP par budgets disjoints. |
| **Optimisation Passe 3 et données de récupération** | Réglage bloc (3a), continu + codes et distillation prévus dans `Q0`. | **ENCADRÉE PAR `EAR-PV` AVEC ROLLBACK MONOTONE** + duel préalable à temps GPU égal entre hors-politique et `GKD`. | Évite toute dérive non contrôlée de taille rANS lors des mises à jour dans les `240` voisins de Kissing de $E_8$, et teste la longueur `4k` avant toute extension à `16k`. |
| **Correcteurs de faible rang (`QERA`)** | Optionnels dans `Q0`. | **DÉSACTIVÉS SUR LE TRONC**, testés uniquement en option sur `lm_head` (`r=64 INT8`, `~16,2 Mo` = `+0,0128 bpw` à cette dimension). | Sur le tronc, les octets sont d'abord investis dans la grille `SU-E8-rANS` ; sur `lm_head` `[248320, 5120]`, le ratio dimensionnel rend le coût unitaire très faible (`REQ-A2B-03`). |

---

## 2. Spécification technique passe par passe (`Q1-v2`)

### Passe 0 — Conventions algébriques, intégrité des types et garde-fous d'exécution
1. **Conventions de transformation en vecteurs colonnes (vérifiées par `test_4_change_of_basis_exactness`) :**
   Avec $W' = D_{\text{out}}^{-1} W D_{\text{in}} U^\top$, $x' = U D_{\text{in}}^{-1} x$ et $y = D_{\text{out}} W' x' = W x$ ($U^\top U = I_n$), les matrices de courbure se transforment selon :
   $$H' = U D_{\text{in}}^{-1} H D_{\text{in}}^{-1} U^\top, \qquad G' = D_{\text{out}}^\top G D_{\text{out}}.$$
   Dans `Q1-v2`, $D_{\text{in}} = I_n$ par défaut et l'amortissement relatif $\lambda_H = \eta_H \frac{\operatorname{tr}(H)}{n} I_n$ ($\eta_H = 10^{-2}$) est appliqué avant toute transformation.
2. **Paramètres protégés en `BF16` natif (`679` tenseurs, `31 097 584` paramètres = `62 195 200` octets alignés 128B dans `Max`) :**
   `48 × linear_attn.{in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm}`, `32 × self_attn.{q_norm, k_norm}`, `129 × RMSNorm` du tronc, `223` petits tenseurs/biais/normes de la Vision Tower et `7` normes du MTP restent dans leur `BF16` d'origine (reconvertis en `FP32` dans les registres du kernel pour `exp(A_log)` et les normes). Ils ne sont promus en `FP32` sur disque (`+1,41 Mo`) que si la Passe 3 met effectivement à jour leurs poids continus.
3. **Séparation stricte des échelles par projection (`Minima` §6) :** aucune fusion GEMM à échelle globale partagée n'est autorisée entre `in_proj_qkv`, `in_proj_z`, `in_proj_a` et `in_proj_b`.

---

### Passe 1 — Collecte multi-domaines des courbures $(H_I, H_O)$ avec branche auxiliaire `MTP`
1. **Couverture des 11 domaines dans `calib_train` (`4 194 304` tokens = `2 048 × 2 048`) :**
   - Séparation stricte par documents, dépôts GitHub, épisodes et sources visuelles vis-à-vis de `select_dev` et `confirmation_holdout`.
   - Inclut explicitement : raisonnement scientifique (D1), mathématiques `<think>` (D2), code multi-fichiers (D3), **connaissances factuelles rares de queue longue** (D4), **12 familles linguistiques** (D5), consignes sous contraintes (D6), **appels d'outils au format XML natif `<tool_call><function=...><parameter=...>` de `chat_template.jinja`** et schémas JSON (D7), traces agentiques à `reasoning_effort="xhigh"` et `preserve_thinking=True` (D8), documents longs continus (D9), et **entrées multimodales image (`<|image_pad|>`) ET vidéo (`<|video_pad|>`, M-RoPE 3D `[11, 11, 10]`)** (D10).
2. **Collecte sans biais de $H_O$ sur les `520` matrices 2D (incluant la branche `MTP`) :**
   - **Graphe principal (`512` matrices 2D du Tronc et de Vision) :** sur les logits $z_t = \mathrm{lm\_head}(h_t^{(63)}) \in \mathbb{R}^{248320}$, tirer $S = 4$ vecteurs de Rademacher $\zeta_t^{(s)} \in \{\pm 1\}^{248320}$ et propager le pseudo-gradient sans biais $u_t^{(s)} = \mathrm{diag}(\sqrt{p_t})\zeta_t^{(s)} - p_t(\sqrt{p_t}^\top \zeta_t^{(s)})$.
   - **Graphe auxiliaire obligatoire pour le module `MTP` (`8` matrices 2D propres dans `mtp.*`) :** comme `Qwen3_5ForConditionalGeneration.forward` ignore `mtp.*`, exécuter explicitement en Passe 1 la branche `mtp(h_t^{(63)}, \mathrm{embed}(x_{t+1})) \to \mathrm{lm\_head}(h_{t+1}^{\text{MTP}})` et y injecter ses propres sondes Rademacher-Fisher par rapport à $p_{\text{prof}, t+1}$ afin d'obtenir des matrices $(H_I^{\text{MTP}}, H_O^{\text{MTP}})$ non nulles et bien conditionnées.
   - **Stockage bloc-diagonal $B_O = 128$ avec gestion exacte des queues :** chaque matrice $m \times n$ est partitionnée en $\lceil m / 128 \rceil$ blocs de taille $b_r = \min(128, m - 128 r)$ (gérant sans erreur `vis_mlp_fc1` où $4304 = 33 \times 128 + 80$). Empreinte exacte en FP32 : `8,91 Mo` pour une matrice `[17408, 5120]`, `127,14 Mo` pour `lm_head`, et `~2,38 Go` sur disque pour les `520` matrices (`~1,20 Go` si seul le triangle supérieur est sérialisé).

---

### Passe 2 — Quantification `KA-BGCC-RDOQ` avec Codeur Sphérique `SU-E8-rANS` et Allocation `EB-RCO-Slack`
1. **Formulation causale exacte de `KA-RDOQ` par bloc de lignes $b_r \le 128$ :**
   Un seul bloc Transformer réside en VRAM (`Algorithme 2` de `GPTAQ`). Avec $H_{q,q}' = U X_q X_q^\top U^\top + \lambda_H I_n = L_I D_I L_I^\top$ ($L_I$ triangulaire supérieure à diagonale unité), $\Delta C_X' = U (X_{\text{fp}} - X_q) X_q^\top U^\top$ et $H_{O,\text{blk}} + \lambda_G I_{b_r} = L_O D_O L_O^\top$, la cible corrigée de la dérive amont vaut (incluant la diagonale) :
   $$W_{\text{eff}}' = W' + W' \Delta C_X' (H_{q,q}')^{-1}.$$
   Notant $U_{s, b_8} = (L_I)_{8b_8:8b_8+8,\; 8b_8:8b_8+8}$ et $M_{\text{blk}, b_8} = U_{s, b_8} \mathrm{diag}(D_{I, 8b_8:8b_8+8}) U_{s, b_8}^\top$, l'onde frontale anti-diagonale ($w = 0,\dots,b_r + n/8 - 2$, couples indépendants $i + b_8 = w$) calcule à partir des seules colonnes **strictement antérieures** `< 8b_8` et lignes **strictement antérieures** `< i` :
   $$u_{i, b_8} = (W_{\text{eff}}')_{i, 8b_8:8b_8+8} + \sum_{k < i} (L_O)_{k,i}\,\Delta W'_{k, 8b_8:8b_8+8} + \Big(\Delta W'_{i, <8b_8} + \sum_{k < i} (L_O)_{k,i}\,\Delta W'_{k, <8b_8}\Big)(L_I)_{<8b_8,\; 8b_8:8b_8+8}\,U_{s, b_8}^{-1},$$
   puis sélectionne $q = \frac{1}{2}\alpha_{i,g} u_{\text{int}}$ ($u_{\text{int}} \in 2E_8$, coquille $m = \|u_{\text{int}}\|_2^2 / 8$) minimisant :
   $$(u_{i, b_8} - q)\,M_{\text{blk}, b_8}\,(u_{i, b_8} - q)^\top + \frac{\lambda_{\text{RDOQ}}}{(D_O)_{i,i}}\Big(-\log_2 \widehat{P}_{\text{shell}}(m) + \log_2 |\mathcal{S}_m|\Big),$$
   suivi optionnellement de `1` à `2` balayages Gauss-Seidel bidirectionnels `ABCD-E8` (`+0,08 dB` mesurés en `EXP-V2-01`).
2. **Couplage Bilinéaire `BGCC` sur les 64 (+1 MTP) blocs SwiGLU (activé si $N_{\text{cal}} / n \ge 10$) :**
   Dans chaque MLP, quantifier **d'abord** `gate_proj` en $\widehat{W}_g$. Lors de la quantification de `up_proj`, calculer sur `calib_train` les portes réelles $g_{q,i,t} = \mathrm{SiLU}(\widehat{w}_{g,i} x_{q,t})$ et $g_{\text{fp},i,t} = \mathrm{SiLU}(w_{g,i} x_{\text{fp},t})$, puis remplacer la cible de chaque ligne $i$ de `up_proj` par la cible régularisée réduisant l'erreur post-SwiGLU $g_q \odot (\widehat{W}_u X_q) - g_{\text{fp}} \odot (W_u X_{\text{fp}})$ (avec mélange convexe $0,7 H_{g_q^2} + 0,3 \bar{g}_{q,i}^2 H_{q,q}$ pour prévenir tout sur-ajustement).
3. **Allocation `EB-RCO-Slack` avec contraintes multi-budgets disjointes :**
   - Initialiser les débits par le solveur Water-Filling (`budget_lumen_max.py`), générer $K = 5$ variantes encodées en `SU-E8-rANS` par groupe (`~48 Go` compressés mis en cache en RAM hôte `128 Go` pour éviter tout goulet d'I/O NVMe pendant l'optimisation).
   - Optimiser sur `select_dev` la perte conjointe $\mathcal{L}_{\text{KL}}^{\text{main}} + \beta_{\text{MTP}} \mathcal{L}_{\text{KL}}^{\text{MTP}}$ sur la variété de budget de `RCO` avec variable d'écart $s \ge 0$ et **trois contraintes simultanées** (supportées nativement par `RCO` §A.8) : plafond global $\le 9\,985\,000\,000$ o, plancher `Vision Tower` $\ge 195\,000\,000$ o, et plancher `MTP propre` $\ge 135\,000\,000$ o.
   - Décoder l'assignation entière par programmation dynamique avec coûts arrondis par excès au pas de grille (`zero-overflow`), puis effectuer une re-passe finale `KA-BGCC-RDOQ` le long de l'assignation retenue.

---

### Passes 3a, 3b & Passe 4 — Récupération `EAR-PV` avec Rollback Monotone et Vérification à Froid
1. **Cache enseignant exact sur NVMe :** pour chaque position $t$, stocker les `128` logits `FP16` (`256` o), leurs **`128` indices `int32` (`512` o, requis car `vocab_size = 248 320 > 65 535`)** et la log-masse restante `FP32` (`4` o) = `772` octets/token, soit **`3,238 Go` pour `4,19M` tokens** (`2,298 Go` avec empaquetage `18 bits/ID`), plus la cible `Top-32` pour `MTP` (`+0,82 Go`).
2. **Boucle `EAR-PV` sécurisée par validation et rollback :**
   - Duel pilote préalable à temps GPU égal (`2 h` vs `2 h`) entre hors-politique et mélange `50 % GKD` (avec sandbox cohérente obligatoire pour les appels XML d'outils).
   - Alterner : (P-step) réglage continu des échelles et `RMSNorm` sur perte conjointe `Main + MTP` avec régularisation proximale vers Passe 2 pour préserver les strates rares ; (V-step) recherche locale dans `{0} ∪ (λ + Kissing_240(E8))` sur une petite fraction ($\le 2\text{–}5\,\%$) des blocs sous pénalité duale $-\mu_{\text{dual}}\log_2 \widehat{P}_{\text{shell}}(m)$ ; (EM-step) mise à jour des `14` tables `SU-E8-rANS`, encodage/décodage complet vers un fichier temporaire portable (`tempfile`), vérification `size_bytes <= 10_000_000_000`, et **rejet automatique (rollback) du pas V si la KL mesurée sur `select_dev` après décodage réel n'a pas diminué**.
3. **Passe 4 :** export définitif, contrôle d'empreinte `SHA-256`, vérification `size_bytes <= 10_000_000_000` (sans remplissage artificiel) et ré-ouverture à froid depuis le disque pour toute évaluation de confirmation.

---

## 3. Transposition aux profils secondaires `Éco` (`9 Go`) et `Mini` (`8 Go` texte seul)

Aucune règle de trois sur les scores n'est appliquée (**« Non estimable avec les données disponibles »**) :

- **`FORMIC-Éco` (`≤ 9 000 000 000` octets complets, `8,9815 Go` calculés, `2,586 bpw` effectif) :** conserve exactement les 11 domaines (Texte `8,665 Go` + Vision `186,2 Mo` + MTP `130,7 Mo`), réutilise les statistiques $(H_I, H_O)$ de Passe 1, relance le Water-Filling (`R_base = 2,403 bpw`) puis `EB-RCO-Slack` sous plafond `9 Go`, et requiert sa propre Passe 3b.
- **`FORMIC-Mini` (`≤ 8 000 000 000` octets texte seul, `7,9815 Go` calculés, `2,374 bpw` effectif hors MTP ou `2,337 bpw` avec MTP `118,0 Mo`) :** l'omission de la Vision Tower exclut le Domaine 10 (Image/Vidéo). Conformément à la **Section 3 des Instructions Permanentes Formic**, `Mini` doit être traitée comme une **branche spécialisée distincte** (évaluée sur les 10 domaines textuels, avec vigilance accrue sur D4 factuel rare, D5 multilingue et D7 balises XML `<tool_call>`), ne satisfaisant pas l'objectif multimodal complet du modèle de référence.

---

## 4. Matrice d'intégration avec le Chantier B (`A0 / A1` × `Q0 / Q1`)

Conformément à `COORDINATION_CHANTIERS_LUMEN.md` §5, l'évaluation finale compare au même plafond pertinent `A0Q0`, `A0Q1`, `A1Q0` et `A1Q1` (en laissant explicitement vide toute cellule non exécutée). Pour `A1Q1`, toutes les statistiques $(H_I, H_O, \Delta C_X)$ et l'alignement de super-tuiles doivent être recalculés dans les nouvelles dimensions et bases de `A1`, sans additionner les gains séparés de A et B.
