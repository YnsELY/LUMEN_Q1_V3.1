# FORMIC v2 — Recette corrigée de compression de Qwen3.8-27B

**Statut :** spécification prête à développer. Remplace `FORMIC_RECETTE_COMPRESSION.md` (v1, conservée telle quelle pour référence).
**Bilan d'octets :** `formic_v2_budget.py` → `formic_v2_budget_results.json` (Python pur, relançable).
**Unités :** 1 Go = 10⁹ octets ; 1 Gio = 2³⁰ octets ; bpw = bits par poids.

---

## 1. Ce qui change par rapport à la v1

### 1.1 Nouvel objectif

L'objectif « 100 % des capacités » est abandonné : il est inatteignable sous ~11 bpw, et le critère « score ≥ base sur 11 domaines » rejetterait même un modèle identique (≈ 0,1 % de chances de passer par simple bruit d'échantillonnage).

Le nouvel objectif est : **à taille donnée, perdre moins que tout autre modèle quantifié de taille équivalente**, et trouver la plus petite taille où la perte reste acceptable. Trois profils sont livrés (section 2).

### 1.2 Logique inversée : la taille est la contrainte, la perte est le résultat

Dans la v1, les débits par famille étaient fixés à la main et la conservation était supposée. Ici :

1. la **taille cible en octets** est imposée ;
2. l'**allocation des bits** est calculée à partir de sensibilités mesurées, sous ce budget ;
3. la **perte** est mesurée, puis comparée à des contrôles de même taille.

### 1.3 Tableau des corrections

| Élément v1 | Décision v2 | Raison |
|---|---|---|
| Inventaire des 1 199 tenseurs, jauges exactes | **Conservé** | Arithmétique reproduite à l'identique. |
| Tenseurs de contrôle DeltaNet et normes gardés exacts | **Conservé** | 0,054 Go pour supprimer un risque majeur à long contexte. |
| Préconditionnement Hadamard à droite (BSHP) | **Conservé sous réserve d'ablation** | Sur le jouet v1, Sinkhorn seul donne déjà crête 4,23 / kurtosis 2,99 : l'apport du Hadamard n'est pas démontré. |
| Égalisation Sinkhorn avec exposants ±1/8 | **Remplacé** par un pas unique dans le domaine pondéré par Fisher | L'égalisation convient à un code à débit fixe ; avec un codage entropique elle annule le bénéfice du débit variable. Le signe de l'exposant différait en plus entre deux documents v1. |
| Réseau E8 infini + codage rANS | **Conservé**, avec arrondi optimisé débit-distorsion (RDOQ) | Écart à Shannon mesuré : 0,16 → 0,135 bpw avec RDOQ. Le débit continu est l'atout décisif (voir 3.3). |
| Échelles par 128 et 1 024 poids (0,27 Go) | **Supprimé** | Après gaussiennisation elles ne portent aucune information. |
| Tuiles 128×256, alignement 128 octets (0,18 Go) | **Remplacé** par tuiles 128×1024, alignement 16 octets (0,06 Go) | Gain sans effet sur la qualité. |
| Débits fixés à la main, identiques sur 64 couches | **Remplacé** par allocation mesurée par couche et par famille | Principal levier de qualité à taille fixe. |
| Allocation HJMA (K > V > Q) | **Suspendu** : Q, K, V restent trois groupes d'allocation, les débits viennent de la mesure | Sur le jouet v1, HJMA dégradait l'erreur de sortie (0,3485 contre 0,3106). |
| Correcteur bas-rang KM-CSRC (0,16 Go) | **Désactivé par défaut**, option soumise à ablation | Coût de 0,129 bpw sur `down_proj` ; validé uniquement sur un jouet qui présuppose l'hypothèse. |
| Passe 3 limitée aux paramètres continus | **Étendu** à la mise à jour des codes discrets | Levier le plus fort connu sous 3 bpw. |
| Critère « 100 % » et kill-switches K2/K3 | **Remplacé** par comparaison à taille égale + budget de perte par profil | Voir section 5. |
| Débits d'inférence (87,8 à 268,9 tok/s) | **Retirés** | Le coût de 1,75 instruction par poids n'est pas réaliste ; à mesurer (Étape 2). |
| Affirmations « 190× », « −0,68 bpw vs scalaire », « −0,28 à −0,35 bpw vs QTIP » | **Retirées** | Non reproductibles ou comparaisons à des références faibles. |

---

## 2. Les trois profils

Le fichier est **modulaire** : un cœur texte, un module Vision et un module MTP séparés. Chaque profil peut donc être déployé dans les trois périmètres.

| Profil | Rôle | Complet (texte + Vision + MTP) | Sans Vision (texte + MTP) | Texte seul | bpw effectif complet |
|---|---|---:|---:|---:|---:|
| **FORMIC-Max** | Le moins de pertes possible vers 10 Go, rien d'amputé | **10,00 Go** (9,31 Gio) | 9,80 Go | 9,66 Go | 2,880 |
| **FORMIC-Éco** | Meilleur rapport taille / perte (coude de la courbe) | **9,00 Go** (8,38 Gio) | 8,82 Go | 8,69 Go | 2,592 |
| **FORMIC-Mini** | Le plus petit, texte seul, perte minimale à cette taille | 8,30 Go | 8,12 Go | **8,00 Go** (7,45 Gio) | 2,390 |

En gras : le périmètre de référence du profil, sur lequel la taille cible est résolue.

### 2.1 Débits (point de départ avant allocation mesurée)

| Composant | Paramètres | Max | Éco | Mini |
|---|---:|---:|---:|---:|
| 64 MLP (`gate`, `up`, `down`) | 17,11 G | 2,74 | 2,45 | 2,24 |
| 48 DeltaNet (`q`, `k`, `v`, `z`, `out`) | 5,54 G | 2,94 | 2,65 | 2,44 |
| 16 Full Attention (`q+gate`, `k`, `v`, `o`) | 1,68 G | 3,19 | 2,90 | 2,69 |
| **Moyenne du tronc** | 24,33 G | **2,81** | **2,53** | **2,32** |
| `embed_tokens` | 1,27 G | 2,80 | 2,50 | 2,30 |
| `lm_head` | 1,27 G | 3,30 | 3,00 | 2,80 |
| Vision (110 matrices) | 0,46 G | 3,30 | 3,00 | 2,90 (option) |
| MTP (8 matrices) | 0,42 G | 2,60 | 2,40 | 2,30 (option) |

Les écarts entre classes (+0,20 pour DeltaNet, +0,45 pour l'attention) sont des a priori provisoires. L'Étape 4 les remplace par l'allocation mesurée, à nombre d'octets identique.

### 2.2 Décomposition en octets

| Poste | Max | Éco | Mini |
|---|---:|---:|---:|
| Cœur texte — payload | 9,528 Go | 8,556 Go | 7,867 Go |
| Cœur texte — tenseurs exacts | 0,054 Go | 0,054 Go | 0,054 Go |
| Cœur texte — auxiliaires (jauges, tuiles, tables) | 0,079 Go | 0,079 Go | 0,079 Go |
| Module Vision | 0,200 Go | 0,183 Go | 0,177 Go |
| Module MTP | 0,139 Go | 0,129 Go | 0,123 Go |

Les auxiliaires passent de 0,625 Go (v1) à 0,083 Go.

### 2.3 Fidélité attendue (indicatif, non mesuré)

Extrapolation par la loi KLD ∝ 4^−R, ancrée sur le point cité par le projet (OrcaSAQ2, EXL3 à 3,21 bpw : KLD 0,031, accord top-1 93,2 %), en supposant que l'allocation mesurée et le réglage des codes valent +0,3 bpw (prudent) à +0,6 bpw (optimiste).

| Profil | KLD (nat/token) | Accord top-1 | Perte typique sur benchmarks |
|---|---:|---:|---|
| Max | 0,023 – 0,035 | 92,7 – 94,1 % | 0 à −1 pt |
| Éco | 0,035 – 0,053 | 91,1 – 92,8 % | −0,5 à −2 pts |
| Mini | 0,046 – 0,070 | 89,8 – 91,7 % | −1,5 à −4 pts |

Lecture : FORMIC-Max viserait la fidélité d'OrcaSAQ2 (12,27 Go sans Vision) dans 10,00 Go complets. Ces chiffres sont des hypothèses de travail ; la colonne « perte sur benchmarks » est une estimation d'ordre de grandeur. Ils seront remplacés par les mesures de l'Étape 6.

### 2.4 Équivalents de taille à battre

| Profil | Références publiques proches (chiffres du dossier, à revérifier à l'Étape 0) | Contrôle à taille égale |
|---|---|---|
| Max | GSQ-RCO IQ3_XXS (10,09 Go texte, 11,37 complet : GPQA-D −1,01, LCB −1,14) ; Escha-W2 (10,15 Go texte : GPQA-D −1,52). Objectif étendu : IQ3_S (13,05 complet), OrcaSAQ2 (13,19 complet). | EXL3 et GGUF imatrix produits au même nombre d'octets, même périmètre |
| Éco | Q2_K ternaire (9,76 Go texte, non évalué publiquement) | idem à 9,00 Go |
| Mini | Aucun modèle public aussi petit dans le dossier | idem à 8,00 Go texte seul |

Le contrôle à taille égale est indispensable : c'est la seule comparaison qui teste réellement « moins de pertes que les équivalents ».

---

## 3. La recette

### 3.1 Vue d'ensemble

```text
Passe 0  Préparation     jauges exactes, mise à l'écart des tenseurs gardés exacts
Passe 1  Statistiques    Gram d'entrée H par matrice, diagonale de Fisher de sortie G, sensibilités a_g
Passe 2  Quantification  préconditionnement -> LDLQ sur E8 avec RDOQ -> rANS, couche par couche sur le préfixe déjà quantifié
Passe 3  Réglage         3a par macro-bloc (erreur de sortie), 3b global (KL vers le modèle d'origine), codes discrets inclus
Passe 4  Empaquetage     trois modules, contrôle de taille à l'octet
```

### 3.2 Passe 0 — Préparation (repris de la v1)

- Jauges exactes fusionnées hors ligne : `in_proj_qkv` ↔ `conv1d`, monomiale SwiGLU, rotation par tête dans la Vision.
- Gardés exacts : `in_proj_a`, `in_proj_b`, `conv1d` (BF16) ; `A_log`, `dt_bias`, normes (FP32) ; petits tenseurs Vision (BF16).
- Aucune rotation fusionnée entre couches (obstructions de la v1 inchangées).

### 3.3 Passe 2 — Quantifieur

**Préconditionnement.** Pour chaque matrice W (m × n) :

- côté entrée : `W̃ = D_out⁻¹ · W · D_in · Uᵀ`, avec U le Hadamard stridé à droite de la v1 (bloc 128 quand 128 divise n, sinon 16) ;
- `D_in` : égalisation des colonnes par l'écart-type des activations d'entrée ;
- `D_out` : **pas d'égalisation des lignes**. `d_out,i ∝ G_ii^(−1/2)`, borné à [1/16, 16], de sorte qu'un pas de quantification unique donne une erreur par ligne inversement proportionnelle à sa sensibilité. Une ligne peu sensible reçoit automatiquement moins de bits.

Trois variantes sont comparées à l'Étape 1 et la meilleure est retenue : Sinkhorn seul, Hadamard à droite, Hadamard des deux côtés.

**Quantification.** LDLQ par blocs de 8 colonnes sur le réseau E8 non tronqué (algorithme de Conway-Sloane de la v1), avec deux ajouts :

- **RDOQ** : pour chaque bloc, choisir parmi les candidats voisins (point le plus proche dans chaque co-ensemble, versions contractées, vecteur nul) celui qui minimise `distorsion + λ · bits`. Mesuré : −0,025 bpw à distorsion égale.
- **Pas par bloc de colonnes** (option, à mesurer) : `pas_j ∝ d_j^(−1/2)` où `d_j` est la diagonale de Cholesky du Hessien. Gain mesuré sur jouet : 0 à 0,13 bpw selon le spectre. Stockage : un FP16 par bloc de 8 colonnes et par matrice.

**Codage.** `Slot-CP-rANS` de la v1 (deux symboles 4D par vecteur, 8ᵉ signe déduit), avec deux corrections de format : entrée de table sur 2 × u32 (le champ 11+10+11 bits de la v1 ne peut pas contenir une fréquence de 4 096), et longueur de flux stockée par lane.

**Pourquoi E8 + rANS plutôt qu'un treillis type QTIP.** Un treillis à débit fixe a un meilleur écart à Shannon (≈ 0,07 bpw selon le papier QTIP, à revérifier), mais ses débits sont entiers. Pour viser 2,74 bpw il faut mélanger des lignes à 2 et 3 bits, ce qui coûte 0,156 bpw (calcul dans `formic_v2_budget.py`), soit plus que l'avantage du treillis. Le débit continu du codage entropique est donc retenu. Un treillis codé par entropie reste une évolution possible, hors périmètre v2.

### 3.4 Passe 1 — Allocation des bits mesurée

- **Groupes** : pour chaque couche, une entrée par famille de tenseurs (`gate`, `up`, `down`, `q`, `k`, `v`, `z`, `out`, etc.), plus `embed_tokens`, `lm_head`, Vision et MTP. Environ 600 groupes.
- **Mesure** : quantifier le groupe g seul à un débit test `R_t` (les autres restant en BF16), mesurer la hausse de KL `ΔKL_g` sur des tokens de calibration, poser `a_g = ΔKL_g · 4^(R_t)`.
- **Résolution** : minimiser `Σ a_g · 4^(−R_g)` sous `Σ n_g · R_g = budget`, soit `R_g = R̄ + ½·log₂((a_g / n_g) / moyenne géométrique)`, avec plancher 1,8 bpw et plafond 4,5 bpw.
- **Contrôle de taille** : le débit réel n'est connu qu'après codage. Le pas (ou λ) de chaque tenseur est ajusté par dichotomie pour tenir son budget à ±0,1 %, et l'écart résiduel est absorbé par les derniers tenseurs.

Cas particuliers :

- `embed_tokens` : allocation par ligne pondérée par la fréquence du token, avec plancher pour protéger les langues rares.
- `lm_head` : test préalable de prédiction depuis `embed_tokens` (Étape 1). S'il est fortement corrélé, on ne code que le résidu.
- MTP : il n'influe que sur la vitesse (le tronc vérifie chaque proposition). Son débit est fixé par le taux d'acceptation, pas par la qualité.

### 3.5 Passe 3 — Réglage après quantification

- **3a, par macro-bloc (4 couches)** : minimiser l'erreur de sortie du bloc quantifié par rapport au bloc d'origine. Paramètres ajustés : échelles de ligne, normes, et codes discrets.
- **3b, global** : minimiser la KL vers les logits du modèle d'origine (top-128 mis en cache), plus un terme d'alignement MTP.
- **Mise à jour des codes** : on garde une copie latente des poids, on applique le pas de gradient, puis on re-projette sur E8 (avec RDOQ au même λ) les seuls blocs dont le déplacement dépasse une fraction du pas. Le débit est recontrôlé à la fin.
- **Données** : le jeu `FORMIC-Calib-v1` et ses quatre barrières de décontamination sont repris de la v1. La passe 3b est étendue à 30–50 M de tokens pour limiter le sur-ajustement.

Chaque sous-passe est livrée séparément et mesurée : v2.0 = Passes 0–2 seules, v2.1 = + 3a, v2.2 = + 3b.

### 3.6 Passe 4 — Format

- Trois fichiers : `formic-core`, `formic-vision`, `formic-mtp`.
- Par matrice : `d_out` FP16[m], `d_in` FP16[n], pas par bloc de colonnes FP16[n/8], graine du Hadamard, puis les tuiles.
- Tuile de 128 × 1 024 poids : en-tête 16 o, 32 états rANS, 32 longueurs de flux rANS, 32 longueurs de flux de signes, alignement 16 o (≈ 280 o par tuile).
- 64 profils de tables rANS partagées.

### 3.7 Inférence : ce qui est acquis et ce qui ne l'est pas

- **Acquis** : la taille sur disque, et l'identité entre le modèle décodé et le modèle évalué.
- **Non acquis** : garder les poids compressés en VRAM suppose un décodage rANS au vol assez rapide. Ce n'est pas démontré.
- **Repli** : transcodage au chargement vers un format à débit fixe (≈ 5 bits par coordonnée E8, soit ≈ 17 Go de VRAM pour Max). La taille de fichier reste celle du profil, la VRAM non.
- L'Étape 2 mesure le débit réel du décodeur et tranche entre les deux modes.

---

## 4. Plan de développement

| Étape | Contenu | Matériel | Critère de passage | Si échec |
|---:|---|---|---|---|
| **0** | Revérifier les sources (dépôts, tailles, scores), figer le harnais d'évaluation, mesurer le bruit plancher : BF16 contre BF16 (autre lot, autre matériel) et FP8 contre BF16 | 1 GPU 80 Go, ~1 jour | Bruit plancher connu pour KLD, top-1 et chaque benchmark | — |
| **1** | Couche 0 réelle (DeltaNet + MLP) et `lm_head`/`embed_tokens`, chargés par tenseur. Ablations : 3 préconditionnements ; E8 simple / RDOQ / pas par colonne ; test de prédiction de `lm_head` | RTX 2050 locale | Erreur de sortie pondérée plus basse qu'un LDLQ scalaire de même débit ; choix du préconditionnement | Garder la variante gagnante, même si c'est Sinkhorn seul |
| **2** | Décodeur de référence (CPU puis CUDA), identité bit à bit avec l'encodeur, mesure du débit de décodage | RTX 2050 locale | 0 erreur de décodage ; débit mesuré | Mode transcodage au chargement |
| **3** | Passes 0–2 sur un macro-bloc greffé dans le modèle d'origine ; ablation KM-CSRC à octets égaux | 1 GPU 80 Go, ~3 h | KLD inférieure au contrôle EXL3 de même débit sur ce bloc | Revoir quantifieur avant d'aller plus loin |
| **4** | Carte de sensibilité des ~600 groupes, puis allocation | 1 GPU 80 Go, ~6 h | Gain de KLD prédit ≥ 15 % par rapport à l'allocation a priori | Garder l'allocation a priori |
| **5** | Compression complète v2.0 des trois profils + contrôles à taille égale | 1 GPU 80 Go, ~10 h par profil | Taille ≤ cible ; KLD et top-1 meilleurs que le contrôle | Analyse par couche des écarts |
| **6** | Passes 3a puis 3b ; balayage de 8,0 à 10,5 Go pour localiser le coude | 1–2 GPU 80 Go, ~2 jours | Gain mesuré de chaque sous-passe ; position du coude | Livrer v2.0 ou v2.1 |
| **7** | Batterie des 11 domaines sur les trois profils, les contrôles et les références publiques | 4 GPU 80 Go, ~14 h | Critères de la section 5 | Ajuster la taille du profil concerné |

Les Étapes 1 et 2 ne coûtent rien et décident si le reste vaut la dépense. Les durées sont des estimations.

Modules à écrire pour l'Étape 1 : `precond` (jauges, Hadamard, D_in, D_out), `lattice_e8` (Conway-Sloane, candidats RDOQ), `ldlq` (rétroaction par blocs), `rans` (codeur, décodeur, tables), `stats` (Gram, Fisher diagonale), `bench_layer` (ablations et rapport).

---

## 5. Critères de réussite

### 5.1 Mesures

- **Niveau 1, sur 1,1 M de tokens réservés** : KLD moyenne et au 99ᵉ centile, accord top-1, ventilés par domaine. Puissance statistique élevée, coût faible.
- **Niveau 2, batterie des 11 domaines de la v1** : différences appariées avec graines communes, intervalle de confiance bootstrap par item.
- **Sentinelles** : rappel à long contexte (RULER 262k et 1M) et dérive de l'état DeltaNet ; `SimpleQA` ; OCR et graphiques pour la Vision ; taux d'acceptation MTP ; longueur des blocs de réflexion.

### 5.2 Critères par profil

| Critère | Max | Éco | Mini |
|---|---|---|---|
| **S1 — Taille** | ≤ 10,00 Go complet | ≤ 9,00 Go complet | ≤ 8,00 Go texte seul |
| **S2 — Contre le contrôle de même taille** | KLD plus basse et top-1 plus haut (intervalle à 95 % excluant 0) ; macro-moyenne des 11 domaines ≥ contrôle ; aucun domaine significativement pire | idem | idem (10 domaines, sans Vision) |
| **S3 — Contre les références publiques** | KLD ≤ IQ3_XXS et Escha-W2 ; objectif étendu : ≤ IQ3_S | KLD ≤ Q2_K ternaire | — |
| **S4 — Budget de perte vs BF16** (cibles) | macro ≤ 0,5 pt ; pire domaine ≤ 1,5 pt | macro ≤ 1,0 pt ; pire domaine ≤ 2,5 pts | macro ≤ 2,0 pts ; pire domaine ≤ 4,0 pts |
| **S5 — Sentinelles** | Aucun effondrement à long contexte ; acceptation MTP ≥ 0,9 × base ; réflexion ≤ +5 % | idem | idem, hors Vision et MTP |

Les seuils S4 sont des cibles à recaler sur le bruit plancher de l'Étape 0.

### 5.3 Définition du coude (profil Éco)

Sur le balayage de l'Étape 6 (points à 8,0 / 8,5 / 9,0 / 9,5 / 10,0 Go complets), Éco est la **plus petite taille** qui vérifie :

1. le budget S4 du profil Éco ;
2. une dégradation encore régulière : retirer 0,5 Go de plus ferait monter la KLD de plus de ×1,4 (la loi 4^−R prédit ×1,26), signe que l'on entre dans la zone d'effondrement.

La valeur de 9,00 Go est un a priori ; la mesure peut la déplacer.

---

## 6. Risques ouverts

| Risque | Conséquence | Parade |
|---|---|---|
| Le réglage des codes rapporte moins que +0,3 bpw sur cette architecture hybride | Les profils perdent plus que prévu | Mesuré dès l'Étape 6 ; remonter la taille du profil |
| Décodage rANS au vol trop lent | VRAM ≈ 17 Go au lieu de ≈ 10 Go | Mode transcodage ; la taille de fichier reste tenue |
| Sensibilité extrême de quelques têtes DeltaNet à long contexte | Perte de rappel au-delà de 262k | Plafond d'allocation relevé pour ces groupes ; sentinelle dédiée |
| Sur-ajustement de la passe 3b | Gain sur calibration, perte ailleurs | Données élargies, arrêt sur jeu réservé |
| Sources publiques non revérifiées | Références de comparaison fausses | Étape 0 |
| Contrôle EXL3 non disponible pour cette architecture | Pas de comparaison à taille égale | Contrôle GGUF imatrix, ou LDLQ scalaire interne |

---

## 7. Statut des affirmations

| Affirmation | Statut |
|---|---|
| Tailles des trois profils et de leurs modules | **Calculé** (contrainte tenue par construction) |
| Auxiliaires ramenés de 0,625 à 0,083 Go | **Calculé** |
| RDOQ : −0,025 bpw à distorsion égale | **Mesuré sur source gaussienne** |
| Pas par bloc de colonnes : 0 à 0,13 bpw | **Mesuré sur jouet**, dépend du spectre réel |
| Perte de mélange d'un code à débit entier : 0,10 à 0,16 bpw | **Calculé** |
| Écart à Shannon de QTIP ≈ 0,07 bpw | **De mémoire**, à revérifier |
| KLD, top-1 et pertes par profil | **Hypothèse de travail**, à mesurer |
| « Moins de pertes que les équivalents de taille » | **Objectif**, tranché aux Étapes 5 à 7 |
