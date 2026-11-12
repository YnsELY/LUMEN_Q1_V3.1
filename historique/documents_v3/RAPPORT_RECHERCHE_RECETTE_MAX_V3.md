# LUMEN / FORMIC-Max — Recherche corrigée et approfondie, Q1-v3-ref

**12 novembre 2026. Branche de recherche CPU ; aucune promotion en recette de production.**

## 1. Résultat décisionnel

Cette livraison remplace les affirmations techniques de la proposition `Q1-v2` par des spécifications corrigées, une bibliothèque exécutable, des tests indépendants et de nouveaux résultats synthétiques. Elle ne remplace pas la référence historique `Q0` ni les fichiers de l’autre agent.

**Trois conclusions :**

1. L’inverse triangulaire de Block-8 LDLQ est réparé et comparé à une formulation dense indépendante. Un vrai encodeur/décodeur CPU E8 est livré : on peut désormais distinguer sans ambiguïté longueur estimée et longueur de fichier.
2. La **calibration asymétrique régularisée** est le premier candidat à éprouver sur les vraies activations. Dans le petit banc synthétique retenu, elle améliore la confirmation pour quatre graines sur cinq, avec une régression conservée dans le rapport. Cela ne prouve aucun gain sur Qwen.
3. Le **couplage SwiGLU sensible à la sortie complète** possède maintenant une réalisation sans Hessien dense par canal. Sa justesse conditionnelle est testée, mais ses résultats de confirmation sont irréguliers. Il reste expérimental et désactivé par défaut. Le polissage à coquille E8 constante maintient exactement les octets dans notre format, mais sur-ajuste les cinq nouveaux essais réalisés : il n’est pas promu non plus.

Aucun package Qwen de 10 Go, benchmark Formic, gain contre GSQ/RCO ou débit GPU n’a été produit ici. Les anciens nombres `+0,71 dB`, `+0,40/+0,61 dB`, `−39,2 % KL`, `0 erreur sur 314 688 vecteurs` restent des déclarations historiques **non utilisables comme validation de cette branche**. Le code utilisé auparavant comportait un défaut qui affecte les premières expériences ; leurs données manquent dans le lot disponible.

## 2. Périmètre, artefacts et degré de preuve

Objectif conservé : minimiser les pertes de capacités sous **10 000 000 000 octets complets**, avec texte, Vision, MTP et auxiliaires, sans abandon délibéré de domaine. Éco et Mini restent des branches secondaires ; aucun score n’est interpolé pour elles.

Le checkpoint demandé est `Qwen/Qwen3.8-27B`, révision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Les originaux et leurs empreintes sont conservés dans `originals/` et `sources/entrees.json`. L’inventaire de 1 199 tenseurs et 27 781 427 952 paramètres est **repris des fichiers du projet**, pas revalidé par un téléchargement des 18 en-têtes. La configuration courante est consultable [S14], mais ne prouve pas l’identité de la révision épinglée.

L’échantillon `qwen38_real_slices_sample.json`, ses lignes BF16 complètes et les traces originales ne sont pas présents. Deux recherches ciblées n’ont pas retrouvé cet échantillon. Il n’a pas été reconstitué à partir de nombres de synthèse. Aucun poids lourd n’a été téléchargé.

| Statut | Ce qui entre dans cette catégorie |
|---|---|
| Rapporté | Inventaire et mesures historiques du projet non reproduites ici |
| Publié | Résultat de la source primaire, dans son propre périmètre |
| Déduit sous hypothèses | Identités, budgets et propriétés établis pour les conventions explicites |
| Mesuré CPU | Tests et expériences exécutés avec les scripts livrés |
| Implémenté de référence | Code CPU utilisable pour un petit problème ou une vérification, pas un moteur 27B |
| À tester sur Qwen | Transfert de qualité, mémoire réelle, contexte long, Vision/MTP, kernels et produit complet |

## 3. Corrections déterminantes

### 3.1 LDLQ et conventions numériques

L’ancienne boucle d’inversion laissait nuls les coefficients hors diagonale. La référence calcule maintenant, de bas en haut :

$$V_{ij}=-\sum_{k=i+1}^{j}U_{ik}V_{kj},\qquad V_{ii}=1.$$

Sur le contre-exemple de l’audit, `max|UV−I|` passe de **0,5 à 0**. Des matrices aléatoires non diagonales, la reconstruction LDL et le résultat d’une onde frontale comparé à une formulation dense `G ⊗ H` sont aussi testés.

Autres corrections : rejet explicite d’un Gram non défini positif ; aucune réparation silencieuse des pivots par `max(x,1e−15)` ; SiLU numériquement stable **sans tronquer son entrée à ±20** ; contraintes de dimension vérifiées ; arrondi des échelles FP16 avant l’évaluation et l’export. Une précision CPU float64 validée ne certifie pas l’équivalence des futurs kernels BF16.

### 3.2 RDOQ et descente coordonnée

Le sélecteur est une **recherche finie de candidats**, pas un minimiseur global exact sur E8 sous toute métrique. L’algorithme inclut le code courant, zéro et des candidats de réseau ; le voisinage de 240 racines est optionnel.

Les balayages alternent effectivement leur sens. Chaque mouvement est accepté après calcul de son effet sur un objectif explicitement figé. La métrique n’est pas normalisée indépendamment de son multiplicateur de débit. La propriété prouvée est la non-augmentation de cet objectif de calibration **avec les mêmes tables**, pas de la KL finale, de la longueur réelle après changement de table ou d’un jeu indépendant.

### 3.3 Activations, calibration et données

Le test `RMSNorm(slice(e))` est retiré comme substitut à `slice(RMSNorm(e))`. La fonction corrigée exige la moyenne des carrés du vecteur complet. Pour le MLP, l’entrée attendue est celle réellement fournie après l’attention/récurrence et la normalisation correspondante, même à la couche 0 [S13].

Sur un test par colonnes, les contributions des colonnes non modifiées doivent rester dans la préactivation avant la non-linéarité. Des embeddings tronqués ne permettent pas de mesurer le vrai SwiGLU.

Les ensembles sont séparés par documents, sources ou épisodes. Le partage de mots entre documents n’est pas interdit : le défaut précédent était la réutilisation des mêmes entrées non contextualisées comme prétendue validation séparée. `tools/validate_capture.py` vérifie le contrat des captures, leurs formes et leur empreinte ; il **ne peut pas certifier leur provenance réelle**.

## 4. Finaliste A — Calibration asymétrique et correction conditionnelle de SwiGLU

### 4.1 Cible asymétrique régularisée : commencer par le cas simple

GPTAQ motive une reconstruction qui vise le chemin haute précision intact plutôt que la couche haute précision appliquée à des entrées déjà dégradées [S01]. La réalisation livrée est plus limitée : elle résout exactement le problème ridge suivant sur les données fournies, puis applique notre quantifieur de référence.

Avec `Xq` les entrées de l’élève, `Xf` celles du professeur et `W` la matrice initiale :

$$\min_B\ \frac{\|BX_q-WX_f\|_F^2}{N}+\rho\|B-W\|_F^2,\qquad\rho>0.$$

En posant `H=Xq Xqᵀ/N+ρI`, les équations normales donnent :

$$B H=W X_fX_q^\top/N+\rho W.$$

Le programme résout le système sans former explicitement `H⁻¹`. Cela inclut le bon ancrage de régularisation ; remplacer simplement un inverse par un inverse amorti sans réexaminer le second membre changerait l’objectif.

**Ce qui est nouveau pour le dossier :** une cible réellement vérifiable, distinguée de la reproduction complète de GPTAQ, et une sélection par qualité réservée **après encodage/décodage réel**. Le contrôle recherche lui aussi plusieurs réglages de quantification.

Les biais et les transformations de base doivent être conservés dans la cible. Pour une couche affine, leur contribution est soustraite ou incluse via une variable définie explicitement. Ne pas appliquer aveuglément la formule sans biais à Vision.

### 4.2 BGCC par ligne : formulation exacte mais coût élevé

Avec la porte quantifiée fixée `g_i`, `F_i=Xq diag(g_i)` et cible `y_i`, le problème est :

$$\min_w\ \|wF_i-y_i\|_2^2/N+\rho\|w-w_{0,i}\|_2^2.$$

Le Gram ET le second membre utilisent les mêmes pondérations. Le Gram pondéré est retourné pour le choix des codes ; il n’est pas remplacé silencieusement par un Gram commun. La règle `Ncal ≥ 10n` est retirée : aucune expérience disponible ne prouve ce seuil universel. Le conditionnement, le rang effectif, la diversité et les pertes réservées sont à mesurer.

### 4.3 BGCC sur la sortie complète du MLP, sans Hessien matérialisé

Le produit `gate × up` seul n’exprime pas la sensibilité de la projection `down`. Notre proposition conditionnelle prend celle-ci en compte. Figer la porte quantifiée `g`, une projection de sortie `D` et des entrées `Xq`. Définir la cible professeur `Y` à la **sortie complète** du MLP, avec biais éventuels traités séparément :

$$\min_W J(W)=\frac{\|D[g\odot(WX_q)]-Y\|_F^2}{N}+\rho\|W-W_0\|_F^2.$$

C’est une quadratique strictement convexe pour `ρ>0`, conditionnellement aux composants figés. Son opérateur de demi-Hessien agit sur une perturbation `V` :

$$\mathcal A(V)=\frac{\{g\odot D^\top[D(g\odot VX_q)]\}X_q^\top}{N}+\rho V.$$

**Ni `DᵀD`, ni un Hessien `(mn)²`, ni `m` Grams `n×n` ne sont construits.** Le solveur par gradient conjugué préconditionné n’a besoin que de produits par cet opérateur. Un préconditionneur diagonal exact pour cet opérateur est :

$$a_{ij}=\|D_{:,i}\|_2^2\frac{\sum_t g_{it}^2X_{jt}^2}{N}+\rho.$$

La solution est vérifiée contre une résolution dense indépendante sur de petits systèmes ; les gradients sont testés par différences finies. L’usage d’un opérateur sans matrice explicite et le besoin de contrôler la convergence sont des propriétés classiques de CG [S11]. L’idée générale d’une reconstruction au niveau bloc possède aussi des antécédents [S12]. **Aucune priorité d’invention n’est revendiquée.**

Pour `[17408,5120]`, stocker un Gram FP32 par canal demanderait **1 825 361 100 800 octets**. La réalisation évite ce poste, mais conserve des activations et effectue plusieurs produits coûteux : elle n’est pas une promesse de campagne rapide. La version actuelle garde les entrées en RAM ; la collecte et les produits par mini-lots GPU restent à développer.

Le coût mémoire de référence est de l’ordre des paramètres et activations `O(mn+(n+m+d)N)`, avec constantes de tableaux temporaires et vecteurs CG. Il n’est pas nul. Les dimensions du vrai bloc rendent nécessaire une version par lots ou flux.

### 4.4 Correction directement dans les codes

Une seconde réalisation limite la recherche à huit coefficients d’une ligne de `up`. Pour la ligne `i`, soit `d_i=D[:,i]`, `F_b=diag(g_i) X_bᵀ` dans la convention appropriée, et `R=Y−D[g⊙WXq]`. Le Gram local est seulement `8×8` :

$$A_b=\|d_i\|^2F_b^\top F_b/N+\rho I_8.$$

Le programme calcule le gradient du résidu global et évalue les changements de code sur la vraie sortie locale du MLP. Ce n’est pas une optimisation simultanée du gate et du down, ni un algorithme globalement optimal de quantification. Son intérêt est un objectif cohérent avec une mémoire réduite ; ses essais hors calibration ne permettent pas encore de le promouvoir.

### 4.5 Courbure globale et terme de premier ordre : extension, pas prérequis

YAQA et KronQ motivent l’utilisation des sorties/gradients, au-delà des seules activations d’entrée [S02][S03]. Leur collecte complète n’est pas implémentée ici. Une identité utile est testée pour la sonde de logits :

$$u=(\operatorname{diag}\sqrt p-p\sqrt p^\top)\zeta,\quad E[uu^\top]=\operatorname{diag}(p)-pp^\top.$$

Le vecteur `u` est de moyenne nulle ; **c’est son produit extérieur** qui estime cette courbure. Un facteur de Kronecker ou une approximation temporelle n’hérite pas automatiquement d’un statut sans biais.

Après quantification d’un préfixe, le gradient de fidélité au point courant n’est généralement plus nul. Pour un gradient `B` et des facteurs SPD `G,H`, le problème local :

$$\langle B,\Delta\rangle+\tfrac12\operatorname{tr}(G\Delta H\Delta^\top)$$

a pour déplacement libre `Δ=−G⁻¹BH⁻¹`. Une borne sur la norme `G/H` limite ce déplacement avant quantification. `fresh_quadratic_target` implémente cette identité, sans prétendre reproduire G2PTQ dont seul le résumé auteur a été consulté [S15]. Le bénéfice sur le modèle reste à tester ; l’export et les données réservées peuvent rejeter la proposition.

## 5. Finaliste B — Fermer le codage réel puis chercher une meilleure utilisation des octets

### 5.1 Ce qui est maintenant encodé et décodé

La branche livre un codec CPU **nouveau et distinct** de `Slot-CP-rANS` : `LE8CPU3`. Elle représente `u=2z`, utilise une coquille E8 et un rang combinatoire, un vrai flux rANS et un format d’échappement explicite. La famille ANS et les mécanismes rANS sont documentés [S08][S09]. La cardinalité `240 sigma_3(m)` est vérifiée par un comptage dynamique indépendant [S10].

Le compteur `len(blob)` comprend l’en-tête, les tables, l’état, les rangs, les échappements et la somme de contrôle. Le conteneur de matrice ajoute les dimensions et les échelles FP16. Aucun terme fictif `+0,011 bit/vecteur` n’est présenté comme un encodage.

La limite est unique : coquilles `0..63`, puis ESC par défaut. ESC stocke huit entiers signés de 32 bits, soit **256 bits**, sans saturation et avec le même coût dans l’objectif. Les valeurs hors int32 sont refusées. Un échantillon de test n’autorise pas à déclarer tous les entiers encodables arbitrairement.

Les rangs sont regroupés en mots à bases mixtes ; chaque groupe occupe une longueur entière d’octets. L’algorithme de rang utilise une mémoïsation CPU qui consomme de la mémoire. **Ce n’est pas un décodeur GPU sans table.** Les tables sont locales au flux, pas 14 profils communs déjà calibrés au modèle. Les 32 lanes, FP8 SG512 et kernels d’inférence ne sont pas implémentés.

### 5.2 Ne pas imposer l’uniformité des directions si les données ne la justifient pas

L’égalité de norme ne prouve pas l’égalité des probabilités des codes d’un modèle appris. Un modèle de code sphérique uniforme est une référence propre, pas un optimum universel.

Deux modes exécutables sont donc fournis : `mode0 = coquille, rang uniforme sur toute la coquille` ; `mode1 = coquille + co-ensemble entier/demi-entier, rang uniforme dans ce co-ensemble`. Le mode1 a une table plus grande et doit payer ses octets supplémentaires. Pour les mêmes codes, il ne change pas les poids reconstruits.

Le choix ne doit pas être décidé sur le nom « adaptatif », mais sur le fichier réel : même mode de calcul, mêmes codes, inclusion des tables. Un profil partagé et son identifiant constituent encore une autre expérience, non livrée comme validée ici.

### 5.3 Polissage à coquille constante : propriété exacte de taille, pas de généralisation

Les essais libres de polissage ont dépassé le plafond local. Une alternative est de ne modifier que la direction du code en conservant **pour chaque vecteur sa coquille**, ses échelles et les paramètres du flux. En mode1, son co-ensemble est également conservé.

Dans ce format précis, les symboles rANS restent identiques ; les radices des rangs aussi ; la longueur des groupes de rangs est inchangée. Les ESC restent de longueur fixe. Donc :

$$\text{taille}(\text{nouveau flux})=\text{taille}(\text{ancien flux}).$$

Cette propriété est testée sur des changements arbitraires de rangs et dans les deux procédures de raffinement. Le candidat courant est toujours possible ; l’objectif de calibration ne peut augmenter lors des mouvements acceptés. **Aucune propriété correspondante sur la confirmation ne suit de cette preuve.** Les nouvelles expériences montrent précisément un sur-ajustement ; le composant est donc conservé comme outil exploratoire, désactivé par défaut.

La propriété n’est pas exportable sans preuve à un codec conditionné sur le voisinage, aux longueurs variables par rang ou à des échelles modifiées. Elle ne dit pas non plus que l’on préserve le taux d’inférence.

## 6. Finaliste C — Allocation mesurée et récupération avec acceptation sur le fichier final

### 6.1 Allocation : deux niveaux clairement séparés

`continuous_waterfill` résout un modèle séparé `Σ N_g ω_g 2^(−2R_g)` sous contraintes de débits fournies. Les coefficients historiques de `ω` sont étiquetés **a priori non mesurés**. L’optimum de ce modèle n’est pas l’optimum de la qualité du réseau.

`allocate_options` accepte des variantes avec **coûts entiers d’octets** et pertes locales fournies. Il résout exactement le problème additif fini par programmation dynamique/Pareto tant que le nombre d’états autorisé suffit. Il échoue explicitement plutôt que masquer une approximation. Les facteurs partagés doivent avoir leur coût attribué une seule fois avant cette optimisation.

Le modèle réel comporte des interactions ; une perte locale ne s’additionne pas forcément. L’approche jointe de RCO reste une option documentée [S07], non reproduite par cette DP. Après un choix local, vérifier le modèle/greffe commun, réencoder et contrôler les coûts. Les planchers d’octets Vision/MTP ne prouvent pas leur qualité.

### 6.2 Récupération continue/discrète

PV-Tuning motive l’optimisation alternée de représentations discrètes et continues [S04]. La présente livraison n’implémente pas une campagne de fine-tuning 27B. Elle livre un garde-fou d’acceptation d’artefact et des briques de correction locale.

La pénalité de débit doit être complète : `−log P(shell[,coset]) + log(cardinalité conditionnelle)` ou ESC, puis tous les coûts physiques à l’export. Garder λ fixe ne garantit pas les octets après refit des tables.

**P et V** sont évalués depuis un état accepté, et non seulement V après une étape P non contrôlée. Le snapshot est l’artefact complet : codes, normes, échelles, tables, affectations et métadonnées. Le candidat doit être décodé indépendamment, tenir la taille et respecter les sentinelles disponibles. Un refus revient à l’artefact précédent entier. Cela garantit une règle d’acceptation sur la sélection, pas la généralisation.

### 6.3 MTP, Vision, tête, cache

Le chemin MTP doit être réellement chargé, raccordé et exécuté. Le fait qu’un forward ignore ses clés peut signifier que les paramètres ne sont pas présents dans le module chargé [S13]. La préservation de distribution dépend du véritable protocole spéculatif ; l’acceptation et la latence se mesurent à part.

Pour Vision, le MLP `fc2(GELU(fc1(x)))` n’est pas un SwiGLU : une cible asymétrique de `fc2`, avec biais et vraies entrées visuelles, remplace l’idée erronée de changer seulement une dérivée. Un test texte ne valide ni Vision ni vidéo.

QERA reste optionnel [S05]. Le correcteur `lm_head`, rang64 INT8, coûte **16 220 160 octets bruts**, soit **0,1020618557 bit/poids de cette matrice**, avant échelles et métadonnées. Le duel à `+0,08 bpw` n’est pas iso-octets. Aucun correcteur n’est ajouté par défaut.

Le cache Top-128 stocke ici des **log-probabilités normalisées FP16**, 128 indices int32, et une log-masse de queue FP32 : `772 octets/position` hors métadonnées. Le total principal sur 4 194 304 positions est **3 238 002 688 octets** ; avec Top32 MTP, **4 060 086 272**. La KL agrégée exacte est une borne inférieure ; l’arrondi FP16 puis renormalisation introduit une approximation qui doit être mesurée. Le code ne nomme pas cette valeur « KL complète exacte ».

## 7. Nouvelles expériences exécutées

### 7.1 Protocole commun et limites

Python CPU, NumPy, poids et données **entièrement synthétiques**. Problème `128 → 32 → 48`, avec une couche précédente quantifiée qui fournit `Xq`, alors que le professeur utilise `Xf`. `512/384/768` positions distinctes en calibration/sélection/confirmation. Les cinq graines sont toutes publiées. Le contrôle `Qplus` cherche 12 réglages ; chaque famille de cibles en cherche aussi 12, sans toucher à la confirmation.

Toutes les variantes retenues respectent le **même plafond local** que la baseline initiale (1 650 à 1 658 octets selon la graine). Les artefacts retenus n’ont pas forcément exactement la même taille : la table donne les nombres. Ce n’est pas une égalité bit à bit de budget consommé. Les coûts de calcul ne sont pas égaux : résoudre CG ajoute du travail. La grille ne constitue pas une optimisation exhaustive de la baseline.

Les gains portent sur la NMSE de la **sortie du MLP synthétique complet**, pas sur des scores LLM. Un gain positif signifie moins d’erreur. `0` peut signifier le maintien du contrôle par la sélection, et non une expérience cachée sans effet.

| Graine | Octets Qplus | Octets asymétrique retenu | Gain confirmation asymétrique | Gain confirmation BGCC-CG |
|---:|---:|---:|---:|---:|
| 211 | 1658 | 1613 | +5.47 % | +2.24 % |
| 307 | 1605 | 1612 | -0.49 % | -4.59 % |
| 401 | 1607 | 1614 | +4.98 % | +0.00 % |
| 503 | 1619 | 1612 | +1.56 % | +0.83 % |
| 601 | 1532 | 1611 | +6.19 % | +0.00 % |

Calibration asymétrique : moyenne des gains relatifs **+3,54 %**, médiane **+4,98 %**, quatre améliorations et une régression **−0,49 %**. BGCC-CG : moyenne **−0,30 %**, deux améliorations, une régression **−4,59 %**, deux contrôles conservés. Cinq graines ne donnent pas une preuve de robustesse générale ; aucune significativité ou pourcentage de capacités n’est annoncé.

Les cinq systèmes CG atteignent le résidu relatif demandé `≤1e−8` en **195 à 211 itérations**. Le premier plafond numérique de 160 ne l’atteignait pas ; il a été relevé à512 uniquement pour satisfaire ce critère. Les résultats antérieurs sont conservés. Les choix et scores quantifiés finaux n’ont pas changé.

Le polissage discret libre réduit son objectif de calibration mais dépasse le plafond local dans les cinq cas : **aucun de ces candidats n’est accepté**. C’est une raison mesurée de ne pas séparer qualité et budget final.

### 7.2 Codage réel et modèle conditionnel

Trois distributions construites de 8 192 vecteurs chacune sont encodées et décodées dans les deux modes, soit **49 152 reconstructions de vecteurs, zéro erreur observée**. Les fichiers `.bin` et leurs empreintes sont livrés. Une seconde exécution, dans un processus neuf et sans appel à l’encodeur, redécode ces six fichiers et les compare aux sources synthétiques régénérées (`cold_decode_verification.json`). Il s’agit des mêmes 49 152 vecteurs reconstruits une seconde fois, pas de nouvelles données indépendantes. Ces vecteurs ne sont pas les anciens 314 688 vecteurs revendiqués par l’autre agent.

| Distribution synthétique | Mode0 : octets | Mode1 : octets | Économie relative du mode1 |
|---|---:|---:|---:|
| uniform_shell1 | 8 489 | 8 759 | -3.18 % |
| integer_coset_shell1 | 8 489 | 7 610 | +10.35 % |
| gaussian_E8 | 17 504 | 17 770 | -1.52 % |

Le cas entier-only favorise volontairement le conditionnement : il démontre qu’un modèle de probabilité plus fin **peut** économiser des octets, pas que les vrais poids ont cette distribution. Sur les deux autres sources, les tables plus grandes coûtent davantage. La conclusion utile est un **choix conditionnel mesuré**, pas « mode1 toujours meilleur ».

Pour la source gaussienne, le modèle de code mode0 prédit environ **2,09044 bpw**, tandis que le fichier réel utilise **2,13672 bpw**. La différence confirme concrètement pourquoi on ne doit pas remplacer les octets par l’entropie. L’échelle d’une petite expérience et ses tables locales ne permettent pas d’extrapoler un overhead global constant.

### 7.3 Polissage à coquille constante : nouvelle propriété et résultat négatif conservé

Cette piste a été formulée après observation des dépassements de taille. Elle est testée sur cinq **nouvelles graines**, sans modifier les hyperparamètres à partir des résultats de confirmation.

| Graine | Mouvements acceptés en calibration | Variation d’octets | Amélioration objectif calibration | Gain NMSE sélection | Candidat accepté |
|---:|---:|---:|---:|---:|:---:|
| 719 | 514 | 0 | +31.44 % | -39.37 % | Non |
| 809 | 525 | 0 | +33.34 % | -26.00 % | Non |
| 907 | 528 | 0 | +33.99 % | -24.16 % | Non |
| 1009 | 494 | 0 | +33.16 % | -28.36 % | Non |
| 1103 | 588 | 0 | +37.96 % | -26.20 % | Non |

Les cinq égalités de taille sont obtenues. Mais la baisse d’objectif de calibration ne se généralise pas : la NMSE de sélection se dégrade d’environ **24 à 39 %**. La règle de sélection rejette les cinq candidats et conserve le contrôle. La confirmation diagnostique est publiée, mais n’a servi à aucun choix.

**Décision :** la propriété de taille est une brique technique vérifiée ; le raffinement large ainsi configuré n’est pas un gain de fidélité. Un éventuel nouveau travail doit limiter sa liberté ou changer sa régularisation et utiliser de nouvelles données de confirmation. Ne pas le présenter comme solution gagnante à Qwen.

## 8. Budget : ce qui peut être conclu maintenant

Sur l’inventaire rapporté, le plafond complet correspond à **2,8796216 bits/paramètre**. Il inclut les éléments protégés et toutes les métadonnées. Il ne signifie pas `2,88 bits` uniformes pour les grandes matrices, ni une VRAM de10Go.

Le programme de budget fournit maintenant des scénarios conditionnels, pas un faux package final. Par exemple, sur les grandes matrices annoncées :

| Hypothèse auxiliaires | Garde | Overhead de codage supposé | Débit disponible pour le modèle de symboles |
|---:|---:|---:|---:|
| 84,067 Mo | 18,5 Mo | 0,0025 bpw | ≈2,83285 bpw |
| 100 Mo | 50 Mo | 0,02 bpw | ≈2,80168 bpw |
| 150 Mo | 100 Mo | 0,05 bpw | ≈2,74285 bpw |

Ces hypothèses sont des **exemples de sensibilité**, non des overheads mesurés ni des budgets conseillés définitivement. Elles ne sont pas les métadonnées du codec CPU `LE8CPU3` transposées au modèle entier.

L’économie historique « +623 Mo » n’est pas un gain par rapport à Q0, qui prévoyait déjà environ83Mo d’auxiliaires : elle corrigeait un format intermédiaire plus lourd. Aucun gain nouveau de623Mo n’est annoncé ici. Compresser Vision/MTP plus fortement pour financer le texte reste un arbitrage à tester sur ces fonctions.

## 9. Intégration dans les passes Lumen

| Passe | Modification proposée | État livré |
|---|---|---|
| 0 | Copies immuables, vérification des bases/normes, dtypes et interfaces | Fonctions numériques et contrats ; chargement 27B absent |
| 1 | Vraies captures, métrique explicite, séparation sources, sensibilité de sortie conditionnelle | Validation de schéma, sondes et identités ; collecte GPU non livrée |
| 2 | LDLQ corrigé ; cible asymétrique ; BGCC séparé ; émission réelle du flux de référence | CPU exécutable pour petites matrices ; moteur Lumen à raccorder |
| Allocation | Choix de variantes avec octets réels et hypothèses explicites | Water-filling de modèle et DP additive ; RCO non implémenté |
| 3 | Réglage discret/continu, acceptation de l’artefact décodé entier, rollback | Raffinements CPU et garde-fou ; entraînement global non livré |
| 4 | Décodage froid, taille, SHA-256, tests du lecteur | Codec CPU réel ; aucun export des18shards ni kernel CUDA |

Le format CPU est une référence de correction séparée, **pas un remplacement binaire transparent**. On peut porter les corrections de cible et de LDLQ dans le codec existant sans adopter ce nouveau format. Cela doit être décidé par comparaison mesurée, non par obligation de réécrire tout Lumen.

## 10. Prochaine séquence recommandée

D’abord réauditer les formules, le code et les tests fournis. Ensuite collecter un petit nombre d’entrées MLP réelles à pleine largeur, sur documents séparés et avec les versions du runtime. Comparer la meilleure référence historique réellement exécutable, la référence corrigée, puis **la cible asymétrique seule**. Introduire BGCC uniquement comme candidat distinct, après mesure de mémoire/temps et avec un contrôle bénéficiant du même effort de recherche.

Ne pas tester d’abord le paquet YAQA + GPTAQ + BGCC + ABCD + RCO + GKD en une seule fois. Un résultat attribuable d’une brique vaut davantage qu’un score combiné incompréhensible.

Les demandes à B restent `proposed`, limitées et acceptées par B selon son budget. Le présent travail n’autorise pas à détourner son sprint de reconstruction profonde. Une expérience à4k ne certifie pas le contexte maximal ; un test texte ne valide pas la Vision ; MTP doit avoir un vrai chemin d’exécution.

Les anciennes durées `2,65 h` et `24,8 h` sont retirées comme chiffres de décision. Produire les durées futures à partir d’un chronométrage incluant collecte, calculs, transferts, codage, confirmation et rapport.

## 11. Vérification et limites finales

**58 tests réussis** dans l’environnement fourni, couvrant algèbre, codec, exceptions, reconstruction, objectifs, budget et métriques. Ils n’épuisent pas les entrées possibles et ne certifient ni la sécurité d’un service exposé, ni les dtypes d’un futur kernel, ni la qualité de Qwen.

Restent nécessaires : échantillons bruts et captures authentiques ; adaptation plein format et kernels ; mesures de taille/mémoire/latence ; contrôle multi-domaines ; comparaison produit externe à périmètre et harnais comparables. Aucun gain de la littérature ni de ces jouets n’est ajouté aux prédictions historiques.

**Décision :** promouvoir les corrections d’algèbre et de traçabilité comme base de travail ; tester en premier la cible asymétrique ; conserver BGCC et le polissage comme options non validées ; ne pas déclarer Q1-v3 « recette supérieure à Q0 » avant les expériences Qwen appropriées.

## Sources

Les sources primaires sont décrites et leurs limites d’accès recensées dans `REGISTRE_PREUVES_ET_SOURCES_V3.md`. Les identités et expériences propres à cette livraison sont identifiées comme telles et ne sont pas attribuées aux auteurs externes.

[S01]: https://proceedings.mlr.press/v267/li25dn.html
[S02]: https://arxiv.org/html/2505.22988v3
[S03]: https://arxiv.org/abs/2607.07964
[S04]: https://arxiv.org/abs/2405.14852
[S05]: https://arxiv.org/abs/2410.06040
[S06]: https://proceedings.mlr.press/v235/tseng24a.html
[S07]: https://arxiv.org/html/2605.00649v2
[S08]: https://arxiv.org/abs/1311.2540
[S09]: https://github.com/rygorous/ryg_rans
[S10]: https://oeis.org/A004009
[S11]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
[S12]: https://arxiv.org/abs/2102.05426
[S13]: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
[S14]: https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
[S15]: https://huggingface.co/papers/2609.31009
