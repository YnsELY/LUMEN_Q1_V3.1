# Formic / Lumen — Attention et pistes architecturales complémentaires

**Date : 13 novembre 2026.**  
**Cible du projet :** `Qwen/Qwen3.8-27B`, révision exigée `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.  
**Budget produit :** au plus **10 000 000 000 octets complets**, texte, Vision, MTP et ressources de représentation comprises.  
**Base :** LUMEN Q1-v3.1, recette candidate et référence CPU, complétées par les mesures que Fable rapporte.  
**Nature de ce dossier :** recherche documentaire, déductions conditionnelles et petites vérifications algébriques indépendantes. Aucun poids Qwen, entraînement, benchmark ou kernel GPU n'a été exécuté pour cette étude.

Ce dossier ne remplace ni la recette ni le prompt V6. Il expose des hypothèses pour la suite ; il ne prescrit pas une architecture à l'agent et ne prolonge pas son budget.

## 1. Conclusion décisionnelle

**L'attention est une piste viable, mais « remplacer l'attention » n'est pas automatiquement le meilleur investissement pour Lumen.** Les directions à examiner en premier sont les représentations cohérentes qui traversent plusieurs blocs, la protection sélective de sous-espaces et les décompositions dont les facteurs se quantifient mieux. Une autre famille crédible consiste à construire des blocs hétérogènes plutôt que réduire uniformément toutes les couches.

Trois critères restent distincts :

1. fidélité / réussite sur les tâches après quantification et récupération ;
2. taille du package effectivement nécessaire ;
3. mémoire et temps d'exécution.

Un gain sur le cache n'est pas une économie du fichier. Une accélération ne prouve pas une amélioration des réponses. Une transformation équivalente n'ajoute pas de connaissances au BF16 ; elle peut en préserver davantage après compression.

### Priorités proposées — jugement de recherche, non probabilités de succès

| Priorité | Famille | Pourquoi l'examiner | Obstacle principal |
|---|---|---|---|
| A | Changement de base cohérent du résiduel, compensé aux interfaces | Peut améliorer plusieurs grandes matrices sans supprimer leur calcul appris | Compatibilité de toutes les branches ; différence réelle avec les Hadamard déjà utilisés |
| A | Préservation de sous-espaces avant quantification + correction résiduelle | Protection ciblée différente d'un biais constant | Facteurs supplémentaires à financer dans les mêmes octets |
| A | Factorisation sélective, puis quantification conjointe des facteurs | Preuve publiée que structure + quantification peut battre quantification seule à mémoire égale | Choix des rangs et dégradation après re-quantification |
| A/B | Réorganisation exacte des canaux SwiGLU et des frontières E8 | Ne nécessite pas de suppression de canaux | Transformations déjà prévues ; certaines permutations ne changent rien |
| B | Bases partagées entre projections / couches | Réduire les poids uniques sans imposer des sorties identiques | Récupération, interactions et comptage correct des poids partagés |
| B | Architecture hétérogène guidée par les résultats quantifiés | Éviter un taux de réduction arbitraire commun à toutes les couches | Coût de recherche et de récupération |
| B | Transformations compatibles avec les portes de l'attention et DeltaNet | Cibler les projections difficiles tout en conservant leur rôle | Portes, normes et positions interdisent certaines fusions naïves |
| C | Fusion de têtes, MLA, remplacement attention/récurrence | Transformation réelle documentée dans d'autres systèmes | Bénéfice souvent surtout sur le cache ; adaptation potentiellement importante |
| C | Mise à jour récurrente plus expressive, partage récursif profond | Axe différenciant possible à long terme | Nouveau problème d'apprentissage, calcul additionnel, transfert non démontré |

A signifie « bon premier sujet de diagnostic », pas « gain assuré ». B est conditionnel aux mesures. C mérite une étude distincte avant une campagne coûteuse.

## 2. Ce que l'on sait du travail de Fable

**Source : captures et messages transmis par l'utilisateur ; pas inspection du code ni des journaux complets.** Les essais visibles concernent des règles de pas, une variante avec biais/centrage, des réductions de largeur SwiGLU et une intention d'étudier la suppression de couches. Ils ne prouvent pas que toutes les autres pistes ont été absentes du travail.

Les valeurs rapportées sont une KL de référence E8+LDLQ autour de 0,1123, contre 0,190 sans rétroaction ; des réductions de largeur donnant 0,136 et 0,278 en BF16 ; et un candidat à biais donnant **0,133 contre 0,1123** dans la dernière capture. La lecture précédente « 0,1133 » était incorrecte. Ces valeurs ne sont pas des scores de capacités et ne certifient pas un fichier complet sous 10 Go.

Un résultat négatif de pas inspiré du Fisher ne réfute pas toute métrique de sensibilité. Un échec de biais ne réfute pas une correction dépendant des entrées. Un dommage BF16 après réduction de largeur ne démontre pas le résultat après récupération et quantification à budget égal.

La recette V3.1 impose déjà de ne pas promouvoir BGCC/polissage par défaut et de distinguer octets estimés, réellement encodés et mémoire de fonctionnement. Les nouvelles recherches doivent battre cette référence correctement identifiée, pas un arrondi simple artificiellement faible.

## 3. Identité du modèle et carte des enjeux

La configuration publique consultée indique une alternance de 48 blocs linéaires/récurrents et 16 attentions complètes, avec des MLP de largeur intermédiaire 17 408 pour un résiduel de 5 120. Les projections de requêtes et les portes, les répétitions de têtes, les normalisations et les positions doivent être lues dans l'implémentation exécutée. [S01–S03]

**Limite de vérification :** les accès directs à la configuration de la révision épinglée n'ont pas abouti dans cette recherche. La configuration `main` et le code public actuel sont des sources d'orientation ; ils ne certifient pas les fichiers locaux ni la version des kernels sur le pod. L'agent doit les confronter au snapshot local, sans changer de checkpoint.

### Ordres de grandeur calculés, non mesurés à nouveau sur les en-têtes

À partir des dimensions publiques et de l'inventaire rapporté du projet (27 781 427 952 paramètres), les grandes matrices représentent :

| Famille | Paramètres calculés | Part du total rapporté |
|---|---:|---:|
| MLP textuels, gate/up/down | 17 112 760 320 | 61,60 % |
| Grandes projections DeltaNet | 5 536 481 280 | 19,93 % |
| Grandes projections de l'attention complète, porte incluse | 1 677 721 600 | 6,04 % |
| Embeddings + tête de vocabulaire | 2 542 796 800 | 9,15 % |

Les petits tenseurs, Vision et MTP complètent le total. Ces proportions ne sont pas des pourcentages d'importance. Un petit composant peut être critique. Elles empêchent seulement de confondre une grosse économie de cache d'attention avec une grosse économie sur les poids.

## 4. Attention : possibilités réelles et restrictions

### 4.1 Transformer la représentation autour de l'attention, plutôt que son calcul

QuaRot, SpinQuant et QuIP# documentent l'intérêt de changements de représentation avant quantification. Leurs mécanismes et régimes diffèrent : des résultats poids+activations à 4 bits ne prédisent pas ceux d'E8 à environ 2,8 bits/poids. QuIP# apporte un précédent particulièrement pertinent pour les poids à faible précision, mais son codebook E8 et son format ne sont pas le codec entropique Lumen. [S04–S07]

**Hypothèse Lumen :** optimiser une base coordonnée pour les matrices qui partagent un même résiduel, au lieu de seulement ajouter un Hadamard local à chacune. Les transformations actuelles de Lumen doivent être incluses dans la comparaison ; deux rotations peuvent se recouvrir, voire s'annuler.

**Dérivation conditionnelle propre à cette étude.** Pour un résiduel `h`, une normalisation RMS sans gain `N`, un gain diagonal `Γ`, une projection d'entrée `A`, un sous-calcul quelconque `f` et une projection de sortie `B` :

```text
h_next = h + B f(A Γ N(h))
h' = R h, R orthogonale
A' = A Γ R^T ; B' = R B ; gain de la nouvelle norme = I
```

L'identité `N(Rh)=R N(h)` donne `h'_next=R h_next` en arithmétique réelle. La non-linéarité de `f` n'est pas un obstacle : elle reçoit les mêmes arguments. Cela vaut aussi si le sous-calcul contient un état, à condition de préserver toutes ses entrées et sa politique d'état.

Le graphe complet demande de raccorder chaque consommateur du résiduel, embeddings, branches de contrôle, sorties multimodales et MTP. Une norme zéro-centrée utilise son multiplicateur effectif `1+w`. Le gain peut être absorbé dans les projections, mais les arrondis BF16 doivent être revérifiés. Cette identité ne certifie pas une greffe complète sur Qwen.

**Test discriminant :** équivalence avant quantification, puis comparaison après le même E8+LDLQ et avec coûts de transformations inclus. Mesurer aussi l'absence de gain pour ne pas optimiser deux fois une symétrie déjà exploitée.

### 4.2 Attention à porte : une rotation V/O classique peut être fausse

Le chemin d'attention examiné multiplie la sortie par une porte dépendante de l'entrée avant la projection finale ; Q et K subissent aussi leurs normes et l'encodage positionnel. [S02–S03] Les recherches sur l'attention à portes montrent que ces portes peuvent participer à la qualité et à la stabilité ; leur suppression n'est pas un simple nettoyage. [S08]

**Contre-exemple algébrique.** Écrivons `y=W_o D_g o`. La rotation naïve `o'=R o`, `W_o'=W_o R^T` donne :

```text
y' = W_o R^T D_g R o
```

Il n'y a égalité générale que si `R` commute avec les portes, ou si le mécanisme de porte est transformé lui aussi. Le transformer en matrice dense dynamique peut annuler l'intérêt pratique.

En revanche, une diagonale inversible `C` commute avec `D_g` : `v'=Cv`, `W_o'=W_o C^-1` préserve cette partie du calcul. Une permutation peut également être compensée si elle réorganise aussi les coordonnées de la porte. Les têtes de requête qui partagent une tête V doivent recevoir une transformation cohérente.

**Limite importante :** une simple échelle de ligne peut déjà être absorbée par le quantifieur et ne rien apporter. Le bénéfice net se teste sur les deux projections et le stockage complet, pas sur l'esthétique d'un histogramme.

### 4.3 Co-concevoir Q et K, les valeurs et la porte

La piste structurelle la plus directe consiste à essayer une représentation factorisée ou partagée de certaines projections, tout en conservant leurs sorties et leurs opérations internes. Une base commune peut être calculée une fois, suivie de coefficients propres à chaque projection. Ce n'est pas une fusion arbitraire des têtes ni une modification de softmax.

La faisabilité s'appuie sur les familles de factorisation et de bases partagées [S13–S14], mais **leur application à ces projections Qwen et à Lumen est une hypothèse de cette étude**. Les erreurs de Q/K doivent être évaluées par leur effet sur les scores d'attention et les sorties, pas uniquement par la proximité de chaque matrice isolée. Une petite erreur de matrice peut changer le routage de l'information.

RoPE ajoute une restriction : une rotation insérée avant l'opérateur de position doit commuter avec celui-ci ou être compensée correctement. Une rotation confinée au sous-espace non rotatif passe ce test, sans résoudre à elle seule l'obstacle des gains de normalisation. Une rotation placée après les projections sans modifier leur représentation quantifiée peut surtout agir sur les activations/cache, pas sur les poids.

### 4.4 Moins de têtes KV ou conversion MLA : viable, mais pas première priorité poids

GQA documente la conversion et la récupération de modèles à partage KV ; TransMLA propose une conversion vers un cache latent. Ce sont des précédents réels de modification d'attention, pas une validation automatique pour le modèle hybride de ce projet. [S09–S10]

**Calcul de tri pour Formic :** passer de quatre à deux têtes KV, en ne réduisant que les projections K/V des 16 attentions complètes, enlèverait 83 886 080 paramètres. À trois bits/poids, cela ne représente qu'environ **31,46 Mo de payload**, avant frais et adaptation. Les économies de cache peuvent être beaucoup plus intéressantes, mais ce n'est pas la même contrainte.

Une conversion MLA peut changer d'autres dimensions et coûts ; elle exige un bilan propre. Dans un exemple du papier TransMLA, la récupération utilise six milliards de tokens. Cela illustre l'écart possible entre « transformation possible » et « petit ajustement réalisable dans le sprint », sans fixer le besoin de Formic.

## 5. DeltaNet : examiner les projections avant de remplacer la mémoire

### 5.1 Représentation des projections, convolutions et contrôles

Le code public fait intervenir convolution par canal, non-linéarité, normalisation et contrôle récurrent. [S02–S03] Les matrices amont peuvent être décomposées ou partager une base sans changer la dimension des sorties remises à ces opérations. Cette voie est moins invasive que remplacer immédiatement toute la récurrence.

**Identité conditionnelle :** multiplier la ligne de projection d'un canal par `s` et diviser son filtre de convolution par `s` conserve la sortie de convolution, donc l'entrée de la non-linéarité. La mémoire du filtre doit suivre la même transformation, ou être initialisée de manière compatible. Cette jauge est déjà évoquée dans la recette historique : l'optimiser et en mesurer l'utilité n'est pas découvrir sa simple existence.

Là encore, une échelle de ligne libre peut neutraliser tout avantage quantifié. Les bases partagées et factorisations changent davantage la représentation, avec un risque d'approximation à mesurer.

### 5.2 Base de l'état récurrent : identité utile, gain de poids non automatique

À l'intérieur du noyau delta, une rotation commune des clés/requêtes et de la dimension correspondante de l'état peut préserver le calcul. Pour `S` de forme clés × valeurs :

```text
S <- alpha S
S <- S + k [beta (v - S^T k)]^T
output = S^T q
```

Avec `q'=R q`, `k'=R k`, `S'=R S`, `R` orthogonale, les sorties sont identiques en arithmétique réelle. [Mécanisme de départ : S11 ; démonstration de changement de base : cette étude.]

Mais l'équivalence du noyau ne signifie pas que `R` puisse être absorbée à travers toutes les non-linéarités/convolutions amont. Une rotation ajoutée seulement en ligne peut coûter du calcul sans améliorer la quantification des grandes matrices.

**Décision :** intéressante pour comprendre les degrés de liberté ; secondaire pour les 10 Go tant que le lien avec les poids quantifiés n'est pas établi. Les travaux sur la quantification ou le faible rang des états persistants traitent d'abord la mémoire d'exécution. [S26–S27]

### 5.3 Reconstruire la règle d'écriture : direction profonde, plus risquée

DeltaProduct remplace la mise à jour élémentaire par un produit de transformations de Householder, avec un compromis expressivité/calcul. La publication montre des résultats dans ses propres modèles et tâches. [S12]

Pour Formic, cela suggère une recherche sur des écritures supplémentaires sélectives ou des sous-espaces de mémoire, avec réutilisation du calcul d'origine et adaptation. **Ce n'est pas une conversion déjà démontrée sur Qwen, ni une garantie de mieux préserver ses capacités.** Les poids additionnels et le calcul par token doivent être comptés. L'initialisation qui conserve la fonction de départ doit être construite, pas présumée.

Ce sujet appartient à une deuxième campagne, à moins qu'un diagnostic du modèle quantifié révèle un problème de mémoire auquel il répond directement.

## 6. Grandes alternatives au retrait uniforme des canaux

### 6.1 Factoriser, puis quantifier : un résultat publié particulièrement pertinent

Le tableau 7 de SVD-LLM V2 compare, sur LLaMA-7B et WikiText-2, à **2,8 GB de mémoire de poids** :

| Méthode | Perplexité, plus bas est meilleur |
|---|---:|
| GPTQ 3 bits | 16,28 |
| SVD-LLM V2 seule | 119 |
| SVD-LLM V2 + GPTQ 4 bits | 9,97 |

Le tableau a été consulté visuellement dans le PDF. Ce résultat montre qu'une combinaison peut battre la quantification seule à mémoire égale, alors que la réduction structurelle poussée seule est mauvaise. Il ne fournit aucune prévision de KL ou de benchmark pour Formic. [S13]

**Hypothèse Lumen :** remplacer seulement certaines grandes matrices par `AB`, puis répartir les bits entre les facteurs. Le coût brut devient `r(m+n)b/8` pour un rang `r` et une précision commune `b`, à comparer à `mn*b0/8`, plus tous les frais.

Le choix du rang doit dépendre des sorties et des octets réels, non d'une norme de poids seule. Les matrices intactes restent disponibles pour les composants peu compressibles.

### 6.2 Optimiser la base latente des facteurs

**Dérivation propre :** pour toute matrice inversible `T` de taille `r`, `AB=(AT)(T^-1 B)`. La factorisation n'a donc pas une représentation unique. On peut rechercher une représentation favorable à l'E8 et aux échelles des deux facteurs, sans changer le produit continu retenu.

Cela combine une reconstruction approximative (choisir `r`) et une reparamétrisation exacte de cette reconstruction. Les deux dommages sont distincts : erreur de factorisation et erreur d'encodage des facteurs. Il faut les mesurer ensemble ; ils ne s'additionnent pas nécessairement.

Les transformations orthogonales ont une meilleure maîtrise du conditionnement, mais ne sont pas forcément les seules admissibles. Une transformation très mal conditionnée peut amplifier les arrondis.

### 6.3 Préserver avant de quantifier : une protection différente d'un biais

QERA étudie des corrections de faible rang tenant compte des activations. SRR (*Preserve-Then-Quantize*) partage le budget de rang entre protection préalable et réparation après quantification. [S15–S16]

**Hypothèse Lumen :** retirer une petite composante importante du bloc que l'on va fortement quantifier, conserver cette composante avec plus de fidélité, puis quantifier le résidu et éventuellement en corriger une partie :

```text
W ≈ L_protege R_protege + Q_E8(W - L_protege R_protege) + L_correction R_correction
```

Un biais ne dépend pas de l'entrée. Une branche `LRx` peut corriger une erreur qui varie avec celle-ci : l'échec d'une correction constante ne tranche donc pas cette piste.

Tout facteur et tout indice doit entrer dans les 10 Go. Le concurrent pertinent est la même dépense d'octets en précision E8 supplémentaire. Les hypothèses du critère analytique SRR sur le bruit doivent être revérifiées après LDLQ : il peut produire des erreurs structurées. Pas d'importation automatique d'une règle de rang d'un autre quantifieur.

### 6.4 Bases partagées entre couches, sans partager leurs états

Basis Sharing représente des matrices de plusieurs couches par des bases communes et des coefficients privés. [S14] Pour Formic, l'agent peut aussi chercher une base commune à plusieurs projections d'un bloc, ce qui est une extension à tester plutôt qu'un résultat repris du papier.

Il faut compter la base une seule fois, mais tous les coefficients, facteurs privés et adaptations. Quantifier cette base modifie plusieurs couches à la fois : une faible erreur individuelle peut devenir corrélée à travers le réseau.

**Partager des poids n'autorise pas à partager les caches ou les états de couches distinctes.** Les transformations doivent respecter leurs interfaces, et chaque couche conserve son propre état sauf hypothèse supplémentaire évaluée.

### 6.5 Réorganiser SwiGLU sans supprimer ses neurones

Pour `f(x)=D[SiLU(Gx) ⊙ Ux]`, une permutation `P` et une diagonale inversible `S` donnent la transformation exacte :

```text
G' = P G
U' = S P U
D' = D P^T S^-1
```

On ne fait pas traverser une rotation générale ou une échelle à SiLU : sa non-linéarité n'est pas homogène. La permutation, en revanche, réordonne ses coordonnées.

L'intérêt possible pour Lumen vient surtout des **frontières des groupes E8**, des colonnes de la projection de sortie et des contraintes d'échelle. DuQuant et RPTQ motivent l'intérêt de groupements ou réordonnancements, principalement dans leurs régimes de quantification d'activations. Le gain sur E8 poids-seuls reste à mesurer. [S17–S18]

**Test nul crucial :** E8 est invariant par permutation des huit coordonnées et par inversion d'un nombre pair de signes. Avec une métrique, un coût sphérique et une recherche également transformés, une simple permutation interne au même bloc ne crée pas un gain géométrique. Chercher une réorganisation entre blocs ou une base non redondante, et examiner les effets de l'heuristique pratique séparément.

### 6.6 Protéger quelques valeurs extrêmes ou scinder des canaux

SpQR fournit un précédent dense quantifié + exceptions plus précises ; les indices de ces exceptions coûtent des octets. OCS fournit un précédent de duplication de canaux avec compensation fonctionnelle. [S19–S20]

Pour SwiGLU, on peut théoriquement dupliquer un canal gate/up et répartir sa colonne de sortie entre les copies : leurs contributions se somment comme avant. C'est une identité, pas une preuve qu'E8 en tirera profit. Les copies, le padding, les échelles et leurs codes peuvent coûter plus que le gain. Un quantifieur par ligne peut déjà absorber certains redimensionnements.

Cette famille mérite un essai uniquement si les diagnostics montrent qu'un petit sous-ensemble domine les erreurs restantes après le vrai LDLQ.

## 7. Réorganisation globale : architectures hétérogènes et profondeur partagée

### 7.1 Composer des blocs différents plutôt qu'appliquer le même taux partout

Puzzle fournit un exemple de recherche d'architecture dérivée par distillation de blocs et sélection sous contraintes. Minitron montre l'intérêt de la récupération dans l'évaluation d'architectures réduites. [S21–S22]

**Application proposée :** constituer quelques variantes réellement quantifiées des blocs importants, puis sélectionner une composition sous budget, en la validant dans le réseau complet. On peut conserver une couche intacte, factoriser une autre et réduire la largeur d'une troisième.

La mesure décisive vient après récupération comparable et réinvestissement des octets. Les KL locales ne sont pas additives ; une somme de coûts locaux ne certifie pas la meilleure architecture globale. Une sélection approximative doit être suivie de contrôles d'interaction.

### 7.2 Réduire le résiduel après rotation

SliceGPT réduit la dimension des représentations via une réorganisation et une suppression structurée de lignes/colonnes. [S23] Ce n'est pas le même essai que retirer directement un pourcentage de canaux MLP.

La piste touche de nombreux poids, y compris les interfaces de vocabulaire. Pour le modèle multimodal du projet, les normes, branches et raccordements Vision/MTP rendent l'opération plus délicate. Le classement des dimensions doit tenir compte des tâches, pas seulement de leur énergie sur un corpus étroit.

### 7.3 Partage profond et récurrence en profondeur

Les *Relaxed Recursive Transformers* étudient un partage de paramètres avec adaptations par couche. [S24] Cela motive un axe de co-conception : moins de poids uniques, réutilisés à différentes profondeurs, puis meilleure précision des poids conservés.

Cette reconstruction n'est pas un simple changement de format. Répéter arbitrairement un bloc préentraîné n'est pas équivalent au réseau d'origine. Le coût de calcul, les positions et les états propres à chaque passage doivent être décrits. Une boucle supplémentaire peut améliorer certaines tâches mais ne respecte pas une comparaison « même calcul » si elle ajoute des opérations. Aucun résultat de ce type n'est ici établi pour Lumen.

### 7.4 Remplacer certaines attentions par d'autres mécanismes

LoLCATs et MOHAWK montrent des transferts vers des calculs de séquence différents à partir de modèles préentraînés. [S25, S28] Ces travaux justifient la possibilité de conversions profondes, mais pas leur supériorité pour les 10 Go du présent projet.

Le modèle étant déjà hybride, supprimer ses attentions complètes restantes peut détériorer des fonctions complémentaires de récupération d'information. Examiner d'abord ce qu'elles apportent sur les tâches réelles ; ne pas déduire leur redondance de leur faible nombre.

## 8. Embeddings, tête, Vision, MTP et autres frontières

Le test de prédiction de la tête depuis les embeddings reste suspendu dans le projet. Cela ne ferme pas une factorisation **séparée** de la tête, une petite correction activée par les vrais états cachés ou une base globale coordonnée. Mais il ne faut pas rouvrir l'ancienne hypothèse sous un autre nom, ni réduire le vocabulaire et présenter la perte de langues comme une compression gratuite.

Vision et MTP font partie du coût complet. Une meilleure représentation de leurs projections peut libérer des octets, mais un plancher de stockage ne garantit pas leurs capacités. Le module visuel doit être testé sur des entrées visuelles ; le MTP doit être réellement chargé et évalué. La neutralité d'un brouillon spéculatif exige un protocole de vérification correct ; elle ne se déduit pas du nom du module.

Les normalisations, amplitudes résiduelles, portes et dtypes peuvent présenter des degrés de liberté ou des défauts d'implémentation. Les corrections qui rétablissent le calcul voulu sont indispensables, mais ne sont pas automatiquement une innovation architecturale. La correction de softmax/cross-entropie déjà évoquée dans le projet relève surtout de l'apprentissage, pas d'une économie de paramètres.

## 9. Ce qu'il vaut mieux ne pas prioriser pour cet objectif

| Proposition | Pourquoi elle ne répond pas directement aux 10 Go |
|---|---|
| Remplacer seulement un kernel par un kernel plus rapide | Gain d'exécution ; mêmes poids et fonction attendue |
| Quantifier le KV cache ou l'état récurrent | Réduit la mémoire des requêtes, pas directement le package |
| Remplacer systématiquement toute attention par une mémoire linéaire | Transfert et récupération risqués ; modèle déjà hybride |
| Transformer le dense en nombreux experts sans compression des poids uniques | Réduit éventuellement le calcul actif, mais tous les experts doivent être stockés |
| Ajouter des boucles de raisonnement internes ou plus de tirages | Peut changer le calcul consommé ; ne prouve pas une meilleure compression à protocole égal |
| Ajouter un Hadamard sans vérifier celui déjà utilisé | Risque de redondance ou d'annulation |
| Choisir une largeur uniquement à partir de son score BF16 brut | Ignore récupération et réinvestissement de précision |
| Relancer des recherches sur un biais constant déjà battu sans nouvelle hypothèse | Faible information supplémentaire |

Minima et les travaux sur les états quantifiés examinent des perturbations différentes. Leurs résultats ne justifient ni « DeltaNet dérive forcément » ni « DeltaNet est toujours robuste sous 3 bits ». [S26–S29]

## 10. Protocole minimal pour départager les pistes

**Avant une campagne :** figer le code et le contrôle réellement utilisés par Fable. Examiner si le débit 2,821 est une entropie, un payload, un coût de format ou un coût complet. Documenter les modules non quantifiés et les échelles. Cela peut changer l'espace disponible pour les candidats.

Pour une reparamétrisation exacte, vérifier l'identité algébrique puis le résidu numérique sur le vrai bloc, avec caches et positions pertinents. Pour une reconstruction approximative, documenter la baisse avant quantification, l'effort de récupération et le résultat après compression. Aucun score BF16 n'est présenté comme une borne mathématique sur la KL finale.

Les expériences doivent pouvoir distinguer :

| Architecture | Recette de contrôle | Recette adaptée |
|---|---|---|
| Originale | A0Q0 | A0Q1 |
| Candidate | A1Q0 si compatible | A1Q1 |

La récupération bénéficie d'un budget comparable pour les contrôles. Les variantes sont gelées avant confirmation ; tout jeu ayant servi à réorienter une méthode devient un jeu de sélection. Les sorties sont comparées sur les mêmes préfixes. Ajouter un test de récupération d'information plus long est nécessaire avant une conclusion sur la mémoire ou les positions ; des textes de 1 024 tokens ne valident pas 256k.

**Plus petite expérience informative :** une poignée de blocs choisis par diagnostic, en conservant toutes les contributions nécessaires, puis greffe dans le modèle pour mesurer la propagation de l'erreur. Le choix des blocs, rangs et étapes appartient à l'agent. Ne pas déployer toutes les propositions simultanément.

Pour chaque candidat, relever : octets des facteurs/transformations, erreur de bloc, KL de greffe, résultats par domaine disponible, coût de récupération, mémoire et temps. Le coût final devra être fermé par un package encodé, pas par une multiplication nominale des paramètres par un nombre de bits.

## 11. Vérifications réalisées spécifiquement pour ce dossier

`verification_algebrique.py` exécute **10 contrôles float64** sur de petites matrices aléatoires, ainsi que deux contre-exemples. Les contrôles vérifient notamment les changements de base résiduels, la reparamétrisation SwiGLU, la diagonale compatible avec une porte, la rotation du noyau delta, la compensation projection/convolution et la symétrie E8.

Toutes les identités testées donnent des résidus inférieurs à 2e-15 dans cette exécution, ou zéro pour les contrôles discrets. Une rotation dense V/O à travers une porte non transformée produit, sur l'exemple choisi, une erreur relative non nulle d'environ 0,65. **Ce nombre est un contre-exemple, pas une mesure d'erreur Qwen.** Une rotation arbitraire ne commute pas non plus avec l'exemple RoPE.

```bash
python verification_algebrique.py --output resultats_algebriques_reexecutes.json
```

Dépendance : NumPy. Les calculs de paramètres et l'exemple d'économie K/V sont inclus. Ces tests ne cherchent ni une architecture optimale ni un gain quantifié sur le modèle réel. Ils ne remplacent pas une preuve et un audit de graphe de bout en bout.

## 12. Références primaires et limites de transfert

Les liens suivants sont des sources de mécanismes, pas des validations de Formic. Les versions évolutives doivent être épinglées dans toute expérience. Des papiers consultés sont des prépublications ; leur existence n'est pas une réplication indépendante.

| ID | Source primaire | Usage et limite |
|---|---|---|
| S01 | [Configuration Qwen publique](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json) | Dimensions/organisation publique ; main ne certifie pas le snapshot local. |
| S02 | [Code Transformers Qwen — modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | Ordre des opérations et portes ; code évolutif, runtime local à contrôler. |
| S03 | [Code Transformers Qwen — modular](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modular_qwen3_5.py) | Définition et héritages des blocs ; ne pas confondre valeurs par défaut et configuration du checkpoint. |
| S04 | [QuaRot](https://arxiv.org/abs/2404.00456) | Réorganisations compatibles avec la quantification ; régime non identique à Lumen. |
| S05 | [SpinQuant](https://arxiv.org/abs/2405.16406) | Rotations apprises ; aucune garantie de transfert aux portes et au codec du projet. |
| S06 | [QuIP#](https://proceedings.mlr.press/v235/tseng24a.html) | Poids à faible précision, Hadamard et E8 ; codebook/format différents. |
| S07 | [FlatQuant](https://arxiv.org/abs/2410.09426) | Transformations affines apprises ; résultats principalement poids/activations à 4 bits. |
| S08 | [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) | Rôle des portes ; ne prouve pas qu'on puisse les retirer après préentraînement. |
| S09 | [GQA](https://arxiv.org/abs/2305.13245) | Partage KV et adaptation ; le modèle du projet est déjà à partage KV. |
| S10 | [TransMLA](https://arxiv.org/html/2502.07864) | Conversion vers cache latent ; bénéfice cache différent du budget de poids. |
| S11 | [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | Structure et motivation de la récurrence ; pas un résultat Lumen. |
| S12 | [DeltaProduct](https://arxiv.org/abs/2502.10297) | Mise à jour récurrente plus expressive ; conversion du checkpoint non démontrée. |
| S13 | [SVD-LLM V2](https://arxiv.org/abs/2503.12340) — [PDF, tableau 7](https://arxiv.org/pdf/2503.12340) | Précédent direct factorisation + quantification sous mémoire égale ; autre modèle/métrique. |
| S14 | [Basis Sharing](https://arxiv.org/abs/2410.03765) | Bases inter-couches ; corrections et interactions à réévaluer avec Lumen. |
| S15 | [QERA](https://arxiv.org/abs/2410.06040) | Reconstruction d'erreur pondérée par les activations ; facteurs à financer. |
| S16 | [Preserve-Then-Quantize / SRR](https://arxiv.org/html/2602.02001) | Partage du rang entre protection et correction ; hypothèses de bruit à contrôler avec LDLQ. |
| S17 | [DuQuant](https://arxiv.org/abs/2406.01721) | Rotations et permutations ; bénéfices activations non transposables sans test. |
| S18 | [RPTQ](https://arxiv.org/abs/2304.01089) | Réorganisation par groupes ; même réserve. |
| S19 | [SpQR](https://arxiv.org/abs/2306.03078) | Exceptions plus précises ; coût des indices et accès. |
| S20 | [Outlier Channel Splitting](https://proceedings.mlr.press/v97/zhao19c.html) | Scission fonctionnellement compensée ; preuve sur d'autres réseaux/régimes. |
| S21 | [Puzzle](https://proceedings.mlr.press/v267/bercovich25a.html) | Recherche par blocs distillés ; objectif à adapter aux octets Lumen. |
| S22 | [Minitron](https://arxiv.org/abs/2407.14679) | Réduction structurelle et récupération ; pas une campagne comparable au petit sprint du projet. |
| S23 | [SliceGPT](https://arxiv.org/abs/2401.15024) | Réduction du résiduel par réorganisation ; interfaces hybrides à adapter. |
| S24 | [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672) | Partage profond avec adaptations ; états et calcul à comptabiliser. |
| S25 | [LoLCATs](https://arxiv.org/abs/2410.10254) | Transfert d'attention et récupération ; modèle cible déjà hybride. |
| S26 | [Preserving Updates in Low-Bit Recurrent State Quantization](https://arxiv.org/abs/2610.15709) | État persistant quantifié, pas directement les poids. |
| S27 | [SinkState](https://arxiv.org/abs/2611.14246) | Faible rang d'état récurrent ; prépublication récente, mémoire d'exécution. |
| S28 | [Transformers to SSMs / MOHAWK](https://arxiv.org/abs/2408.10189) | Conversion profonde par distillation ; pas un bénéfice Lumen établi. |
| S29 | [Minima](https://arxiv.org/abs/2609.04098) | Stabilité observée dans un régime NVFP4 différent ; ni garantie ni réfutation pour sub-3-bit. |

**Sources internes :** `RECETTE_MAX_PROPOSEE_V3_1.md`, `PROMPT_AGENT_REVERSE_ENGINEERING_MAX_V6.md` et captures de progression de Fable. Aucun de ces messages ne remplace les expériences brutes ni le rapport final de l'agent.

## Conclusion

La recherche visible n'a pas fermé l'espace architectural. Les échecs du biais et des réductions uniformes doivent servir à mieux choisir les prochaines expériences, pas à déclarer Qwen optimal sous Lumen. La proposition prioritaire est d'examiner une représentation cohérente non redondante avec la recette actuelle, puis une décomposition sélective ou un sous-espace protégé, chacun confronté à une alternative de précision simple au même coût. Les reconstructions de mémoire et d'attention restent ouvertes, mais leur intérêt doit être démontré sur la contrainte réellement visée.
