# LUMEN Q1-v3.1 — Analyse de la contre-vérification et correctifs exécutés

**Date de cette livraison : 12 novembre 2026.**  
**Nature : référence CPU corrigée et tests reproductibles. Aucun Qwen complet exécuté.**  
**Version de travail : `LUMEN-Max-Q1-v3.1-ref`, distincte de Q0 et de l’architecture de l’agent B.**

## 1. Verdict

La contre-vérification est majoritairement juste. J’ai reproduit ses principaux défauts logiciels, ainsi que ses vérifications positives de l’algèbre et du décodage. Elle identifie de vrais problèmes de robustesse, de stabilité numérique et de comparaison expérimentale. Elle ne réfute pas les équations centrales du quantifieur et de BGCC.

Elle comporte cependant plusieurs conclusions à corriger ou à nuancer : le fichier annoncé absent est présent dans l’archive de cette conversation ; « je n’ai pas pu vérifier une source » n’est pas une affirmation que cette source n’existe pas ; quelques règles proposées sur les captures et les coûts ne sont pas universelles. Son patch est utile mais insuffisant pour couvrir tous les cas qu’il prétend fermer.

**Décision : intégrer les correctifs vérifiés dans cette référence CPU ; garder Q0 intact ; ne pas annoncer une compression complète à 10 Go ou une amélioration des capacités de Qwen.**

## 2. Ce qui a été effectivement exécuté

- Extraction sûre de l’archive V3 originale dans un répertoire séparé. SHA-256 : `619ce07bd2055cec913d4f8116e4d0b31ca3ea05fb023b7b46b56611c7c27e73`.
- Vérification de son manifeste : **64 entrées vérifiées**, 65 fichiers ZIP en comptant le manifeste. Aucun fichier manquant dans cet artefact.
- Réexécution de la suite reçue sur cette V3 : **11 groupes sur 12 reproduisent le constat annoncé** ; le test qui exige que le prompt manque échoue, parce que ce fichier est présent. Les tests passent ici en recherchant les anciens bugs : ce n’est pas une certification du code défaillant.
- Application du patch reçu sur une copie. Les 58 tests historiques passent ; des contre-tests supplémentaires révèlent encore des lacunes.
- Corrections supplémentaires, puis **107 tests positifs réussis** : les anciens défauts doivent être absents, pas seulement observables.
- Décodage à froid de **49 152 vecteurs** répartis sur six flux : aucune erreur observée. Le lecteur indépendant fourni par le contre-auditeur est aussi utilisé dans la suite et relit les flux, les racines, les échappements et des cas limites.
- Cinq nouvelles expériences synthétiques, avec une règle empêchant le candidat d’obtenir plus d’octets que le contrôle choisi. Les blobs retenus et témoins sont fournis.
- Recalcul des scénarios budgétaires et des coûts des correcteurs.

L’environnement exécuté ici est Python 3.13.5 / NumPy 2.3.5 sur Linux CPU, avec un thread BLAS pour les essais. La réexécution Windows est rapportée par le contre-auditeur ; je ne prétends pas avoir testé cette nouvelle version sur Windows.

Preuves : `results/reaudit/`, `results/verification.json`, `results/tests_final.log`, `results/cold_decode_verification.json`, `results/matched_budget/`. Les versions historiques sont rangées dans `historique/` et ne sont pas des résultats de v3.1.

## 3. Les erreurs confirmées et réparées

### 3.1 Métriques KL et caches Top-K

Les IDs dupliqués créaient une partition invalide du vocabulaire, pouvant donner une KL négative. Les IDs sont désormais vérifiés : type entier, bornes, forme et unicité par ligne.

L’oracle `exact_grouped_kl` utilisait des probabilités exponentiées et un plancher `1e-300`. Le cas enseignant `[0,0,-50]`, élève `[0,-800,-800]` donnait environ 344,69 nats au lieu de **399,30685282 nats**. La réalisation en domaine logarithmique reproduit maintenant la valeur attendue sans ce plancher arbitraire.

Les masques `-inf` sont acceptés comme probabilités exactement nulles ; une ligne entièrement masquée, un `NaN` ou un `+inf` est rejeté. Une masse professeur positive contre un support élève nul donne `+inf`, et non une valeur tronquée. Les petites valeurs négatives dues à l’arrondi sont bornées à zéro seulement dans une tolérance numérique explicite ; une négativité significative provoque une erreur.

**Le patch reçu ne suffisait pas ici.** Il proposait de tronquer les log-probabilités à -65504 et de borner toute KL négative à zéro. La version livrée évite ce masquage : les valeurs finies non représentables sont promues vers FP32/FP64, ou rejetées selon l’option choisie. Le nombre d’octets des tableaux effectivement retournés est enregistré.

L’arrondi FP16 conserve un biais : la KL du professeur exact contre son cache arrondi peut être légèrement positive et inverser un classement très fin. Ce n’est pas une erreur que l’on peut effacer par un clamp. Un cache FP64 et l’oracle en domaine logarithmique servent de contrôles ; le biais doit être mesuré sur les candidats utiles.

### 3.2 Algèbre et gradient conjugué

`checked_spd` utilise désormais la partie symétrique `0.5*A + 0.5*A.T` après le contrôle de tolérance, puis vérifie qu’elle est définie positive. Une matrice dont la partie symétrique est indéfinie est rejetée ; elle n’est pas artificiellement réparée en déplaçant ses valeurs propres. Cela aligne la factorisation, les gradients et les objectifs sur la même matrice.

Le solveur PCG ne termine plus à tort lorsque seul son résidu récursif passe le seuil. Il vérifie le vrai résidu, le rafraîchit et continue dans le budget restant.

**Lacune supplémentaire dans le patch reçu :** normaliser par `np.linalg.norm(rhs)` ne résout pas le sous-flux de cette norme elle-même. Sur l’identité et `rhs=[1e-200]`, le patch pouvait encore retourner zéro et `converged=True`. La v3.1 utilise l’amplitude maximale du second membre avant de calculer une norme stable. Les tests couvrent notamment des seconds membres de 1e-160, 1e-200, 1e-300 et 1e200.

Cela ne garantit pas la convergence de tout système mal conditionné. Dans le cas difficile fourni par le contre-auditeur, cette exécution peut atteindre les 200 itérations avec un vrai résidu supérieur à 1e-9 : elle retourne alors honnêtement `converged=False`, sans promouvoir la cible. Même un solveur dense en float64 a un résidu non négligeable sur cette instance. L’engagement correct est de vérifier le résidu et de ne pas accepter une solution non convergée, pas de promettre une tolérance inaccessible dans chaque environnement.

### 3.3 Codec, échelles et métadonnées

L’encodeur refuse les matrices vides que le décodeur ne peut pas lire. Les échelles doivent être positives, finies et déjà représentables en FP16 : elles ne sont plus silencieusement réarrondies lors de l’export. Un appelant qui veut les arrondir doit le faire avant la sélection des codes et réévaluer les poids correspondants.

Les types entiers, booléens, bornes et fréquences sont contrôlés. L’asymétrie entre +1e8 et -1e8 de la recherche géométrique est corrigée. L’indexation des coquilles reste une référence bornée ; les grands codes passent par ESC ou sont explicitement rejetés selon le contrat.

Le dépaquetage des rangs utilise l’arbre proposé, au lieu des divisions successives sur tout le groupe. Cela ne transforme pas cette réalisation Python en kernel GPU : elle utilise toujours des grands entiers, de la mémoire et du calcul CPU. Aucune nouvelle promesse de débit n’en découle.

**Correction non achevée par le patch reçu :** un polissage pouvait réencoder avec `group_size=128` alors que l’artefact initial en utilisait un autre. La réalisation lit désormais la configuration de l’artefact, vérifie sa concordance avec les tableaux en mémoire et conserve le groupement. Les tests couvrent un groupement non standard, y compris pour le polissage SwiGLU.

Les métadonnées disent correctement si une table a été réestimée. Une baisse de l’objectif à table fixe ne garantit pas une baisse des octets après réestimation : ce comportement était déjà déclaré comme limite, il demeure contrôlé à l’export.

### 3.4 Garde-fou d’acceptation

Les plafonds non entiers, `NaN`, les artefacts vides et les sentinelles non finies sont refusés. Un gain de taille à score strictement identique peut être retenu sans relâcher les autres contraintes. Les scores/sentinelles acceptés sont copiés dans un état immuable pour éviter leur modification après la décision.

`accept_decoded` reste un comparateur de scores dont l’appelant est responsable, sauf si un validateur lui est fourni. **La nouvelle fonction `compare_matrix_artifacts` décode elle-même les deux blobs LMAT, évalue leurs poids et applique la décision.** Elle évite de prendre des scores pré-export pour des scores du fichier livré. Cela est fonctionnel pour les matrices CPU de référence, pas pour un checkpoint multimodal complet.

### 3.5 Budgets

Les planchers du solveur continu sont stricts ; les dépassements numériques et les types ambigus sont rejetés. L’allocation discrète continue d’être exacte pour la table additive fournie seulement. Elle ne mesure pas les interactions du réseau et n’est pas rebaptisée RCO.

Le scénario Mini texte reprend ses propres auxiliaires, **80 529 024 octets** au lieu de ceux du modèle complet. Les scénarios réservent aussi la borne supérieure d’alignement des petits tenseurs protégés. Ce choix est conservateur, pas un comptage exact des paddings des shards non lus.

Le correcteur de tête rang 64 INT8 coûte **16 220 160 octets bruts**. Avec le schéma explicite « une échelle FP16 par ligne de chacun des deux facteurs + un descripteur 128 octets », il coûte **16 717 056 octets**, soit environ **0,10519 bit par poids de la tête**. L’affirmation du contre-audit selon laquelle ces échelles seraient un minimum universel est trop forte : d’autres groupements ou granularités ont d’autres coûts. Le budget précise maintenant le schéma choisi.

### 3.6 Captures et traçabilité

Le validateur de capture rejette les tableaux vides, les types non flottants, les placeholders, les largeurs fractionnaires et les partitions de sources mal formées. Le schéma 2 sépare les **identités de documents** des **identités de source par position capturée** ; leurs longueurs ne sont pas supposées égales. Il gère des captures d’entrées seules, linéaires avec cibles et poids, ou SwiGLU avec ses trois matrices.

Deux corrections d’interprétation sont importantes : `X_fp == X_q` peut être légitime au début du réseau ou avec un préfixe intact ; ce n’est pas automatiquement une capture invalide. Un checksum et des métadonnées ne prouvent pas que le vrai modèle a été exécuté. Le validateur retourne toujours `provenance_verified=False`.

Le manifeste de livraison vérifie désormais les fichiers manquants, modifiés, surnuméraires, doublons et chemins non canoniques. Les chemins sérialisés sont en notation POSIX. La commande de contre-vérification travaille sur une copie pour ne pas modifier les résultats livrés.

## 4. Ce qui est incorrect ou excessif dans la contre-vérification

### 4.1 Le prompt n’est pas absent de l’archive disponible ici

Le fichier `PROMPT_REVERIFICATION_V3.md` est présent, fait **6 983 octets** et a exactement l’empreinte annoncée :

`da659a4a2c0373a5441bf5c9cc21f339013d92987074a3e19f2e04a79ac3815a`.

Le manifeste original passe. Cela peut s’expliquer par un dossier de transfert/extraction différent chez le contre-auditeur, mais je ne peux pas en déterminer la cause. Sa suite encode l’absence comme résultat obligatoire : il s’agit d’une hypothèse sur son paquet, pas d’une propriété du code. **D-18 est donc réfuté pour le fichier manquant dans cet artefact précis ; les remarques de portabilité et de couverture du manifeste restent valables.**

### 4.2 Accès aux sources et bibliographie

Le titre de G²PTQ était incorrect : je le corrige en *G^2PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation*. Le résultat primaire indexé mentionne le dépôt des auteurs. Je ne prétends pas avoir lu le code ou vérifié sa licence à partir du seul titre.

En revanche, « texte non vérifié ici » ou « révision non relue dans cet environnement » ne signifie pas « indisponible publiquement ». L’accès d’un autre agent ne réfute pas une limite historique d’accès. Mes propres tentatives directes échouent encore sur certains endpoints.

La réponse de l’API officielle et la page des fichiers consultées via le navigateur renvoient un HEAD `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, pas `37561816...`. Ces réponses peuvent être mises en cache ou provenir d’un état différent : je n’en déduis pas que la révision épinglée est inexistante. **Ni HEAD ni main ne doivent remplacer la révision du projet sans contrôle local.** Les reçus et limites d’accès sont conservés dans `sources/SOURCES_REVERIFIEES_V3_1.json` et `results/reaudit/source_fetch_receipts.json`.

La date « 16 avril 2026 » inscrite en tête du contre-audit est incohérente avec l’audit d’une livraison du 12 novembre 2026 et les sources de septembre qu’il cite. Je la signale comme anomalie de traçabilité, sans en tirer une conclusion sur la justesse des contre-exemples.

### 4.3 Interprétation des expériences

Un plafond commun est une comparaison recevable pour le choix d’un produit sous budget. Il ne suffit pas à attribuer un gain au seul algorithme lorsque les tailles effectivement consommées diffèrent. Le contre-audit a raison de demander une comparaison plus discriminante.

En revanche, appliquer `D(R) ∝ 4^-R` pour dire que 79 octets « expliquent à eux seuls » un gain n’est pas une démonstration dans ce régime. Cette loi n’a pas été validée sur ce petit MLP et ce format. J’accepte donc le diagnostic de **facteur confondant**, pas l’explication causale chiffrée.

Les tests KL défaillants ne sont pas utilisés dans les expériences de NMSE publiées. Les cas extrêmes reproduits sont sérieux pour l’intégration ; ils ne signifient pas que toutes les valeurs des cinq essais ordinaires étaient numériquement fausses. Le rapport doit distinguer ce qui change les résultats existants de ce qui empêche un futur usage fiable.

### 4.4 Le patch reçu ne fermait pas tout

Il n’inclut pas la restauration supposée du prompt, la révision des sources, une comparaison à octets contrôlés, le nouveau contrat de capture ou la correction du groupement après polissage. Il laisse aussi le sous-flux du normb, les masques -inf et certaines limites de validation. « 58 tests passent » n’équivaut donc pas à « tous les bugs ont été éliminés ». La v3.1 ajoute des contre-tests pour ces points plutôt que de recopier cette conclusion.

## 5. Nouvelle comparaison, sans octets supplémentaires

Les cinq nouvelles graines ont été fixées avant l’exécution : **2003, 2011, 2027, 2029, 2039**. Le banc reste synthétique, dimensions `32 × 128 → 48`, sans Hadamard, avec porte et projection de sortie identiques entre méthodes. La taille indiquée ci-dessous est celle de la matrice `up` quantifiée, pas du MLP complet.

Les deux méthodes disposent de la même grille de 14 combinaisons de pas/pénalité, incluant le pas 0,85 manquant dans l’ancienne grille du contrôle. Le temps du calcul de cible fait partie du coût ; le temps de calcul n’est pas égalisé artificiellement. Les choix se font sur sélection, puis sont gelés avant de lire la confirmation.

**Règle principale : le candidat asymétrique ne peut pas dépasser la taille réellement utilisée par le contrôle retenu.** Les courbes sous plafonds communs sont également enregistrées, sans confondre les deux questions. Si le candidat ne gagne pas en sélection sans octets supplémentaires, le contrôle est conservé.

| Graine | Contrôle, octets | Candidat retenu, octets | Réduction de NMSE sur confirmation |
|---|---:|---:|---:|
| 2003 | 1631 | 1612 | 2.65 % |
| 2011 | 1640 | 1631 | 6.75 % |
| 2027 | 1632 | 1590 | 4.62 % |
| 2029 | 1644 | 1635 | 7.96 % |
| 2039 | 1656 | 1635 | 6.19 % |

Sur le point de plafond maximal de ces cinq problèmes, la moyenne des réductions relatives est **5.63 %**. Les cinq candidats sont plus petits que leur témoin sélectionné et améliorent la confirmation. Ce signal n’est donc pas expliqué par un octroi d’octets supplémentaires au candidat dans cette nouvelle comparaison.

**Ce n’est ni un gain de 5.63 % des capacités de Qwen, ni une preuve statistique générale, ni une comparaison aux meilleurs pipelines publics.** Il n’y a que cinq instances synthétiques d’une même construction ; le codec local a des frais fixes importants ; seul `up_proj` change ; la référence historique L0 de l’utilisateur n’a pas été reproduite. Ces limites restent importantes.

Les nouveaux résultats remplacent l’usage promotionnel du « +3,54 % » historique. BGCC et le polissage restent expérimentaux, sans nouveau gain revendiqué ici. La recette recommande d’abord un petit pilote de calibration asymétrique sur une vraie capture contextualisée.

## 6. État opérationnel et intégration

| Brique | État v3.1 | Ce que cela autorise |
|---|---|---|
| Algèbre et LDLQ | Tests numériques et comparaison dense passés | Usage comme référence CPU, puis portage mesuré |
| Codec LE8CPU3/LMATv3 | Fichiers réels, contrôles stricts, lecteur indépendant | Oracle CPU de petites matrices ; pas CUDA ni format produit |
| Calibration asymétrique ridge | Formulation, code et signal synthétique sans octets supplémentaires | Pilote Qwen ciblé ; pas reproduction complète de GPTAQ |
| BGCC/PCG | Opérateur vérifié, non-convergence explicite | Expérience conditionnelle, désactivée par défaut |
| Polissage | Garde d’export et paramètres conservés | Exploration uniquement ; résultats négatifs historiques maintenus |
| Métriques et budgets | Cas de l’audit corrigés et contre-tests ajoutés | Évaluation et comptabilité dans les contrats documentés |
| Captures | Schéma 2 vérifié | Préfiltre, pas preuve d’authenticité du forward |
| Modèle Formic 10 Go | Non encodé / non évalué | Aucune revendication de capacités ou de VRAM globale |

La solution est donc **fonctionnelle en tant que bibliothèque et protocole CPU de référence dans le périmètre testé**. Elle n’est pas un moteur d’inférence Qwen complet. L’intégration réelle exige des captures authentiques, la référence L0/S0 du projet, le portage des calculs, le format GPU et les tests de bout en bout.

## 7. Prochaine décision recommandée

1. Exécuter `tools/check_release.py` sur le paquet livré pour vérifier manifeste, suite positive et décodage indépendant sans modifier la livraison.
2. Fournir au pilote une capture pleine largeur et contextualisée, ses cibles et sa provenance. Comparer la cible symétrique et la cible asymétrique séparément, à tailles réellement contrôlées.
3. Réexaminer BGCC seulement si un diagnostic et un budget d’adaptation le justifient. Ne pas empiler les pistes avant de mesurer leur contribution.
4. Garder l’architecture A0/Q0 et la proposition A1 de l’autre agent versionnées ; recalculer les statistiques après changement d’architecture.

## 8. Sources et preuves

- Contre-audit, tests et patch reçus : copies exactes dans `contre_audit_recu/`.
- Ancienne livraison : empreinte de l’archive, manifeste vérifié et logs dans `results/reaudit/` ; documents/expériences historiques dans `historique/`.
- [NumPy — Cholesky et lecture d’un seul triangle](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html).
- [SciPy — critère de convergence de CG et convention de préconditionnement](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).
- [G²PTQ — page primaire et titre corrigé](https://arxiv.org/abs/2609.31009) ; [dépôt mentionné par les auteurs](https://github.com/G2PTQ/G2PTQ), non audité ici.
- [API publique Qwen](https://huggingface.co/api/models/Qwen/Qwen3.8-27B), [arbre public](https://huggingface.co/Qwen/Qwen3.8-27B/tree/main) et [révision exigée](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0). Les observations d’accès sont séparées de la preuve d’identité des poids.

Les tests ci-dessus ont été écrits/exécutés dans ce travail, sauf le lecteur indépendant et les tests historiques dont la provenance est conservée. Le nombre de tests n’est pas une garantie d’absence de tout défaut futur. Tous les résultats de qualité nouveaux restent synthétiques.
