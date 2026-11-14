# FORMIC / LUMEN — Comprendre les meilleures quantifications et construire une comparaison compétitive

**Recherche documentaire du 13 novembre 2026 — version 2 du dossier concurrentiel.**

**Cible du projet :** `Qwen/Qwen3.8-27B`, révision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. La mention « Qwen 3.1 27B » dans la dernière demande est interprétée comme une référence à ce même projet, et non comme une autorisation de changer de modèle. L’identité locale devra être vérifiée.

**Objectif produit :** au plus **10 000 000 000 octets complets**, avec texte, embeddings, tête de sortie, Vision, MTP et tous les éléments propres au décodage. La fidélité textuelle, les capacités, la qualité multimodale et l’efficacité de MTP sont des axes distincts.

**Nature de cette livraison :** étude de sources primaires, rapprochement avec la recette fournie et propositions de recherche. Aucun concurrent ni Qwen complet n’a été exécuté dans cette étude. Les scores externes sont publiés par leurs évaluateurs, pas reproduits ici. Les hypothèses d’intégration ci-dessous ne sont pas des résultats.

---

## 1. Réponse directe : comment arrivent-ils à ces KL ?

La réponse documentée est principalement une combinaison de **compensation d’erreur, choix de précision par tenseur, optimisation des valeurs quantifiées et calibration adaptée aux sorties recherchées**. Ce n’est pas nécessairement une nouvelle architecture, ni une distillation globale systématique. [S01, S03–S09]

Trois corrections de cadrage sont importantes :

1. **0,1064 pour Lumen et 0,0823 pour Arki05 ne mesurent pas encore un retard réel.** Les protocoles sont différents. Seule une évaluation commune peut déterminer l’écart et son origine.
2. **0,0640 chez Arki05 correspond à 10,09 Go de texte**, avant Vision et MTP. Cette référence dépasse déjà le plafond complet de Formic. Elle reste informative, mais n’est pas un concurrent complet iso-taille. [S01]
3. **« Quantification post-entraînement » ne veut pas forcément dire « aucun apprentissage ».** GSQ apprend ses affectations discrètes et ses échelles ; son article utilise aussi une distillation globale des seules échelles dans certaines expériences Llama. Il n’est pas établi que chaque GGUF Qwen publié a reçu toutes ces étapes. [S04–S05]

La recette Lumen connaissait déjà plusieurs de ces familles. Son document V3.1 distingue cependant explicitement des outils CPU, des propositions, des options désactivées et des mécanismes non intégrés à un produit complet. **Une brique citée ou prévue n’est pas une brique exécutée et validée sur les 64 couches.** [P01]

La prochaine campagne doit donc commencer par une confrontation expérimentale, pas par l’affirmation qu’E8 est nécessairement supérieur ou que les concurrents ont déjà gagné.

## 2. Les résultats à utiliser comme repères

### 2.1 Une même évaluation publiée par Arki05

Les tailles ci-dessous sont déclarées et arrondies en Go décimaux. Ce tableau reprend seulement les points utiles autour du budget visé. Il ne constitue pas une mesure de capacités ni une évaluation multimodale. **Les scores de GSQ-RCO et Unsloth dans ce tableau sont ceux mesurés par Arki05.** [S01]

| Référence textuelle | Taille publiée | KL moyenne publiée, nat/token |
|---|---:|---:|
| GSQ-RCO IQ2_XS | 8,42 Go | 0,151 |
| Arki05 2,51 bpw | 8,42 Go | 0,122 |
| GSQ-RCO IQ2_S | 9,26 Go | 0,101 |
| **Arki05 2,75 bpw** | **9,26 Go** | **0,0900** |
| Unsloth UD-Q2_K_XL, périmètre sans MTP | 9,48 Go | 0,0983 |
| **Arki05 2,82 bpw** | **9,48 Go** | **0,0823** |
| GSQ-RCO IQ3_XXS | 10,09 Go | 0,0740 |
| **Arki05 3,00 bpw** | **10,09 Go** | **0,0640** |

Le protocole annoncé comprend 128 fenêtres de 4 096 tokens, neuf domaines, environ 220 000 positions effectivement scorées. La première moitié sert de contexte ; les réponses du BF16, raisonnement inclus, sont scorées pour les jeux à prompts. Les masques et fenêtres sont annoncés dans `eval/`, avec seulement 108 fenêtres redistribuées et les identifiants des 20 autres. Les tailles excluent MTP. [S01]

**Conséquence pour Formic :** 600 documents de 1 024 tokens scorés autrement ne sont pas automatiquement un test plus facile, plus difficile ou équivalent. Changer contexte, domaines, positions ou préfixes change la mesure. On ne convertit pas une KL entre ces protocoles par règle de trois.

### 2.2 Un autre concurrent concret : Axolotl XQD

XQD Q2 est annoncé à **9,47 Go sans MTP**, **9,82 Go avec MTP**, sans Vision. Sa fiche rapporte une **baisse relative de KL de 22,2 %** contre Unsloth UD-Q2_K_XL sur 1,2 million de positions assistant, conversations et usages d’outils. Elle ne donne pas ici une KL absolue transférable au tableau Arki05. Les auteurs signalent leur spécialisation chat/agentique et ne revendiquent pas de progression démontrée des benchmarks aval. [S09]

Il est interdit de multiplier 0,0983 du tableau Arki05 par 0,778 pour inventer une KL XQD : les ensembles d’évaluation diffèrent.

### 2.3 Un repère plus gros, à ne pas confondre avec la cible

La fiche **BaseQuant Small** rapporte 0,0460 à **10,93 Go** sur son propre ensemble. Ce point montre qu’une autre recette et un autre protocole peuvent publier un chiffre très faible ; il ne démontre pas une solution complète de 10 Go ni une supériorité sur le corpus de Fable. [S16]

### 2.4 Compléter les tailles sans fabriquer de produit concurrent

GSQ-RCO propose Vision BF16 séparément et des variantes avec MTP, environ 0,9 et 0,35 Go supplémentaires selon sa fiche. Les KL du tableau Arki05 concernent les versions sans MTP. [S01, S03]

Une version Arki05 de 9,48 Go laisse **520 Mo** sous le plafond de 10 Go ; celle de 9,26 Go laisse **740 Mo**. Ce sont des soustractions sur tailles arrondies, **pas la preuve qu’un assemblage Vision/MTP compatible et fidèle existe**. Il faut examiner les tenseurs, interfaces, précisions et fichiers réels.

Inversement, il serait biaisé d’obliger les concurrents à conserver une Vision BF16 volumineuse alors que Lumen a le droit de la quantifier. Distinguer :

- **Comparaison produit :** paquets complets réellement disponibles, avec leurs choix et limites.
- **Comparaison algorithmique :** budget textuel égal et traitement comparable des accessoires, pour isoler la méthode de compression.

## 3. Arki05 : ce que la documentation établit, et ce qu’elle ne révèle pas

### Mécanismes documentés

La fiche annonce une requantification depuis le BF16, guidée par le Hessien, de type GPTQ, puis un choix de format par tenseur avec **qalloc**, à partir d’un magasin de variantes. Elle renvoie aux allocations et aux fragments permettant de reconstruire les fichiers. [S01]

La famille GPTQ utilise des informations de second ordre pour compenser les effets de l’arrondi sur les calculs, plutôt que d’arrondir simplement chaque valeur au plus proche. Le papier GPTQ documente ce mécanisme général ; cela ne révèle pas tous les réglages d’Arki05. [S10]

### Ce qu’il faut vérifier avant une reproduction

Les accès directs à certains fichiers Arki05 ont échoué dans cette session ; la fiche était consultable via son indexation. **Le solveur qalloc, les fichiers d’allocation, les shards, les masques détaillés et les hyperparamètres n’ont pas été audités ici.** On ne connaît donc pas une recette ligne par ligne garantie.

Le document ne permet pas d’attribuer séparément le gain au compensateur, à l’allocation ou aux données. Il ne prouve pas non plus que qalloc soit RCO, ni qu’un algorithme particulier de courbure de sortie ait été utilisé dans ces GGUF.

La fiche NVFP4 du même auteur décrit des informations d’entrée et de sortie pour un autre produit, nettement plus gros. **Elle motive une question à examiner, pas une preuve que le GGUF emploie exactement le même algorithme.** [S02]

### Proposition pour Lumen — déduction, pas résultat publié

Construire une petite banque de variantes réellement décodables sur les groupes qui dominent le compromis taille/fidélité. Mesurer, pour chaque choix, son coût complet et son effet, puis choisir une combinaison sous budget. Le premier diagnostic à obtenir est : **le gain vient-il de meilleurs arrondis à allocation fixe, ou d’une meilleure allocation à quantifieur fixe ?**

Il faut garder les deux contrôles séparés. Copier une allocation concurrente sans reproduire son quantifieur ou ses données ne garantit pas de copier ses résultats.

## 4. GSQ et RCO : deux optimisations différentes, à ne pas réduire à une étiquette

### 4.1 GSQ améliore les valeurs réellement retenues

Le code des auteurs part notamment d’une initialisation GPTQ et optimise les affectations aux valeurs discrètes ainsi que les échelles de groupe. La relaxation Gumbel-Softmax permet d’explorer plusieurs choix avant de figer les codes. L’export revient à un format scalaire compatible avec les moteurs existants. [S05]

L’article décrit une optimisation par étapes à l’intérieur des blocs et des entrées provenant d’un préfixe déjà quantifié. Dans ses expériences Llama, il utilise 4 096 séquences de 4 096 tokens, plusieurs époques locales et une phase finale de distillation des échelles. **Ces volumes ne sont pas une mesure du coût de la version Qwen3.8 étudiée ici.** [S04]

**Conséquence :** il était incorrect de résumer tous les concurrents comme des compressions « sans entraînement ». Il faut distinguer calibration, optimisation locale des codes, récupération globale et SFT/RL. Le détail de la phase finale des GGUF Qwen doit être retrouvé dans leur configuration de fabrication, pas déduit du seul acronyme GSQ.

### 4.2 RCO choisit où dépenser la précision

RCO vise une affectation discrète de formats sous budget en tenant compte d’un objectif réseau non séparable. Sa réalisation combine une relaxation, des mises à jour contraintes et une sélection discrète. Cela va au-delà d’une allocation calculée une fois à partir de coefficients manuels. [S06–S08]

**La DP additive de la bibliothèque Lumen n’est pas RCO.** Elle optimise exactement la table additive qu’on lui fournit, pas automatiquement la perte jointe du modèle. [P01]

### 4.3 Adaptation à Lumen : point technique essentiel

Dans Lumen, un « choix à 2,8 bits » n’a pas nécessairement un coût constant après mise à jour des codes : histogrammes, échelles, tables et groupements peuvent modifier la taille. Le budget relaxé d’un solveur n’est donc pas le fichier final.

Une adaptation doit raisonner sur des variantes encodées, par exemple :

\[
\min_{k_1,\ldots,k_G}\ \mathcal L(Q_{1,k_1},\ldots,Q_{G,k_G})
\quad\text{sous}\quad C_{\mathrm{fixe}}+\sum_g C_{g,k_g}\leq 10^{10}.
\]

Cette écriture est exacte seulement lorsque les coûts sont effectivement additifs. Si des tables sont partagées, compter leur activation et les interactions séparément. La réestimation des tables doit être suivie d’un nouveau contrôle d’octets.

L’agent peut préférer des échanges locaux de budget et une banque réduite à une reproduction coûteuse de RCO. **Le nom de l’algorithme n’est pas l’objectif ; la qualité mesurée sous contrainte l’est.**

## 5. Axolotl : la tête de sortie et la distribution de calibration

La documentation XQD annonce une compensation de type GPTQ, des types par tenseur et un traitement spécifique de la tête de sortie avec ses propres données d’importance. Elle explique réinvestir ailleurs l’espace économisé. Elle décrit une quantification sans fine-tuning, orientée vers les tokens écrits par l’assistant. [S09]

### Pourquoi cela mérite un diagnostic ciblé

La tête convertit la représentation interne en probabilités sur le vocabulaire. Une erreur à cet endroit affecte directement la métrique finale. La fixer à une précision élevée « par prudence » peut être coûteux ; la baisser sans la calibrer peut être destructeur.

Pour les dimensions du projet, `248320 × 5120 = 1 271 398 400` paramètres : **un bit par poids vaut environ 158,9 Mo de payload brut**. Passer de 3,3 à 3,0 bits ne libérerait qu’environ **47,7 Mo**, avant format. Ce n’est pas une économie démontrée ; les mesures doivent autoriser la baisse.

**Ne pas inventer 795 Mo de gain pour Lumen** en comparant une tête 8 bits à 3 bits si Lumen prévoit déjà environ 3,3 bits. L’allocation actuelle exécutée par Fable doit être inspectée, pas remplacée par une hypothèse avantageuse.

Ce diagnostic n’a rien à voir avec la prédiction de `lm_head` depuis les embeddings, suspendue dans l’Étape 1. On étudie ici **comment quantifier la tête elle-même**.

### Données : tirer une leçon sans sacrifier le modèle généraliste

Il faut tester un mélange incluant les sorties BF16, le raisonnement et les appels d’outils, tout en gardant prose, langues, connaissances et entrées multimodales. Une amélioration sur les seuls tokens d’assistant ne garantit pas la préservation générale. L’agent doit choisir les pondérations sur un développement séparé, puis les figer.

## 6. Pourquoi Lumen peut être en retrait — et ce qui n’est pas encore démontré

**Faits du projet :** la recette V3.1 est une proposition d’intégration avec oracle CPU. Le témoin GPU de Fable à KL 0,1064 est rapporté sur un autre corpus, sans dossier complet des artefacts et de leur taille dans cette étude. [P01, P02]

| Question à auditer dans le code effectivement exécuté | Pourquoi elle change la comparaison |
|---|---|
| Quels poids sont quantifiés, lesquels restent BF16 ? | Le débit du tronc ne décrit pas le package complet. |
| L’allocation varie-t-elle réellement par tenseur/profondeur ? | Un débit moyen ne révèle ni uniformité ni optimisation. |
| Quel objectif optimise LDLQ : sortie locale, bloc, logits ? | Les erreurs numériquement similaires peuvent avoir des effets différents. |
| Embeddings et tête ont-ils leurs propres statistiques ? | Même forme stockée ne signifie pas même opération. |
| Les codes/échelles ont-ils reçu une optimisation locale apprise ? | Un passage d’arrondi n’est pas un pipeline GSQ complet. |
| Les captures suivent-elles le préfixe quantifié et les bonnes coordonnées ? | Une statistique mal placée fausse la reconstruction. |
| La mesure est-elle faite sur les poids décodés et le vrai format final ? | Une quantification simulée n’atteste ni taille ni moteur. |

Ces lignes sont des **questions**, pas des défauts déjà prouvés de Fable. Il faut remplir chaque case avec fichier, fonction, configuration et résultat.

De même, « 2,821 bits par poids » dans un banc Lumen et « 2,82 bpw » pour un fichier GGUF ne garantissent pas le même budget : vérifier le dénominateur, les poids exclus/protégés et les auxiliaires. Un débit entropique ou limité aux matrices testées n’est pas le débit effectif du package.

La faiblesse de notre démarche antérieure était de traiter trop tôt la recherche de recette comme suffisamment terminée pour viser la première place, sans confrontation exécutable avec les meilleurs concurrents. Les audits ont fiabilisé des outils ; ils n’ont pas établi l’optimalité de la recette. La priorité devient la **co-optimisation mesurée de l’ensemble du pipeline**.

## 7. Les pistes les plus utiles à approfondir

### A. Allocation mesurée et formats hybrides — priorité de diagnostic

Comparer E8 et au moins un scalaire appris/format GGUF pertinent sur les mêmes captures et budgets, puis tester une allocation par tenseur. Une variante hybride peut être plus intéressante qu’un format unique : E8 sur certains composants, scalarité sur d’autres. Il faut compter les tables, chemins de décodage et éventuels coûts de conversion.

**Hypothèse à réfuter :** le quantifieur vectoriel actuel est localement bon, mais les octets ne sont pas dépensés là où ils réduisent le plus la KL globale. Une comparaison à allocation figée, puis quantifieur figé, permet de séparer les causes.

### B. Améliorer les codes après une bonne initialisation

GSQ et PV-Tuning fournissent deux familles d’optimisation discrète/continue. QuIP# montre également que la géométrie E8 est un élément d’un pipeline plus large, avec transformations et affinage : **le réseau E8 seul ne garantit pas la meilleure fidélité finale**. [S04–S05, S11, S14]

Pour Lumen, explorer un voisinage de codes compatible avec son format, une relaxation locale ou une variante scalaire. Les polissages V3.1 ayant sur-ajusté, ne pas les réactiver sans régularisation, nouvelles données de sélection et confirmation vierge. Mesurer après reprojection et export, pas avant.

### C. Objectif sensible à la sortie et à la dérive amont

GPTAQ motive la reconstruction de la sortie BF16 à partir des entrées déjà perturbées ; YAQA motive un objectif plus proche de l’effet final que le seul Gram d’entrée. Ce sont des précédents, pas des garanties pour leur combinaison dans Lumen. [S12–S13]

Le changement de pas « Fisher » qui a échoué chez Fable ne réfute pas toutes les méthodes utilisant une courbure. Avant un nouvel essai, définir l’objectif, les coordonnées, l’estimateur et ce qui sera changé à octets égaux. Une cible ridge est une réalisation possible, pas une reproduction automatique de GPTAQ ou YAQA.

### D. Architecture : conserver toute la liberté, demander le bénéfice final

Les rotations, factorisations, bases partagées, sous-espaces protégés et modifications de mémoire restent admissibles. Une architecture moins fidèle avant quantification peut devenir meilleure après récupération et réallocation ; inversement une transformation exacte peut ne rien apporter au codec final.

Ne pas obliger l’agent à produire une « nouvelle architecture » pour justifier la mission. Le résultat prioritaire est un meilleur modèle sous budget ; la nature des modifications doit être nommée correctement. Les sous-systèmes Vision/MTP doivent rester comptés et testés.

### E. Récupération progressive plutôt qu’un gros entraînement présumé indispensable

Séparer les résultats sans apprentissage, après apprentissage local de compression, et après distillation globale. La première version doit déjà être mesurable ; elle ne doit pas être privée d’un mécanisme local nécessaire sous prétexte que toute adaptation devrait attendre un produit final.

Une campagne lourde n’est autorisée qu’après un pilote informatif et un budget accepté. Le coût des générations professeur, des essais de précision, de l’allocation, du rechargement et du stockage des variantes doit être inclus.

## 8. Le protocole qui rendra l’objectif chiffrable

### 8.1 Deux pistes d’évaluation complémentaires

**P-A, reproduction publique :** chercher à reproduire le protocole Arki05 avec ses positions et ses versions. Si les 20 fenêtres non redistribuées manquent, publier le résultat sur le sous-ensemble disponible sous un autre nom ; ne pas le comparer numériquement au tableau complet comme s’il était identique.

**P-F, évaluation Formic :** un ensemble généraliste commun aux candidats et concurrents, séparé en calibration, sélection et confirmation. Les 600 documents déjà examinés servent au développement ou à la continuité, pas à une nouvelle confirmation indépendante.

Le corpus MetaMathQA déjà utilisé demande notamment une vérification de parenté avec les jeux mathématiques employés en évaluation. Absence de texte identique ou de `source_id` partagé ne suffit pas à exclure un problème reformulé.

### 8.2 Contrat de KL

Pour chaque position retenue du même préfixe :

\[
K = \frac{1}{|T|}\sum_{t\in T}\sum_{v}p_{\mathrm{BF16}}(v\mid x_{<t})
\log\frac{p_{\mathrm{BF16}}(v\mid x_{<t})}{q(v\mid x_{<t})}.
\]

Fixer avant comparaison : professeur et révision, tokens, masques, sens de KL, logarithme naturel, longueur de contexte, agrégation par token/document/domaine, templates, positions et types des caches. Ne pas choisir une moyenne différente après avoir vu le classement.

Si les moteurs diffèrent, mesurer des contrôles haute précision avec les mêmes entrées. **Ne pas soustraire mécaniquement une KL de moteur ou de rotation** : les divergences ne sont pas additives. Publier le résidu comme limite de comparaison.

La suite publique StillDeadcode offre un autre protocole utile, mais son score agrège **top-512 et masse restante**, et ses tailles concernent le texte hors Vision/MTP. Cette KL agrégée n’est pas la KL plein vocabulaire. Ses Hessiennes publiques peuvent aider dans leurs coordonnées d’origine ; elles ne remplacent pas celles d’une architecture modifiée. [S15]

### 8.3 Départager sans surpromettre

Calculer différences appariées, IC par documents/épisodes, mesures par domaine, top-1 et queues de distribution. Un intervalle contenant zéro ne prouve pas l’équivalence. Une baisse moyenne ne suffit pas si un domaine critique régresse.

Geler le candidat avant confirmation. Si l’on change la recette après lecture du résultat, cet ensemble est devenu un jeu de sélection. Les benchmarks de capacités et les évaluations image/vidéo restent séparés de la fidélité textuelle. MTP exige au minimum une vérification fonctionnelle et son taux d’acceptation/débit dans un protocole correctement vérifié.

## 9. Objectifs numériques proposés — non prédictifs

### Objectif concurrentiel principal

Sur un protocole commun et un budget pertinent identique, viser une KL **au moins 5 % plus faible** que le meilleur concurrent effectivement évalué ; une baisse de **10 %** constitue un objectif renforcé. Ce sont des marges de recherche choisies, pas des seuils universels de signification.

Exiger en plus un écart apparié confirmé et des garde-fous de non-régression définis avant le test. « Meilleur au monde » ne peut être établi par une table incomplète : la revendication défendable est **meilleur parmi les références publiques comparables évaluées, à une date, sur un protocole et un périmètre explicités**.

### Jalons absolus sur un protocole Formic figé

| Jalon interne | KL visée | Baisse arithmétique par rapport à 0,1064* |
|---|---:|---:|
| Première amélioration | ≤ 0,100 | 6,02 % |
| Objectif central ambitieux | **≤ 0,080** | **24,81 %** |
| Objectif renforcé | ≤ 0,070 | 34,21 % |
| Objectif très ambitieux | **≤ 0,060** | **43,61 %** |
| Exploration ultérieure | ≤ 0,050 | 53,01 % |

\* Calcul `100 × (1 − cible/0,1064)`, pas projection scientifique. Sur un nouvel ensemble, remesurer le témoin et recalculer la réduction. Le passage de 0,10 à 0,08 ne dit pas combien de points de benchmark seront récupérés.

L’objectif 0,080 de Formic ne bat pas automatiquement 0,0823 d’Arki05 : il faut que les deux mesures utilisent le même protocole. Les seuils sont motivants, mais ne doivent jamais pousser à changer les masques ou les domaines pour rendre la KL plus petite.

## 10. Recherche opérationnelle et intégration

Le prochain agent doit être libre de choisir la solution. Une première décision utile serait un **audit de maturité court**, puis une comparaison discriminante avec un concurrent fort, plutôt que de retester toutes les idées historiques.

Les documents de référence à transmettre sont ce dossier, le prompt V7, l’archive Lumen Q1-v3.1 et les résultats finaux/code de Fable. Garder leurs identifiants immuables. Une recette modifiée devient une nouvelle candidate, pas un changement silencieux de Q1-v3.1.

Prévoir peu de concurrents téléchargés simultanément. Une banque multi-formats ou un ensemble public de Hessiennes peut dépasser de loin la taille du produit final. Ne pas télécharger automatiquement toutes les variantes sur un disque de pod déjà occupé par le BF16. Mesurer le coût et choisir le plus petit test qui change une décision.

**Livraison attendue de la campagne :** tableau de candidats réellement exécutés, octets du package et du cœur texte, KL sous protocole nommé, résultats par domaine, coûts, modifications de recette et d’architecture, artefacts reproductibles, résultats négatifs et limites. Un gain local ou une proposition non exécutée conserve ce statut.

---

## 11. Sources primaires et limites d’accès

Les pages évolutives ont été consultées le 13 novembre 2026. Les liens ne constituent pas des révisions de code immuables : épingler les commits au moment de reproduire. Les scores n’ont pas été réexécutés ici.

- **[S01] Arki05, Qwen3.8-27B-GGUF — fiche primaire indexée :** https://huggingface.co/Arki05/Qwen3.8-27B-GGUF . Méthode générale, table KL, périmètre et protocole. Accès direct à certains fichiers échoué ; ni solveur qalloc ni allocations audités. Ressources annoncées à examiner : https://huggingface.co/Arki05/Qwen3.8-27B-GGUF-shards et répertoires `allocations/`, `eval/` du dépôt principal.
- **[S02] Arki05, NVFP4 — fiche primaire indexée :** https://huggingface.co/Arki05/Qwen3.8-27B-NVFP4 . Autre produit, à ne pas assimiler à la méthode GGUF.
- **[S03] ISTA-DASLab, fiche Qwen GSQ-RCO :** https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF . Formats, modules séparés et chaîne de fabrication.
- **[S04] Dadgarnia et al., GSQ, texte HTML v2 :** https://arxiv.org/html/2604.18556v2 . Sections 3–4 et annexes C, G, H : optimisation, données, phases locales et distillation d’échelles. La v2 consultée n’est pas présentée comme la dernière version possible.
- **[S05] GSQ, dépôt des auteurs :** https://github.com/IST-DASLab/GSQ . README et structure consultés ; pas d’audit complet ni d’exécution du dépôt.
- **[S06] Helcig et Alistarh, RCO, texte HTML v2 :** https://arxiv.org/html/2605.00649v2 . Allocation contrainte, relaxation et discretisation.
- **[S07] RCO, dépôt des auteurs :** https://github.com/IST-DASLab/RCO . README et organisation de la recherche sous budget.
- **[S08] RCO, code de recherche quantification :** https://raw.githubusercontent.com/IST-DASLab/RCO/main/src/search/quant.py . Lecture du point d’entrée et des mécanismes d’interpolation/recherche ; pas validation de tous ses chemins d’exécution.
- **[S09] Axolotl XQD — fiche primaire indexée :** https://huggingface.co/axolotl-ai-co/Qwen3.8-27B-GGUF . Résultats relatifs et détails de fabrication. Des rendus indexés exposent davantage de détails sur la tête que d’autres ; vérifier la version locale. Le write-up annoncé n’a pas été audité dans cette étude.
- **[S10] Frantar et al., GPTQ :** https://arxiv.org/abs/2210.17323 . Compensation de second ordre, fondement général, pas reproduction d’Arki05.
- **[S11] Tseng et al., QuIP# :** https://proceedings.mlr.press/v235/tseng24a.html . Quantification vectorielle E8, transformations et pipeline ; pas le même codec que Lumen.
- **[S12] Li et al., GPTAQ :** https://proceedings.mlr.press/v267/li25dn.html . Calibration asymétrique.
- **[S13] Tseng et al., Model-Preserving Adaptive Rounding / YAQA :** https://arxiv.org/html/2505.22988v3 . Sensibilité de sortie et approximation du problème de reconstruction.
- **[S14] Malinovskii et al., PV-Tuning :** https://arxiv.org/abs/2405.14852 . Optimisation continue/discrète, transfert à Lumen non établi.
- **[S15] StillDeadcode, suite de comparaison Qwen :** https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge . Fiche primaire indexée ; fichiers de données/code annoncés non exécutés ici.
- **[S16] BaseCompute, BaseQuant :** https://huggingface.co/basecompute/Qwen3.8-27B-BaseQuant-V1 . Fiche primaire indexée ; point hors budget retenu uniquement comme contexte.
- **[P01] Projet :** `RECETTE_MAX_PROPOSEE_V3_1.md`, fourni dans la conversation et lu intégralement.
- **[P02] Projet :** `PROMPT_BENCHMARKS_ET_OBJECTIFS_KL_FORMIC_MAX.md`, fourni dans la conversation et lu intégralement. Les observations de Fable restent rapportées tant que leurs sorties brutes ne sont pas auditées.

**Conclusion :** plusieurs leviers concrets existent pour approfondir Lumen. Le plus urgent n’est pas de promettre 0,06 : c’est de mesurer le meilleur concurrent dans les mêmes conditions, puis de relier chaque amélioration au composant, aux octets et au coût qui l’ont produite.
