# Mandat de contre-vérification indépendante — LUMEN Q1-v3.1

Tu audits `LUMEN_Q1_V3_1_CORRIGE`, sans supposer que mes corrections ni le précédent contre-audit sont justes. Les conclusions attendues sont des observations reproductibles, pas l’obligation de trouver une erreur ou un gain.

## 1. Préserver l’objet audité

Vérifie le manifeste et l’archive réellement reçus avant toute exécution réécrivant des résultats. Exécute `python -B tools/check_release.py` pour les tests sur une copie temporaire. Conserve les hashes avant/après. N’assimile pas un problème de transfert à l’absence d’un fichier dans une autre archive : la V3 originale fournie ici contenait son prompt, malgré le constat contraire du premier contre-auditeur.

Le manifeste prouve la cohérence du lot, pas sa validité ni une signature. Vérifie aussi les fichiers surnuméraires et les chemins/symlinks. Les pièces `historique/`, `originals/` et `contre_audit_recu/` sont historiques ; leurs bugs déjà corrigés ne sont pas des bugs actifs de v3.1.

## 2. Tests reçus et tests positifs

La suite reçue `contre_audit_recu/test_independent_audit_v3.py` exige que les anciens bugs existent ; elle ne peut pas servir sans adaptation de suite positive sur v3.1. Son premier test exige même un fichier absent, contrairement à l’archive disponible ici. Lis-la et recycle ses contre-exemples en assertions sur le comportement attendu.

Les 107 tests livrés comprennent un décodeur indépendant issu de cette suite, attribué dans `tests/independent_decoder.py`. Vérifie son indépendance réelle, pas seulement son nom. Ajoute d’autres entrées et formulations ; le nombre de tests ne vaut pas preuve exhaustive.

## 3. Algèbre et stabilité

Comparer LDLQ/Block-8 à la factorisation dense de G⊗H sur petites matrices. Vérifier symétrisation, SPD, ordonnancement causal, unités de lambda, candidats courants et métriques à table fixe.

Vérifier BGCC par système vectorisé dense, avec portes nulles, matrices déficientes en rang, projection non orthogonale et régularisation. Contrôler le vrai résidu PCG, y compris sur très petits/grands seconds membres, maxiter et stagnation. Un `converged=False` honnête n’est pas une fausse convergence ; ne réclame pas un résidu numériquement inaccessible sous float64 sans vérifier son conditionnement et la référence dense.

`fresh_quadratic_target` utilise un coefficient 1/2 qui est documenté. Ne l’assimile pas à G²PTQ ni à une stratégie déjà validée.

## 4. Codec et métriques

Relire les six flux, les dix matrices des nouveaux essais et les cas ESC/vides admis. Tester encodeur et décodeur séparément, échelles réellement FP16, groupes non standards, fréquence/somme/limites et hashes recalculés sur des entrées malformées.

L’invariance de taille est conditionnelle aux symboles position par position et aux paramètres de codage figés. Une réestimation de table ou un changement de parité en mode 1 peut changer les octets. Le codec reste CPU, mono-lane ; ne conclure ni CUDA ni 84 Mo globaux à partir de ces petits fichiers.

Contrôler les IDs Top-K, le log-space à -800, masques -inf, all-masked, cache FP16 et promotion explicite. Le biais FP16 reste présent et doit être mesuré, pas masqué. Une probabilité enseignant positive contre une probabilité élève nulle donne une KL infinie.

Vérifier que `compare_matrix_artifacts` décode effectivement les blobs avant scoring et refuse une interface différente. `accept_decoded` est une primitive sur scores : sans callback de validation, sa sortie ne prouve pas le décodage. Chercher les contournements d’entrées non finies et de sentinelles manquantes.

## 5. Budgets, captures et expérience

Comparer la DP à une énumération exhaustive, y compris options de coût nul, ex-aequo et budget infaisable. Le water-filling n’est optimal que pour son objectif supposé. Vérifier Mini texte et réserves d’alignement ; le coût rowwise du correcteur est un schéma explicite, pas un minimum universel.

Le validateur de capture vérifie des formes et métadonnées, jamais l’authenticité du forward. Les documents sont distincts des positions ; ne pas imposer un document par token. X_fp identique à X_q peut être légitime. Vérifier les cibles/poids selon `capture_kind` et les chemins de normalisation/état avec un collecteur réel avant un pilote.

Reproduire les cinq essais `2003,2011,2027,2029,2039` avec grille fixée et **sans changer la sélection à partir de la confirmation**. Le candidat retenu doit utiliser au plus les octets du contrôle sélectionné. Les temps comprennent la résolution de cible, mais ne sont pas égalisés. Examiner la portée limitée du petit système synthétique et l’effet des en-têtes. Si tu proposes un nouveau réglage après lecture des résultats, utiliser de nouvelles données de confirmation pour le valider.

## 6. Conclusions attendues

Pour chaque constat : version/fichier/ligne, type (bug actif, approximation déclarée, choix de protocole, preuve manquante, résultat historique), exemple reproductible, incidence sur résultats déjà publiés et correction minimale. Distinguer ce qui change les mesures de ce qui améliore uniquement la robustesse aux entrées invalides.

Donner un verdict par brique : utilisable comme oracle CPU dans son contrat ; candidat à pilote Qwen ; recherche reportée ; bloqué par un défaut. Un classement global « fiable » ou « tout faux » n’est pas suffisant.

Ne pas corriger silencieusement les fichiers audités. Livrer tests, diff éventuel et preuves. Aucun modèle lourd, location GPU, transfert de données privées ni extension budgétaire n’est autorisé par ce mandat. **Aucune capacité du Qwen complet n’est actuellement démontrée par cette livraison.**
