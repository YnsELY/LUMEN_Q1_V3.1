# Mandat de contre-vérification indépendante — LUMEN Q1-v3-ref

Tu dois rechercher les erreurs et les limites de cette livraison, pas confirmer qu’elle est bonne. Ne te fie ni au nombre de tests, ni au vocabulaire « exact », ni aux empreintes pour conclure à sa justesse scientifique.

## 1. Périmètre

Lire le rapport V3, la recette proposée V3, la matrice des corrections, la spécification du format, le README, les modules Python et leurs résultats. Les originaux restent dans `originals/` et peuvent contenir les erreurs retirées : ne mélange pas leurs versions avec V3.

Aucun modèle Qwen complet n’a tourné ici. Les expériences nouvelles sont synthétiques. La révision exacte des en-têtes n’a pas été revérifiée. Ne transforme pas ces absences en validation ni en réfutation générale des mécanismes.

## 2. Première exécution

Sur une copie du dossier, depuis un autre répertoire :

```bash
python /chemin/LUMEN_Q1_V3_CORRIGE/tools/verify_manifest.py
python /chemin/LUMEN_Q1_V3_CORRIGE/tools/run_verification.py
python /chemin/LUMEN_Q1_V3_CORRIGE/budget_lumen_max_v3.py
python /chemin/LUMEN_Q1_V3_CORRIGE/tools/run_experiments.py
python /chemin/LUMEN_Q1_V3_CORRIGE/tools/run_isoshell_experiment.py
```

Comparer les métriques déterministes et les flux, pas les durées CPU. La réexécution réécrit des résultats : vérifier d’abord le manifeste d’origine et préserver la copie initiale. L’environnement de référence est Python 3.13.5 / NumPy 2.3.5 sur CPU. Documenter toute différence avec les versions, le BLAS et les paramètres d’exécution.

## 3. Vérifications à refaire indépendamment

### Algèbre et LDLQ

Tester `U·U⁻¹`, la reconstruction LDL, la formule Block-8 et l’onde frontale contre une formulation dense `G ⊗ H`, sur d’autres dimensions, graines et conditionnements. Vérifier les transposées et les unités bits/vecteur. Le sélecteur fini n’est pas un argmin global E8 : cela ne constitue une erreur que si un document lui attribue cette garantie.

### Calibration et BGCC

Partir de l’objectif et dériver les équations sans regarder les fonctions. Vérifier l’opérateur SPD, son préconditionneur et la solution CG contre un système explicitement vectorisé. Tester les portes presque nulles, les entrées de rang faible et une projection down non orthogonale. Vérifier la régularisation, son ancre, le facteur `1/N`, les gradients et le cas de résidu nul. Mesurer le résidu véritable, pas seulement le drapeau de convergence.

Le code de cible asymétrique est une résolution ridge suivie de quantification ; ce n’est pas une reproduction complète de GPTAQ. Le code de gradient frais est une identité quadratique sous hypothèses ; ce n’est pas une reproduction de G2PTQ.

### RDOQ et polissage

Vérifier la présence du code courant parmi les candidats, les deux sens de balayage, la coquille et le co-ensemble conservés dans le mode constant. La décroissance est conditionnelle à un objectif et à des tables figés. Chercher des cas où la réestimation des tables augmente les octets : la règle d’export doit les détecter. Ne pas prendre la baisse en apprentissage pour une amélioration sur données réservées.

### Codec

Utiliser ou écrire un lecteur indépendant pour de petits flux. Tester les 240 racines, les grandes coquilles, les deux co-ensembles, ESC, les flux vides, les fréquences extrêmes, le remplissage des rangs et les tailles exactes. Fuzzer les en-têtes avec checksum recalculé, les limites de N, les longueurs des sections, les faux rangs, les entiers hors plage et les troncatures.

Les checksums ne protègent pas contre un émetteur malveillant. Vérifier l’invariance de longueur à coquilles constantes dans les seules conditions déclarées. Identifier les coûts de mémoïsation et de grands entiers du décodeur CPU.

### Débit et échelles

Recalculer les octets à partir du fichier et des types stockés. Ne pas assimiler `model.total_model_bits` à `len(blob)*8`. Vérifier les échelles FP16 dès le choix des codes. Identifier les limites de compacité du format CPU et les coûts du produit non implémentés. Aucune extrapolation gratuite vers 84 Mo d’auxiliaires, 14 profils ou un débit GPU n’est autorisée.

### Budget

Vérifier le facteur huit du correcteur, les modules, les ressources partagées, les arrondis, les alignements et la séparation entre sensibilité supposée et mesurée. Contrôler la programmation dynamique par énumération brute sur de petits cas, y compris budgets impossibles et limite du nombre d’états. Elle n’est pas un solveur RCO ni une optimisation exacte de la perte jointe d’un modèle.

### Métriques et rollback

Reconstruire le Top-K depuis les log-probabilités. Distinguer l’agrégation exacte de l’arrondi FP16. Vérifier la covariance des sondes sans l’étendre au Hessien de Kronecker complet. Tester le retour à l’artefact entier pour les étapes continues et discrètes : données manquantes, sentinelles NaN, artefact trop grand, métrique inchangée et provenance non vérifiée.

## 4. Audit du protocole et des résultats

Les essais principaux ont un plafond local commun, pas des fichiers exactement de même taille. Le calcul n’est pas égalisé. Vérifier comment ces limites réduisent la portée des conclusions. Les grilles sont modestes, les poids synthétiques et les graines peu nombreuses.

Une nouvelle variante choisie à partir de la confirmation doit recevoir un nouvel ensemble réservé. Le résultat BGCC est irrégulier ; le polissage à coquille constante échoue sur la sélection. Ces résultats doivent rester dans le verdict. Ne pas rechercher rétroactivement un seul cas favorable pour promouvoir une méthode.

Pour une transposition à Qwen, exiger des captures authentiques : pleine dimension, véritable entrée MLP, mêmes préfixes, états, hooks, normes, non-linéarités complètes, biais, sources disjointes et révision confirmée. Ne pas remplacer ces entrées par des embeddings tronqués à 128 coordonnées.

Sans ces artefacts, la conclusion correcte reste : « testé algébriquement ou synthétiquement ; transfert à Qwen non établi ».

## 5. Livrable attendu

Produire un tableau priorisé des défauts avec fichier/ligne, contre-exemple reproductible, incidence, correction minimale et nécessité de recalculer les résultats. Séparer bug, approximation déclarée, choix à tester, manque de preuve et affirmation déjà retirée. Fournir des patches et des tests supplémentaires lorsque c’est possible.

Terminer par une décision par brique : intégrable comme outil, prête pour un pilote Qwen, bloquée par un défaut, ou reportée. Ne louer aucune machine, ne télécharger aucun poids lourd et ne lancer aucun apprentissage coûteux sans accord explicite.

Le résultat attendu est une contre-vérification utile et falsifiable, pas une attestation globale sans réserve.
