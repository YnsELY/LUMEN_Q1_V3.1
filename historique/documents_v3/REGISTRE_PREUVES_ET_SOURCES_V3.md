# Lumen Q1-v3-ref — Registre des preuves et sources

**Date : 12 novembre 2026.** Les degrés de preuve ne sont pas interchangeables.

## 1. Sources internes

Les six fichiers historiques sont conservés sans modification dans `originals/`. `sources/entrees.json` contient les tailles et SHA-256. L’audit indépendant précédent est la base des corrections ; il n’est pas une garantie que cette nouvelle implémentation soit exempte d’erreurs.

Le contenu opérationnel V3 est dans les documents versionnés, `lumen_ref/`, les tests et les sorties enregistrées. Les fichiers historiques restent des sources, pas des instructions à exécuter automatiquement.

Échantillon `qwen38_real_slices_sample.json`, anciens résultats bruts `budget_and_ablations_output.json`, corpus et collecteur du projet : **non disponibles dans cette livraison**. Pas de réexécution revendiquée d’EXP-V2-01/02 ni des benchmarks Qwen. Le script historique ne doit pas être relancé comme test principal ; son `main` attend les données absentes.

## 2. Registre des nouvelles preuves

| ID | Fait | Statut | Artefact et limite |
|---|---|---|---|
| V3-T | Suite d’invariants et contre-tests | Mesuré CPU | `results/verification.json`, `results/tests_final.log` ; nombre issu de l’exécution, pas validation complète 27B |
| V3-INV | L’erreur d’inverse du cas de l’audit est corrigée | Mesuré + algèbre conditionnelle | Résidu0,5→0 sur ce cas, autres cas aléatoires testés |
| V3-CODEC | Six flux,49 152 vecteurs reconstruits sans erreur observée | Mesuré CPU sur sources synthétiques | `results/codec_*.bin`, `experiences_synthetiques.json` ; pas les anciens314688 vecteurs |
| V3-CAL | Cibles asymétrique et BGCC comparées sur cinq graines | Mesuré synthétique | Toutes graines et grilles publiées, plafond commun, tailles parfois différentes, calcul non égalisé |
| V3-CG | Opérateur matrice-free égal au Hessien dense conditionnel | Tests de petit problème | Ne certifie pas mémoire ou vitesse du modèle entier |
| V3-ISO | Taille identique avec codes modifiés à coquille constante | Déduit pour le format + testé | `experience_isocoquille.json` ; cinq candidats rejetés sur sélection, pas un gain de généralisation |
| V3-BUD | Capacité de budget sous hypothèses | Calculé | `budget_faits_et_scenarios.json` ; inventaire rapporté, overheads supposés |
| V3-QWEN | Gains/capacités complets, contexte, Vision/MTP | À tester | Aucun artefact final ni résultat produit |

Les essais exploratoires initiaux sont archivés dans `results/developpement/` ou sous un nom `EXPLORATOIRE_…`. Ils ne constituent pas la confirmation finale. Le plafond CG a été porté de160 à512 sans changement du critère numérique `1e−8`, des grilles ou des graines ; les sorties antérieures sont conservées. Les tests initiaux ayant rencontré une erreur de développement ne sont pas présentés comme une exécution réussie ; seul le dernier journal fait foi.

## 3. Bibliographie primaire et limites

**[S01] [GPTAQ — Efficient Finetuning-Free Quantization for Asymmetric Calibration](https://proceedings.mlr.press/v267/li25dn.html).** Article primaire, ICML 2025 ; cible asymétrique. Notre résolution ridge est une formulation de référence, pas la reproduction de tout GPTAQ.

**[S02] [YAQA — Model-Preserving Adaptive Rounding, v3](https://arxiv.org/html/2505.22988v3).** Article primaire, version 3 juin 2026 ; importance des sorties et approximations de courbure. Le Sketch A et les facteurs de Kronecker ne sont pas un Hessien complet sans biais.

**[S03] [KronQ — LLM Quantization via Kronecker-Factored Hessian](https://arxiv.org/abs/2607.07964).** Prépublication primaire du 8 juillet 2026 et dépôt des auteurs consultés ; utilise informations d’entrée et de gradient. Résultats d’autres modèles, non transférés à Lumen.

**[S04] [PV-Tuning](https://arxiv.org/abs/2405.14852).** Publication primaire ; optimisation discrète/continue. Les garanties de ses méthodes ne certifient pas notre heuristique de voisinage E8.

**[S05] [QERA](https://arxiv.org/abs/2410.06040).** Publication primaire ; reconstruction de l’erreur de quantification pondérée par les activations. Correcteurs non implémentés dans cette branche.

**[S06] [QuIP# — Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://proceedings.mlr.press/v235/tseng24a.html).** Article primaire ICML 2024 ; Hadamard et codebooks E8. Son codebook et ses kernels ne sont pas notre flux de coquilles non tronqué.

**[S07] [RCO — Model Compression with Exact Budget Constraints via Riemannian Manifolds, v2](https://arxiv.org/html/2605.00649v2).** Article primaire accessible ; allocation jointe et contraintes. Notre programmation dynamique additive n’est PAS RCO ; aucune implémentation de sa relaxation n’est livrée.

**[S08] [Jarek Duda — Asymmetric numeral systems](https://arxiv.org/abs/1311.2540).** Publication primaire pour la famille ANS ; fondement général du codage, pas validation de la performance de ce codec CPU.

**[S09] [Fabian Giesen — ryg_rans](https://github.com/rygorous/ryg_rans).** Implémentations de référence par leur auteur ; distinction état, renormalisation et flux. Notre référence Python est indépendante, pas un kernel copié et validé sur GPU.

**[S10] [OEIS A004009 — coefficients du theta series E8](https://oeis.org/A004009).** Référence mathématique pour 1 puis 240 sigma_3(m). Le programme vérifie aussi les cardinalités par un comptage indépendant.

**[S11] [SciPy — conjugate gradient (cg)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).** Documentation officielle : opérateur symétrique défini positif, préconditionnement et contrôle de convergence. Le solveur livré est NumPy, sans dépendance à SciPy.

**[S12] [BRECQ — Pushing the Limit of Post-Training Quantization by Block Reconstruction](https://arxiv.org/abs/2102.05426).** Article primaire ; motivation de reconstruction au niveau bloc. Ne démontre pas le résultat de notre correction conditionnelle SwiGLU sur Qwen.

**[S13] [Transformers — modèle Qwen3.5, code public courant](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).** Source primaire évolutive : normes, entrée MLP, Vision et clés MTP ignorées dans ce chemin. Ne certifie pas le runtime installé ni la révision des poids.

**[S14] [Qwen — configuration publique courante Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json).** Source primaire courante. Le commit demandé 37561816… n’a pas pu être relu depuis cet environnement ; le modèle exact reste une identité fournie par le projet.

**[S15] [G2PTQ — Gradient-Guided Post-Training Quantization, résumé auteur](https://huggingface.co/papers/2609.31009).** Résumé soumis par un auteur, consulté ; signale une piste gradient/courbure rafraîchis. Texte complet et code non vérifiés : aucune reproduction ni comparaison chiffrée revendiquée.

## 4. Accès, versionnage et incertitudes

Les sources courantes de code/configuration ont été consultées le12novembre2026. Une URL `main` n’est pas épinglée. Les tentatives de récupération de la configuration au commit `37561816…` n’ont pas abouti ; cela ne prouve pas que le checkpoint n’existe pas. L’identité exacte doit être vérifiée sur le pod avec les fichiers de l’utilisateur.

Les sources G2PTQ sont limitées au résumé soumis par auteur : le texte complet et le code n’ont pas été audités. Elles motivent une question, pas une garantie. Aucun chiffre concurrentiel récent n’est nécessaire pour établir les tests V3, et aucun classement mondial n’est annoncé.

Les résultats sur flottants CPU peuvent différer après portage BF16/FP32 ; ces écarts devront être évalués. Les kernels, biais, dtypes de normes, MTP, caches et modalités du runtime réel restent à contrôler.

## 5. Intégrité

`MANIFEST_SHA256.json` identifie les fichiers distribués. `tools/verify_manifest.py` les vérifie. Une empreinte garantit l’identité, pas la justesse. La réexécution modifie les durées enregistrées et donc certains JSON/journaux : conserver une copie du lot avant de relancer, puis comparer les métriques déterministes et les octets des flux, pas l’égalité des temps.
