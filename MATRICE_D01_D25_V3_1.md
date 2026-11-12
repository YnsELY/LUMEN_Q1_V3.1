# Matrice de traitement D-01 à D-25 — Q1-v3.1

12 novembre 2026. Les décisions concernent l’archive réellement disponible.

| ID | Verdict sur la contre-vérification | Traitement | Preuve / localisation |
|---|---|---|---|
| D-01 | Confirmé | IDs Top-K mal formés : refus du type, des doublons, des bornes et des formes invalides. | tests MetricsRegressions |
| D-02 | Confirmé | Oracle entièrement en log-space ; contre-exemple -800 corrigé à 399,30685282 nats. | test_logspace_minus800 |
| D-03 | Confirmé | Partie symétrique explicitement factorisée ; rejet si elle n’est pas SPD. | NumericalRegressions |
| D-04 | Confirmé | Résidu vrai rafraîchi et poursuite ; non-convergence honnête au plafond. Aucune garantie universelle de convergence en 200 pas. | test_no_premature_false_return |
| D-05 | Confirmé ; patch insuffisant | Normalisation par maximum absolu, norme stable ; le cas 1e-200 corrige un défaut restant dans le patch. | test_tiny_rhs_beyond_proposed_patch |
| D-06 | Confirmé | Matrices vides rejetées par encode_matrix ; flux E8 vide reste valide. | test_matrix_empty_rejected_by_encoder |
| D-07 | Confirmé | Échelles déjà FP16 vérifiées ; pas de troncature cachée. | test_scale_not_silently_rounded |
| D-08 | Confirmé | Planchers et types stricts, tableaux valides, DP additive exhaustive. | BudgetRegressions |
| D-09 | Confirmé | Hyperparamètres NaN/inf invalides ; formulations d’ancrage maintenues explicites. | test_nan_regularizers |
| D-10 | Confirmé | Cible non convergée non promue ; ancienne archive cap160 conservée comme historique. | Code historique patché + protocole v3.1 sans BGCC par défaut |
| D-11 | Limite CPU confirmée | Dépaquetage par arbre ; pas de garantie universelle O(G log² G), grands entiers encore nécessaires. | test_rank_tree_same_as_mixed_integer_reference |
| D-12 | Confirmé | Types et bornes symétriques ; identité géométrique et limites de format distinctes. | test_e8_int_types_and_signed_boundary |
| D-13 | Confirmé avec nuance | Mini et alignement corrigés ; coût des échelles QERA présenté comme schéma explicite, non minimum universel. | budget_faits_et_scenarios.json |
| D-14 | Confirmé ; patch incomplet | Métadonnée corrigée ET group_size conservé depuis l’artefact ; cohérence des champs vérifiée. | test_refinement_preserves_nondefault_codec_group |
| D-15 | Défaut et approximation distincts | Overflow cache corrigé par promotion/rejet ; masques pris en charge ; biais FP16 conservé et contrôlé, pas prétendument éliminé. | MetricsRegressions |
| D-16 | Gardes confirmées ; choix Pareto explicite | NaN/vide/sentinelles refusés ; égalité qualité et baisse taille acceptables ; wrapper effectuant réellement le décodage et l’évaluation. | test_decode_and_evaluate_actual_artifacts |
| D-17 | Confirmé en partie | Schéma renforcé ; pas d’interdiction de Xfp=Xq ; mapping source par position au lieu d’exiger un document par token. | CaptureAndManifestRegressions |
| D-18 | Fichier absent réfuté ici ; autres remarques confirmées | Prompt présent à empreinte exacte, manifeste V3 valide. Portabilité et fichiers surnuméraires corrigés dans v3.1. | original_manifest.json ; original_counteraudit.log |
| D-19 | Facteur confondant confirmé | Nouvelle comparaison sans octets supplémentaires. Une loi gaussienne ne prouve pas que les anciens gains venaient seulement des octets. | matched_budget/experiences_taille_controlee.json |
| D-20 | Confirmé | Temps de cible, de grille et de préparation enregistrés. Calcul non prétendu égalisé. | timings des nouvelles expériences |
| D-21 | Approximation déclarée, pas réfutation de la propriété conditionnelle | Invariance à symboles positionnels, tables, groupements, échelles et dimensions figés ; plafond toujours contrôlé sur les bytes. | FORMAT_REFERENCE_E8_V3_1.md |
| D-22 | Conventions à expliciter | Cibles ridge/fraîche non assimilées aux articles complets ; coefficient 1/2 et conversion de lambda documentés. | docstring fresh_quadratic_target ; recette |
| D-23 | Titre confirmé ; inférence d’accès contestée | Titre G²PTQ corrigé. Non relu ne signifie pas indisponible ; HEAD public retourné différent dans cette consultation. | SOURCES_REVERIFIEES_V3_1.json |
| D-24 | Compléments documentaires | Division par diag(A) applique bien un préconditionneur inverse de Jacobi. Les métadonnées bibliographiques non revérifiées ne sont pas déclarées confirmées. | rapport §4.2 et sources |
| D-25 | Limite de preuve confirmée | Aucun Qwen complet, aucune validation des onze domaines, aucune preuve du meilleur 10 Go. | Périmètre de tous les documents v3.1 |

Les points historiques R-01 à R-05 restent retirés. La v3.1 ne rétablit ni les faux scores, ni une économie de 623 Mo contre Q0, ni l’égalité entre entropie et fichier encodé.
