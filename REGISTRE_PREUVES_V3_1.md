# Registre de preuves — Q1-v3.1

**12 novembre 2026.** Les statuts ci-dessous ne sont pas interchangeables.

| Élément | Statut | Artefact |
|---|---|---|
| Contre-exemples et constats D-01..D-25 | Reçus puis réexaminés | `contre_audit_recu/`, bilan et matrice de réponse |
| Manifeste de l’archive V3 complète | Exécuté ici, complet | `historique/MANIFEST_ORIGINAL_V3.json`, `results/reaudit/original_manifest.json` |
| 11/12 groupes du contre-auditeur | Exécutés sur la V3, un constat de fichier absent réfuté pour cet artefact | `results/reaudit/original_counteraudit.log` |
| 58 tests historiques + 49 tests positifs nouveaux | Exécutés sur v3.1 | `results/verification.json`, `results/tests_final.log` |
| Lecteur indépendant | Implémentation reçue, isolée puis réutilisée avec attribution | `tests/independent_decoder.py` |
| Six flux, 49 152 vecteurs | Décodage à froid exécuté | `results/cold_decode_verification.json`, `results/codec_*.bin` |
| Calibration asymétrique sur cinq nouvelles graines | Expérience synthétique exécutée, portée locale | `results/matched_budget/` |
| Chaîne export → décodage → score → acceptation | Démonstration synthétique exécutée | `results/demo_cpu/` |
| Coûts Max/Éco/Mini | Calculs conditionnels sur inventaire rapporté | `results/budget_faits_et_scenarios.json` |
| Statut des sources publiques | Observations d’accès, pas lecture supposée des poids | `sources/SOURCES_REVERIFIEES_V3_1.json` |
| Poids BF16 Qwen / captures cachées | Non présents / non exécutés | Aucun résultat réel n’est inventé |
| Modèle complet de 10 Go / qualité multi-domaines / GPU | Non validé | Jalons à exécuter séparément |

Les chiffres de qualité historiques sont conservés dans `historique/` et ne sont pas promus par la réussite des tests actuels. Le codec correct sur les cas couverts ne garantit pas que ses choix de quantification préservent les capacités. Une équation juste et une solution numérique contrôlée peuvent néanmoins sur-ajuster leur calibration.

Les sources et statuts historiques ne sont pas silencieusement réécrits. Le titre de G²PTQ est corrigé dans le registre actuel ; le texte complet et son dépôt n’ont pas été audités ici. Un accès infructueux à la révision épinglée ne signifie ni inexistence ni autorisation de prendre `main`.
