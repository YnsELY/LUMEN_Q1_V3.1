# LUMEN / FORMIC-Max — Livraison corrigée Q1-v3-ref

**12 novembre 2026 — dossier de recherche + référence exécutable CPU.**

Cette livraison corrige les réalisations mathématiques et les affirmations de Q1-v2, approfondit les pistes et livre des tests/résultats reproductibles. Ce n’est pas un modèle Qwen compressé, un moteurCUDA ou une recette supérieure globalement démontrée.

## Lire dans cet ordre

| Fichier | Rôle |
|---|---|
| `RAPPORT_RECHERCHE_RECETTE_MAX_V3.md` | Conclusions, nouveaux mécanismes, mathématiques, mesures et limites |
| `RECETTE_MAX_PROPOSEE_V3.md` | Passes 0–4, options actives/inactives et conditions d’intégration |
| `CORRECTIONS_ET_STATUTS_V3.md` | Résolution de 36 points et statut des preuves manquantes |
| `FORMAT_REFERENCE_E8_V3.md` | Format binaire réellement exécuté, différent du produit Lumen |
| `BUDGET_LUMEN_MAX_ECO_MINI_V3.md` | Plafonds, hypothèses et calculs de taille corrigés |
| `REGISTRE_PREUVES_ET_SOURCES_V3.md` | Sources, provenance et degré de validation |
| `DEMANDES_MESURES_POD_V3.md` | Trois demandes minimales proposées à l’agentB |
| `PROMPT_REVERIFICATION_V3.md` | Mandat pour un nouvel auditeur indépendant |
| `MANIFESTE_INTERFACE_A_B_V3.json` | Identités, dépendances et statut de la transmission |

## Code

`lumen_ref/` contient algèbre, E8, codec réversible CPU, Block-8 LDLQ, objectifs asymétriques/BGCC, métriques et solveurs de budget. Les résultats historiques restent dans `originals/` ; ne pas confondre leur script avec la référence V3. Les anciens jeux d’entrée manquent et n’ont pas été remplacés par des données inventées.

Environnement réellement exécuté : **Python 3.13.5, NumPy 2.3.5, CPU**. Aucun GPU ni poids Qwen n’est requis pour les vérifications livrées. Utiliser Python 3.11+ compatible avec la dépendance fournie ; seule la version déclarée a été testée ici.

```bash
python -m venv .venv
# Activer cet environnement selon votre système, puis :
python -m pip install -r requirements.txt
python tools/verify_manifest.py
python tools/run_verification.py
python tools/check_cold_streams.py
python budget_lumen_max_v3.py
python tools/run_experiments.py
python tools/run_isoshell_experiment.py
```

SousLinux, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` permet de retrouver les conditions des mesuresCPU. Les temps changent selon la machine. Les scripts principaux fonctionnent depuis un autre répertoire, grâce à leurs chemins relatifs à leur fichier.

**Vérifier le manifeste avant de relancer.** Les scripts réécrivent certains résultats/journaux et leurs durées ; conserver une copie du dossier distribué. `tools/verify_manifest.py` échouera ensuite sur les fichiers modifiés : c’est attendu, pas une raison de supprimer la traçabilité. Les tests numériques et les fichiers binaires sont à comparer séparément des temps.

## Résultats utiles

- `results/verification.json` et `tests_final.log` : nombre et résultats des tests exécutés.
- `results/experiences_synthetiques.json` : cinq graines, contrôles recherchés, toutes variantes et octets réels.
- `results/experience_isocoquille.json` : nouvelle piste à longueur constante, cinq résultats négatifs de sélection conservés.
- `results/codec_*.bin` : sixflux réellement écrits et décodés.
- `results/budget_faits_et_scenarios.json` : calculs corrigés et hypothèses, pas unpackage10Go.
- `results/developpement/` : essais antérieurs, explicitement non utilisés comme confirmation principale.

## Statut d’intégration

Algèbre et formatCPU : implémentés et testés dans leur périmètre. Cible asymétrique : premier pilote recommandé sur Qwen. BGCC et polissage : options de recherche, pas paramètres par défaut. Collecteur Qwen, kernels GPU, optimisation RCO complète, grande distillation et package final : non livrés comme fonctionnels.

Le codec ne garantit pas une inférence rapide ; il utilise du comptage dynamique et des grands entiers CPU. Le modèle 27B n’a pas été chargé. Aucun résultat produit, score de benchmark ou classement mondial n’est revendiqué.

## À transmettre

**Agent recette :** dossier V3 complet, recette Q0 et scripts/captures réellement disponibles. **Agent architecture :** rapport, recette candidate, demandes et manifeste ; joindre le code si des contrôles sont acceptés. **Contre-auditeur :** archive entière et prompt de revérification, pas seulement le résumé des résultats.

Pour une capture réelle : lire `tools/validate_capture.py` et `CAPTURE_MANIFEST_EXEMPLE.json`. Le schéma vérifie des invariants mais n’atteste pas la provenance ; ne pas remplir lemanifeste d’exemple avec une fausse affirmation de forward réel.

Les fichiers `MANIFEST_SHA256.json` et le ZIP permettent d’identifier exactement cette livraison. Une empreinte ne démontre pas la justesse scientifique.
