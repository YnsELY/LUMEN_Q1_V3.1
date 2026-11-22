# Artefacts du pod (22/11/2026) — copies de traçabilité

- `logs/` : journaux complets de tous les jobs du serveur de laboratoire (`/workspace/work_ink/logs`).
- `sealed/` : résultats scellés (moitié conf de P-I, P-I2, mesures secondaires des concurrents, KL par fenêtre), lus seulement après les gels.
- `*.pass.json`, `*.state.json` : bilans des passes LDLQ (multiplicateurs, couverture des experts par couche, durées) pour L0, V1a, V4, C1.
- `JOURNAL_POD.md` : résumé du pod ; `tables_phase1.md` : tableaux de la phase 1.
Les données lourdes (paquets, caches du professeur, GGUF) ne sont pas dans le dépôt : voir la liste dans le message de fin de session.
