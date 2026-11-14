# formic_v7 — campagne V7 : comparaison aux concurrents et recette retenue

| Fichier | Contenu |
|---|---|
| `RAPPORT_V7.md` | Rapport décisionnel : méthode, résultats, sélection, confirmation, paquet, limites |
| `RECETTE_LUMEN_V7_EXP.md` | La recette expérimentale retenue (F1), ses paramètres, ce qui change par rapport à Q1-v3.1, reproduction |
| `docs/AUDIT_PIPELINE.md` | Ce que le pipeline GPU exécute réellement, relié au code |
| `lab/v7.py` | Chargement d'un GGUF concurrent dans notre graphe, protocole public d'Arki05, export et relecture de paquet, Vision / MTP |
| `lab/gguf_dequant.py`, `lab/summary7.py` | Déquantification GGUF en flux ; tableaux et intervalles de confiance |
| `codec/` | Codec réel rapide (C + Python), paquet, vérification, tests (`test_fcodec.log`) |
| `jobs/` | Tâches exécutées, dans l'ordre (100 → 110) |
| `data/` | Scripts et manifestes des jeux `cal2` (calibration) et `conf2` (confirmation) |
| `results/` | Résultats bruts par document / par fenêtre, verrous de sélection, manifeste et bilan d'octets du paquet |

Le code du quantifieur (E8, LDLQ, passe séquentielle, banc d'évaluation) reste dans `../formic_arch/lab/`.
Campagne précédente (architecture × Lumen) : `../formic_arch/RAPPORT_ARCHITECTURE_X_LUMEN.md`.
