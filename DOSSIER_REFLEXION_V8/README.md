# DOSSIER_REFLEXION_V8 — tout ce qu'il faut pour continuer le travail hors du pod

Mandat : écrire `PROPOSITIONS_ARCHITECTURALES_FORMIC_F1.md` (voir `01_mandat/`). Ce fichier n'existe pas encore.

**Commencer par `00_JOURNAL_COMPTE_RENDU_SESSIONS.md`** : résultats, état d'avancement, plan de reprise.

| Dossier | Contenu |
|---|---|
| `01_mandat/` | Le mandat V8 |
| `02_donnees_statistiques/` | `stats_fisher.pt` (X, MU, G, TS par matrice), `dstat_L1.pt` (pivots LDL, pas par ligne), `checkpoint_headers.json`, logs des campagnes, résultats scellés par variante |
| `03_code_modele_qwen/` | Code `transformers` 5.22.2 du modèle (`modeling_qwen3_5.py`…), config du checkpoint, index des shards |
| `04_projet_F1_et_campagnes/` | Copie de `formic_v7/` (recette, rapport, codec, manifestes) et `formic_arch/` (labo, jobs, résultats JSON, rapports), plus les dossiers de pistes historiques |
| `05_artefacts_sessions/` | `colenergy.pt` (énergies colonnes/lignes), scripts de calcul, sorties complètes des commandes des deux sessions, rapports bibliographiques, transcripts bruts |

## Non inclus (trop volumineux ou inutiles à la réflexion)
- Poids du modèle BF16 : `/root/models/qwen3.8-27b` (≈ 55 Go), révision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
- Paquet F1 quantifié : `/workspace/work/packages/F1_cal12_uniform` (≈ 9,3 Go). Les manifestes sont dans `04_…/formic_v7/results/`.
- `/workspace/work/competitors` (≈ 18 Go) et `/workspace/work/data` (corpus, 92 Mo).

## Pour relancer le calcul interrompu (`beta.py`)
Scripts dans `05_artefacts_sessions/scripts/`. Ils lisent `colenergy.pt` dans le répertoire courant et `stats_fisher.pt` via un chemin absolu `/workspace/work/` : adapter vers `../02_donnees_statistiques/`.
