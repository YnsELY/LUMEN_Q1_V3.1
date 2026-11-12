# Schéma 2 — Captures destinées au pilote LUMEN

La commande `python -B tools/validate_capture.py capture.npz capture.json` vérifie un schéma et des hashes. Elle retourne toujours **`provenance_verified: false`** : seul un audit du collecteur et du forward établit la provenance.

## Champs JSON obligatoires

| Champ | Contrat |
|---|---|
| `schema_version` | Entier strict `2` |
| `capture_kind` | `input_pair`, `linear` ou `swiglu` |
| `checkpoint` | `Qwen/Qwen3.8-27B` pour une capture réelle |
| `revision` | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| `runtime_commit` | Identifiant hexadécimal 7 à 64 caractères du code exécuté |
| `activation_path` | Chemin exact de la sous-couche/hook |
| `full_input_width` | Entier JSON strictement positif, pas un booléen ou un flottant |
| `projection_kind` | Pour A0, famille permettant de vérifier la largeur réelle |
| `architecture_id` | `A0` ou identifiant immuable de la structure candidate |
| `weight_status` | Description non vide des poids/précisions du chemin capturé |
| `source_kind` | `captured_from_model_forward` |
| `array_sha256` | Empreinte complète du fichier NPZ |
| `splits` | Objet `cal`, `select`, `confirm` : listes non vides de sources uniques et disjointes |
| `sample_source_ids` | Pour chaque split, une identité de source par colonne d’activation |

Les identités de sources sont des documents, dépôts ou épisodes, pas nécessairement des tokens. Une source peut produire plusieurs colonnes. `sample_source_ids` relie les positions aux sources listées. Le même contenu renommé n’est pas détecté automatiquement : une décontamination sémantique/source demeure nécessaire.

Les tableaux sont flottants, finis, non vides et chargés avec `allow_pickle=False`. Des paires identiques `X_fp == X_q` restent permises : elles peuvent être légitimes ; le rapport signale leur identité.

## Tableaux

Pour les trois splits : `X_fp_cal`, `X_q_cal`, `X_fp_select`, `X_q_select`, `X_fp_confirm`, `X_q_confirm`, formes `[largeur, positions]` avec positions non nulles.

- `input_pair` : entrées seules, **insuffisantes pour valider BGCC ou une cible de sortie complète**.
- `linear` : ajouter `W_fp`, `Y_fp_cal`, `Y_fp_select`, `Y_fp_confirm`; `bias_fp` si la couche possède un biais.
- `swiglu` : ajouter `W_gate_fp`, `W_up_fp`, `W_down_fp` et les trois `Y_fp_*` correspondant à la sortie du MLP.

Les noms de tableaux ne suffisent pas à établir leurs rôles : les masques, positions et conventions doivent être documentés par le collecteur.

## Largeurs A0 contrôlées

| Famille déclarée | Largeur |
|---|---:|
| `mlp_gate`, `mlp_up`, `swiglu` | 5120 |
| `mlp_down` | 17408 |
| `attention_input`, `deltanet_input` | 5120 |
| `attention_output`, `deltanet_output` | 6144 |
| `lm_head` | 5120 |

Ces dimensions proviennent du dossier cible et restent à rapprocher des poids réellement chargés. Une structure A1 exige ses dimensions et contrats propres, pas une falsification d’A0. Pour Vision ou un chemin non listé, étendre le schéma par une révision déclarée après inspection du modèle ; ne pas attribuer une largeur textuelle arbitraire.

## Collecte à auditer hors schéma

Vérifier mêmes tokens/préfixes, masques et positions ; normalisation sur la largeur complète avant toute tranche ; convention RMSNorm `1+w` là où elle s’applique ; SiLU non tronquée ; biais et activation Vision ; état initial, reset et transitions de GatedDeltaNet ; chargement et forward explicites de MTP. En cas de tranche, conserver la contribution des autres colonnes aux préactivations.

Pour les tests du validateur seulement, `source_kind="synthetic_fixture"` demande `--allow-synthetic`. Cette option ne rend jamais les données synthétiques équivalentes à une capture Qwen.

Aucun collecteur complet, corpus privé ou poids BF16 n’est ajouté par cette livraison.
