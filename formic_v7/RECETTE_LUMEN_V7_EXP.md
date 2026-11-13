# Recette expérimentale retenue — LUMEN V7-EXP « F1 »

**13 novembre 2026.** Recette de quantification sélectionnée à l'issue de la campagne V7 pour `Qwen/Qwen3.8-27B`.
Statut : **expérimentale**. Elle a produit un paquet réel de 9,974 Go, relu depuis ses fichiers et évalué ; elle n'a pas de moteur
d'inférence dédié et n'a été jugée que sur la fidélité des logits. Résultats et preuves : `RAPPORT_V7.md`.

## 1. La recette en une page

| Élément | Valeur retenue | Code |
|---|---|---|
| Architecture | **inchangée** (64 couches, aucun biais ajouté, aucune rotation) | — |
| Quantifieur des 498 matrices du chemin texte | réseau **E8**, codes « doublés », poids décodé = code × pas de ligne / 2 | `formic_arch/lab/formic_q.py` |
| Pas | un pas **FP16 par ligne de sortie** = multiplicateur global × RMS de la ligne | `row_steps` |
| Multiplicateur global | **0,598541** (identique pour toutes les matrices : allocation **uniforme**) | `formic_v7/jobs/108_cal12_exported.py` |
| Compensation d'erreur | **LDLQ par blocs de 8 colonnes**, 7 candidats par bloc (zéro + 3 rétrécissements × 2 cosets) choisis par la métrique 8×8 du bloc, amortissement 1 % de la diagonale moyenne, `rate_lambda = 0` | `ldl_prepare`, `ldlq_rows` |
| Objectif | erreur de sortie locale de chaque linéaire, **symétrique**, sur les entrées de l'élève (préfixe déjà quantifié), couche par couche | `formic_arch/lab/seq.py : ldlq_pass` |
| Ordre dans une couche | projections d'entrée du mélangeur (même Gram, lignes empilées) → projection de sortie → `gate`/`up` → `down` | idem |
| `embed_tokens` | E8 au plus proche, **sans rétroaction**, même multiplicateur | idem |
| `lm_head` | LDLQ sur le Gram des états finaux normalisés de l'élève, même multiplicateur | idem |
| **Calibration** | **409 600 tokens** : `cal` (144 documents × 1024 : prose, 7 langues, code, math, conversations, science ; texte brut sans jeton initial) **+ `cal2`** (128 documents × 2048 : prose, web, 8 langues, code Python et C, math, science, **conversations, appels d'outils et traces de raisonnement rendus avec le gabarit de conversation** ; un document brut sur deux commence par `<|endoftext|>`) | `formic_arch/data/build_corpus.py`, `formic_v7/data/build_cal2.py` |
| Apprentissage | **aucun** (ni codes, ni échelles, ni distillation) | — |
| Codec | codeur par intervalles 64 bits, deux modèles par matrice au choix du plus petit (indice de coquille + rang exact, ou coordonnées par coset avec contexte de parité) ; tables et somme de contrôle dans chaque fichier | `formic_v7/codec/` |
| Tour Vision (460,3 M poids) | E8 au plus proche sans rétroaction à **5 bpw** visés (4,941 réels) | `formic_v7/lab/v7.py : package_vision` |
| Bloc MTP (424,7 M poids) | E8 au plus proche sans rétroaction à **4 bpw** visés (3,994 réels) ; **non évalué** | `package_mtp` |
| Petits tenseurs (normes, convolutions, biais Vision…) | FP16 | `export_end`, `write_raw` |
| Tokenizer, gabarit, configurations | copiés tels quels (22,9 Mo) | `package_aux_files` |

## 2. Ce qui change par rapport à LUMEN Q1-v3.1

| Point | Q1-v3.1 (référence CPU + proposition) | V7-EXP F1 | Effet mesuré |
|---|---|---|---|
| Exécution | petites matrices, CPU | 498 matrices du vrai modèle, GPU, codes identiques à la référence sur les cas de parité | — |
| Calibration | non spécifiée à l'échelle | `cal` + `cal2` (conversations, outils, raisonnement, jeton initial) | **−11,6 %** de KL sur le protocole public et **−8,1 %** sur 600 documents par rapport à la même recette calibrée sur `cal` seul |
| Codec | Python de référence, limite de 128 coquilles | codeur rapide, fichiers réels = coût idéal (−0,1 %), pas de limite de coquille | paquet réel |
| Vision, MTP | « à mettre en œuvre » | quantifiés et comptés dans le paquet ; Vision mesurée sur 24 invites | paquet complet < 10 Go |
| Allocation | DP additive ou modèles conditionnels proposés | **uniforme** | les deux allocations non uniformes essayées font moins bien (ci-dessous) |
| Cible asymétrique, BGCC, polissages, `rate_lambda > 0` | options | **non activés** | cible asymétrique : pilote mitigé, non promue |

## 3. Ce qui a été essayé et écarté dans V7

| Variante | Résultat à octets égaux | Décision |
|---|---|---|
| Allocation par tenseur d'Arki05 transplantée sur E8-LDLQ (`L1_arkialloc`) | +8,1 % de KL sur le protocole public, +18,2 % sur 600 documents par rapport à l'allocation uniforme | écartée |
| Allocation propre par modèle Fisher × pivots LDL | le modèle prédit −53 % pour sa propre allocation et −21 % pour celle d'Arki05, alors que celle-ci mesure +8 % : modèle non prédictif, aucune passe lancée | écartée |
| Pondération Fisher bornée des pas par ligne (±0,5 et ±1 bit ; corps, écrivains, tête) | +12 à +20 % sans rétroaction ; tête seule ±1 bit : −6 % sur le protocole public mais +4 % sur la sélection | écartée |
| Calibration `cal2` seule (`L2_cal2_uniform`) | égalité avec F1 sur le protocole public (−0,1 %, IC [−2,1 ; +2,2] en confirmation), +9,0 % sur 600 documents | non retenue |
| Cible de calibration asymétrique (pilote sur 12 couches) | −7,1 % sur le protocole public, +2,9 % sur la sélection (ridge 0,1) | non promue ; à reprendre séparément |

## 4. Reproduire

```bash
# prérequis : BF16 dans /root/models/qwen3.8-27b, transformers 5.22.2, A100 80 Go, ~90 Go de disque libre
python3 -B formic_arch/data/build_corpus.py ; python3 -B formic_v7/data/build_cal2.py     # calibration
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -B formic_arch/lab/lab_server.py &
formic_arch/lab/submit.sh formic_arch/jobs/030_reinit.py            # professeur, statistiques
formic_arch/lab/submit.sh formic_v7/jobs/108_cal12_exported.py 7200  # passe F1 + paquet texte + relecture (≈ 45 min)
echo F1_cal12_uniform > /workspace/work/SELECTED
formic_arch/lab/submit.sh formic_v7/jobs/110_finalize.py 3600        # Vision, MTP, fichiers annexes, vérification
python3 -I formic_v7/codec/package.py verify /workspace/work/packages/F1_cal12_uniform
```

Le paquet lui-même (9,97 Go) n'est pas dans le dépôt : il est sur le disque du pod (`/workspace/work/packages/F1_cal12_uniform`) et
disparaît avec lui. Son manifeste (1199 entrées, taille et SHA-256 de chaque fichier) est archivé dans `results/package_manifest_F1.json`.

## 5. Limites de la recette

* **Expérimentale.** Jugée sur la fidélité des logits (KL, accord top-1), pas sur des tâches. Aucune garantie de capacités.
* **Pas de moteur d'inférence.** Le paquet se décode vers des poids denses BF16 ; mémoire et vitesse d'exécution du format ne sont pas établies.
* **MTP non évalué** (quantifié à 4 bpw sans rétroaction, jamais exécuté). **Vision** évaluée sur 24 invites seulement, tour quantifiée sans rétroaction.
* **Marge de 25,5 Mo** sous les 10 Go : tout ajout (correcteur, tables d'un moteur, métadonnées) doit être repris sur le débit du texte
  (1 Mo ≈ 0,0003 bpw).
* **Calibration** : composition choisie une fois, non optimisée ; `cal` contient MetaMathQA (parent de GSM8K / MATH). Une seule graine.
* **Allocation uniforme** par défaut de preuve du contraire : deux allocations non uniformes ont échoué, aucune recherche d'allocation
  mesurée avec E8-LDLQ n'a été menée.
* **Mémoire de la passe** : pic de 76,9 Gio sur 80 avec les deux jeux de calibration ; la tête de sortie a nécessité une reprise sur
  point de sauvegarde. Sur un GPU plus petit, réduire la taille des lots de calibration.
* **Format** : le codec rapide n'est pas le codec CPU `LE8CPU3` de la référence v3.1 ; sa description est dans `codec/README.md`.
* **Identité du checkpoint** : révision déclarée par l'opérateur, non attestée par un fichier local.
