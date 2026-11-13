# Campagne V7 — état au 13 novembre 2026, 16:20 UTC

## Premier résultat comparable (mesuré)

Concurrent : `Arki05/Qwen3.8-27B-GGUF`, fichier `Qwen3.8-27B-2.82bpw.gguf` (9 477 971 776 octets, rév. `cdcbea9996`), texte seul.
Ses 498 matrices quantifiées occupent 9,456 Go (2,813 bpw). Elles sont déquantifiées avec `gguf-py` et exécutées **dans notre graphe**
(même runtime HF, même professeur BF16, même KL plein vocabulaire). Contrôle du chargement : cosinus avec le BF16 ≥ 0,867 sur tous les
tenseurs, et KL 0,0868 sur le protocole public contre 0,0823 publié sur les 128 fenêtres complètes.

Candidat : recette Lumen inchangée (E8-LDLQ uniforme) ramenée **aux mêmes octets pour les mêmes 498 matrices** : 2,808 bpw de coût
idéal + 0,005 bpw de surcoût de codec mesuré = 2,813 bpw.

| Protocole | Arki05 2,82 bpw | Lumen uniforme 2,808 | Écart |
|---|---:|---:|---|
| **P-A108** : 108 des 128 fenêtres publiques d'Arki05, 4096 tokens, 191 486 positions scorées | 0,0868 (p99 0,910 ; top-1 89,17 %) | **0,0666** (p99 0,622 ; top-1 90,01 %) | **−23,3 %**, IC 95 % [−26,2 ; −19,9] (bootstrap par fenêtre) ; meilleur sur 75 fenêtres / 108 |
| P-F600 : nos 600 documents × 1024 tokens | 0,2116 (top-1 85,6 %) | 0,1104 (top-1 89,4 %) | −48 % (notre calibration vient des mêmes jeux de données : avantage Formic) |
| Long : 12 documents × 8192 tokens | 0,188 | 0,111 | −41 % |

Par domaine sur P-A108 (Arki05 → Lumen) : encyclopédique 0,118 → 0,077 ; multilingue 0,187 → 0,134 ; web 0,091 → 0,069 ;
instructions 0,076 → 0,058 ; contexte long 0,067 → 0,061 ; code 0,026 → 0,025 ; math 0,027 → 0,027 ; **outils 0,037 → 0,038 (+2 %)**.

## Ce que ce résultat ne dit pas

- 108 fenêtres sur 128 : les 20 fenêtres non redistribuées (science, une partie du code) manquent. Non comparable au tableau publié.
- Octets Formic = **coût idéal du codec + surcoût mesuré sur extraits** ; aucun fichier exporté ni exécuté par un moteur. Le concurrent est un GGUF réel.
- Les deux modèles sont exécutés en BF16 dans le runtime HF, pas dans llama.cpp.
- Vision et MTP : hors périmètre des deux côtés.
- Parenté des données : seul le texte fixe du gabarit de conversation est commun entre notre calibration et leurs fenêtres (26 séquences de 13 mots, toutes dans le prompt système, jamais scorées). Notre calibration contient MetaMathQA, dérivé de GSM8K/MATH dont viennent les prompts de leurs fenêtres « math » ; aucun avantage n'apparaît sur ce domaine (0,027 des deux côtés).
- KL n'est pas une mesure de capacités.

## En cours

`L1_arkialloc` : l'allocation par tenseur d'Arki05 appliquée à E8-LDLQ (quantifieur varié, allocation fixée).
