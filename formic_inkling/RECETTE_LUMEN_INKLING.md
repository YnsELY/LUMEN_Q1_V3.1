# Recette retenue — LUMEN « Formic V1.5 » pour Inkling 2 Mini (F1 transposée)

**22 novembre 2026.** Recette de quantification sélectionnée à l'issue de la campagne FORMIC-Ink pour
`thinkingmachines/Inkling-2-Mini` (révision `b85c3cfa615fc413cc9af3379f491b963f3b586a`). Nom du modèle quantifié : **Formic V1.5**.
Statut : **expérimentale**. Elle produit un paquet réel relu depuis ses fichiers et évalué ; elle n'a pas de moteur d'inférence
dédié et n'est jugée que sur la fidélité des logits. Résultats et preuves : `RAPPORT_INKLING_F1.md`, `results/`.

> Vocabulaire : **mesuré** (cette campagne, fichiers bruts dans `results/`), **publié** (par un tiers, non reproduit),
> **estimé**, **non testé**. KL et accord top-1 ne sont pas des pourcentages de capacités.

## 1. La recette en une page

| Élément | Valeur retenue | Code |
|---|---|---|
| Professeur | checkpoint publié **déquantifié exactement** (W4A16) : NVFP4 = e2m1 × fp8(bloc de 16) × (amax/dénominateur en **FP32**), MXFP8 = e4m3 × 2^(s−127) ; arrondi BF16 (RNE). **Correctif du chargeur de Transformers 5.24** (facteur arrondi en BF16) | `lab/ink_model.py` |
| Architecture | **inchangée** (24 couches, cœur 6–17 bouclé 3 fois, MoE 128 × 8, Vision, drafter) | — |
| Matrices E8 du chemin texte | `embed`, `unembed` (200 058 lignes), et par couche : `q_proj`, `k_proj`, `v_proj`, `relative_position_bias.linear` (wr_du), `o_proj.linear`, puis **une matrice par expert** : `w13` (gate+up, 2304 × 2304) et `w2` (2304 × 1152) — 6 266 matrices | `lab/ink_quant.py : matrix_list` |
| Quantifieur | réseau **E8**, codes doublés, poids décodé = code × pas / 2, décodé canonique `bf16_rne(f32(U)·(f32(pas)/2))` | `lab/formic_q.py` |
| Pas | un pas **FP16 par ligne** = multiplicateur × RMS de la ligne | `row_steps` |
| Multiplicateur | **RÉSULTAT** | `jobs/006_L0.py` |
| Compensation | **LDLQ par blocs de 8 colonnes**, 7 candidats, amortissement 1 % de la diagonale moyenne ; experts traités par lots de 16 (vectorisation **identique bit à bit** à `ldlq_rows`, test unitaire) | `ldl_prepare_batched`, `ldlq_rows_batched` |
| Gram des experts | **par expert, non pondéré**, comptes = tokens × applications (×3 dans le cœur) ; seuil n_min = 4·2304 (w13), 4·1152 (w2) → sinon E8 au plus proche | `Capture._exp`, `q_experts` |
| Objectif et ordre | erreur de sortie locale, symétrique, entrées de l'**élève** capturées sur le vrai graphe Transformers (forward complet, arrêt anticipé) ; **groupes** [0–5], [6–9], [10–13], [14–17], [18–23] ; phase A (entrées d'attention, wo, w13), phase B (w2, Gram recalculé après quantification des w13) ; Gram du cœur = somme des 3 passes | `run_pass` |
| `embed` | E8 au plus proche sans rétroaction, multiplicateur global | idem |
| `unembed` | LDLQ sur le Gram des états finaux normalisés de l'élève complet ; multiplicateur ajusté pour que les octets texte réels égalent le budget (UD-IQ2_XXS) | idem |
| Non quantifiés | routeurs (poids + biais), contrôleurs de boucle, requêtes inter-passes et `attn_res`, normes, puits, `rel_logits_proj`, `gate_attn`, `sconv`, clips : **bit à bit** dans le type du checkpoint (BF16/FP32) | `small_tensors`, `write_small` |
| Calibration | **1 179 648 tokens** (576 × 2048), préfixe entrelacé d'un corpus de 2,1 M tokens : 35 % conversations + outils + raisonnement (UltraChat, ToolACE, glaive-function-calling-v2, OpenThoughts-114k, glaive reasoning-v1-20m) rendus avec le gabarit d'Inkling, 20 % code (Python, Rust, C/C++, JS), 15 % Wikipedia en, 15 % Wikipedia fr/de/es/ru/zh/ja/ar, 10 % MetaMathQA, 5 % FineWeb-Edu ; un document brut sur deux commence par `<|begin_of_text|>` | `data/build_data_ink.py` |
| Apprentissage | **aucun** | — |
| Codec | codec FE8 de V7 (inchangé), un fichier par matrice | `codec/` |
| Vision | E8 au plus proche à **5 bpw** visés | `lab/ink_package.py : package_vision` |
| Drafter DSpark | E8 au plus proche à **4 bpw** visés ; **non évalué** | `package_mtp` |
| Annexes | tokenizer, `tiktoken/`, gabarit, configurations, copiés tels quels | `package_aux` |

## 2. Ce qui change par rapport à F1 (Qwen3.8-27B)

| Point | F1 (Qwen) | Formic V1.5 (Inkling) | Raison |
|---|---|---|---|
| Professeur | BF16 publié | checkpoint NVFP4/MXFP8 déquantifié exactement, correctif du chargeur | aucun BF16 publié |
| Passe séquentielle | `Seq` couche par couche | forwards complets du vrai graphe + crochets + arrêt anticipé, **par groupes de couches** | résidus d'attention, cœur bouclé ; budget de temps |
| MLP | dense | **Gram par expert**, seuil de couverture | MoE 128 × 8 |
| Cœur bouclé | — | Gram = somme des 3 passes | 12 couches appliquées 3 fois |
| Logits | sans softcap | `30·tanh(z/30)` rejoué, 200 058 entrées | architecture |
| Calibration | 409 600 tokens | 1 179 648 tokens | MoE (données par expert) ; plafonnée par le temps |
| Petits tenseurs | FP16 | type du checkpoint, bit à bit | — |

## 3. Ce qui a été essayé

RÉSULTAT (variantes C5).

## 4. Reproduire

```bash
cd /workspace/LUMEN_Q1_V3.1/formic_inkling
python -B data/build_data_ink.py pi && python -B data/build_data_ink.py cal      # P-I (textes, invites) et calibration
python -B data/build_secondary_ink.py && python -B data/build_vision_ink.py       # P-W, P-D32, Vision
python -B lab/lab_server_ink.py &                                                  # professeur en VRAM (chargeur corrigé)
for j in 002_teacher_gen 002b_teacher_gen_more 003_protocol_teacher 003c_teacher_cache 004_teacher_fp32_check \
         006_L0 006b_precision 007_V1a 008_competitors 010_finalize; do lab/submit_ink.sh jobs/$j.py 20000; done
python -B codec/package.py verify /workspace/work_ink/packages/<VID>
```

## 5. Limites

- Protocole maison (P-I, 106 fenêtres) ; le corpus de la KL publiée par Unsloth est inconnu : leurs chiffres ne sont que des repères.
- Aucune tâche exécutée ; Vision mesurée en KL sur 96 invites ; drafter non évalué ; une seule graine.
- Capture par groupes de couches (écart déclaré) ; calibration plafonnée à 1,18 M tokens.
- Volet llama.cpp non testé.
