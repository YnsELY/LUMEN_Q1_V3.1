# Plan d'amélioration du domaine agentique — Formic V1.5 (Inkling 2 Mini)

**22 novembre 2026, 19h50 UTC.** Document de passation pour la prochaine session pod. Il résume le diagnostic **mesuré** du
22/11, en tire des hypothèses (**non testées**) et décrit le travail à faire, dans l'ordre, avec les durées et les décisions à
prendre par l'opérateur. Sources : `RAPPORT_INKLING_F1.md` § 11, `results/diag_agentic.json`, `jobs/014_diag_agentic.py`.

> Vocabulaire : **mesuré** (fichiers bruts dans `results/`), **publié**, **estimé**, **non testé**. KL et top-1 ne sont pas des
> pourcentages de capacités.

---

## 1. Point de départ (mesuré)

**Formic V1.5 = recette F1 transposée + V4** : E8-LDLQ, Gram par expert, multiplicateurs 0,787197 pour les experts et 0,393599 pour
l'attention et `embed` ; `unembed` ajusté au budget (0,365758). Paquet `/workspace/work_ink/packages/V4`, 8 860 059 425 octets, à
octets texte égaux à UD-IQ2_XXS (8 474 257 664).

| KL (notre graphe, même professeur) | Formic V1.5 (V4) | UD-IQ2_XXS | Écart |
|---|---:|---:|---|
| P-I2, ensemble (48 fenêtres, scellé, lu une fois) | **0,0584** | 0,0779 | −25,0 % [−39,6 ; −11,7] |
| P-I2, domaine agentique (6 fenêtres) | 0,2318 | **0,1880** | +23 % |
| P-I conf, agentique (non aveugle en phase 2) | 0,1872 | **0,1559** | +20 % |
| P-I dev, agentique | 0,1389 | **0,1365** | +2 % |

L'agentique est le **seul** domaine sur 9 où UD-IQ2_XXS reste devant. Sur tous les autres, V4 a une KL plus basse de 15 à 60 %.

## 2. Diagnostic (mesuré)

Méthode (`jobs/014_diag_agentic.py`) : sur les 22 fenêtres agentiques de P-I (22 528 positions scorées), chaque position est
étiquetée par le segment du **jeton prédit**, d'après les jetons de rôle et de contenu du gabarit d'Inkling (`<|message_tool|>`,
`<|content_invoke_tool_json|>`, `<|content_xml|>` = `tool_declare`, etc.). Quatre modèles : Formic V4, UD-IQ2_XXS, et chacun en
« experts seuls », c'est-à-dire avec l'attention, `embed` et `unembed` remis au professeur.

| Segment du jeton prédit | positions | Formic V4 | V4 experts seuls | UD-IQ2_XXS | IQ2 experts seuls |
|---|---:|---:|---:|---:|---:|
| Texte de réponse du modèle | 5 284 | **0,0234** | 0,0186 | 0,0460 | 0,0480 |
| Raisonnement du modèle | 2 630 | **0,0390** | 0,0303 | 0,0622 | 0,0649 |
| Appels d'outil du modèle (JSON) | 1 551 | **0,0005** | 0,0005 | 0,0014 | 0,0019 |
| Texte de l'utilisateur | 1 875 | **0,189** | 0,172 | 0,275 | 0,292 |
| Déclarations d'outils (JSON) | 7 470 | 0,119 | 0,106 | **0,110** | 0,122 |
| **Réponses d'outil** | 2 349 | 0,684 | 0,590 | **0,484** | 0,543 |
| Message système | 341 | 1,002 | 0,948 | **0,684** | 0,940 |
| Jetons spéciaux | 1 017 | 0,198 | 0,124 | **0,157** | 0,208 |
| **Domaine entier** | 22 528 | 0,1609 | 0,1389 | **0,1453** | 0,1640 |

(Le domaine raisonnement, mesuré dans le même job : V4 0,1436 contre UD-IQ2_XXS 0,1679 ; raisonnement du modèle 0,0302 contre 0,0486.)

### Ce que cela montre

1. **Sur tout ce que l'agent génère lui-même, Formic V4 est 2 à 3 fois plus fidèle qu'UD-IQ2_XXS** : réponses 0,023 contre 0,046,
   raisonnement 0,039 contre 0,062, appels d'outil 0,0005 contre 0,0014.
2. **L'écart défavorable vient de jetons que le modèle ne génère jamais en usage agentique.** En unités de KL (KL × positions), V4
   perd environ 480 unités sur les réponses d'outil, 108 sur le message système, 70 sur les déclarations JSON et 42 sur les jetons
   spéciaux, et en regagne sur le texte du modèle et de l'utilisateur. Les réponses d'outil pèsent à elles seules environ 70 % de
   l'excès.
3. **Sur ces segments, l'écart vient surtout des experts.** Remettre l'attention et le vocabulaire du professeur n'enlève que 14 %
   (réponses d'outil) à 37 % (jetons spéciaux) de la KL de V4. Pour les jetons spéciaux, en revanche, l'attention et le vocabulaire
   comptent nettement : 0,198 → 0,124.
4. **La décomposition n'est pas additive chez UD-IQ2_XXS** : remettre l'attention du professeur y augmente la KL sur plusieurs
   segments (0,145 → 0,164 sur le domaine), signe de compensations entre tenseurs quantifiés. On ne peut donc pas conclure qu'Unsloth
   « gagne grâce à l'attention ».

### Hypothèses (non testées)

* **H1 — calibration.** Notre calibration (576 documents, 1,18 M tokens) ne contient qu'environ 65 documents de conversations
  outillées (ToolACE, glaive), soit environ 130 k tokens, avec des réponses d'outil courtes et synthétiques. L'imatrix d'Unsloth a
  été calculée sur un corpus conversationnel (`quantize.imatrix.dataset = imx_s0.conv.jsonl`, 8 238 morceaux, métadonnées du GGUF,
  **publié**). Les experts choisis par le routeur sur du texte d'outil (JSON renvoyé par une API, journaux, tableaux) ont donc des
  Gram peu représentatifs. C'est l'hypothèse principale, cohérente avec le point 3.
* **H2 — jetons spéciaux.** Les ~60 jetons spéciaux du gabarit sont des lignes rares de `embed` et `unembed`, quantifiées comme les
  autres. Les remettre en BF16 dans V4 fait passer leur KL de 0,198 à 0,124 (attention comprise).
* **H3 — métrique.** La KL « tout jeton » du domaine agentique mesure surtout la prédiction de textes d'entrée à haute entropie
  (réponses d'outil, système), que le modèle ne produit pas en usage. Une métrique restreinte aux **positions générées par le modèle**
  représente mieux la qualité d'un agent quantifié.

## 3. Travail à faire à la prochaine session pod

### Étape 0 — remettre le pod en état (~15 min)

1. Cloner la branche `formic-inkling` du dépôt `LUMEN_Q1_V3.1` dans `/workspace/LUMEN_Q1_V3.1`.
2. Télécharger le modèle dans `/root/models/inkling-2-mini` (révision `b85c3cfa615fc413cc9af3379f491b963f3b586a`, 20,2 Go) et vérifier
   les SHA-256 (`results/checkpoint_headers_inkling.json`).
3. Restaurer depuis la sauvegarde locale `work_ink/data/` (protocoles, caches du professeur, calibration) et `work_ink/packages/V4/` ;
   restaurer `work_ink/sealed/` (résultats scellés). Ne retélécharger UD-IQ2_XXS (8,5 Go) que si une mesure contre lui est
   nécessaire ; les autres GGUF ne sont pas utiles.
4. Lancer `python -B formic_inkling/lab/lab_server_ink.py` : il charge le professeur **avec le correctif NVFP4** (vérifier la ligne
   « défaut expliqué … True » dans le journal). Contrôle rapide : `jobs/006b_precision.py` n'est pas à refaire ; recharger le paquet
   V4 et vérifier que la KL P-I dev vaut 0,061174.

### Étape 1 — nouvelle métrique, sans nouvelle passe (~20 min)

* Ajouter à `lab/ink_eval.py` une KL restreinte aux **positions générées par le modèle** (segments `model_text`,
  `model_thinking`, `model_tool_call` ; étiquetage de `jobs/014_diag_agentic.py`), rapportée pour les domaines agentique,
  raisonnement et instructions, **à côté** de la KL tout jeton (jamais à sa place).
* La recalculer sur P-I dev pour L0, V4, UD-IQ2_XXS et UD-Q2_K_XL (états du professeur en cache : rien à régénérer).
* La déclarer dans un nouveau verrou **avant** toute variante (`results/selection_rule_phase3.lock.json`).

### Étape 2 — construire un nouveau jeu de confirmation scellé P-I3 (~40 min)

P-I conf et P-I2 ont été lus : il faut un jeu neuf pour confirmer la phase 3.
* 48 fenêtres, dont **24 agentiques** : conversations outillées multi-tours avec de vraies réponses d'outil longues (sources à
  plages disjointes : ToolACE j ≥ 12 000, glaive j ≥ 60 000, et si possible un jeu de traces d'agents réelles), réponses de
  l'assistant **générées par le professeur** (effort `high`, glouton, lots de 16 au plus pour éviter l'OOM), plus 24 fenêtres brutes
  des autres domaines.
* Disjonction par identifiant avec P-I, P-I2, P-W, P-D32 et toutes les calibrations ; hash et scellement avant tout résultat.

### Étape 3 — variantes (une passe de ~75 min chacune, plus ~5 min d'évaluation)

À déclarer dans le verrou de phase 3 avant de lancer :

| Id | Changement (à octets texte égaux) | Hypothèse visée | Accord de l'opérateur |
|---|---|---|---|
| **A1** | V4 + **calibration agentique enrichie** : remplacer environ 10 % de la calibration (les documents « prose » en surnombre) par des conversations outillées réelles et longues (réponses d'outil de 500 à 3 000 tokens, multi-tours), rendues au gabarit, réponses de l'assistant générées par le professeur ; viser au moins 400 k tokens d'outils | H1 | **oui** : le mandat interdit d'optimiser la composition de la calibration au-delà de V5 |
| A2 | V4 + lignes des jetons spéciaux de `embed` et `unembed` à +2 bits (multiplicateur ÷4 sur ces lignes ; coût < 1 Mo) | H2 | non (variante d'allocation à octets égaux) |
| A3 | V4 + V3 du mandat : Gram des experts pondéré par (poids de routage)² | erreur réellement transmise | non |
| A4 | V4 + `gate_attn` quantifié en E8 au multiplicateur de l'attention (libère ~0,29 Go réinvesti dans les experts) | budget | **oui** (le mandat impose de le garder en BF16) |

Ordre recommandé : **A1** (levier principal selon le diagnostic), puis A2 (peu coûteux, combinable), puis une combinaison A1 + A2.
A3 et A4 seulement si le crédit le permet.

Implémentation :
* A1 : nouveau script `data/build_cal_agentic_ink.py` (dérivé de `build_data_ink.py`), génération des réponses du professeur par un
  job dérivé de `jobs/002b_teacher_gen_more.py`, puis remplacement par position dans `cal_ink.pt` pour garder 576 documents et la
  même durée de passe.
* A2 : dans `ink_quant.q_dense`, pas par ligne divisé par 4 pour les identifiants spéciaux (liste dans `tokenizer_config.json`),
  pour `llm.embed` et `llm.unembed` ; le calage du multiplicateur des experts absorbe le coût.
* Lancer chaque variante comme `jobs/011_V4.py` (même calibration de référence, même pilote, même ajustement de la tête).

### Étape 4 — sélection, confirmation, paquet (~35 min)

* Sélection sur P-I dev par la règle de phase 3 : KL tout jeton la plus basse, sous garde-fou « KL des positions générées en
  agentique ≤ celle de V4 », routage ≥ V4 − 1 point.
* Lecture unique de P-I3 sur le gagnant, V4 et UD-IQ2_XXS ; remplacement de Formic V1.5 seulement si le gain contre V4 a un IC 95 %
  entièrement sous 0 **et** que l'agentique ne se dégrade pas.
* Paquet complet (Vision 5 bpw, drafter 4 bpw, annexes), `verify`, reconstruction depuis les fichiers (KL identique à 6 décimales),
  rapport (`RAPPORT_INKLING_F1.md`, nouvelle section 12) et recette.

## 4. Budget de la prochaine session (estimé)

| Programme | Durée pod | Coût à 2,21 $/h |
|---|---:|---:|
| Étapes 0 à 2 (remise en état, nouvelle métrique, P-I3) | ~1h15 | ~3 $ |
| + A1 (une variante confirmée et empaquetée) | ~3h30 au total | ~8 $ |
| + A2 et la combinaison A1 + A2 | ~6h au total | ~13 $ |
| + A3 et A4 | ~9h au total | ~20 $ |

## 5. Critère de réussite proposé pour la phase 3 (à geler avant les résultats)

Sur P-I3 (scellé), à octets texte égaux : (a) **KL du domaine agentique ≤ celle d'UD-IQ2_XXS**, IC 95 % apparié par fenêtre ; (b)
KL tout jeton ≤ celle de V4, sans perte significative sur aucun autre domaine ; (c) KL des positions générées par le modèle en
agentique au moins aussi basse que V4.

## 6. Pièges connus (rencontrés le 22/11)

* Transformers 5.24 charge mal les experts NVFP4 : toujours passer par `lab/ink_model.py` (correctif inclus).
* Le graphe a un plancher de sensibilité numérique d'environ 0,017–0,019 de KL entre exécutions équivalentes : comparer uniquement
  dans le même graphe (`grouped_mm`, BF16).
* Génération du professeur : lots de 16 au plus (OOM au pré-remplissage à 32–48), ~0,25 s par pas.
* La génération Vision du professeur prend ~22 min : réutiliser `work_ink/data/vision/vision_teacher.pt`.
* `pkill -f <motif>` tue aussi le shell qui contient le motif : tuer par PID.
* Les `skip()` de `datasets` en streaming lisent tout ce qui précède : préférer des décalages courts et le contrôle par identifiant.
