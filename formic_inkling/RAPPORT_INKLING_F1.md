# FORMIC-Ink — Campagne Inkling 2 Mini : Formic V1.5 (LUMEN F1 transposée) contre les GGUF d'Unsloth

**22 novembre 2026 — pod A100-SXM4 80 Go.** Mandat : `PROMPT_AGENT_QUANTIFICATION_LUMEN_INKLING2_MINI.md` ; étude :
`DOSSIER_REFLEXION_V8/06_etude_inkling2_mini/ETUDE_FAISABILITE_LUMEN_INKLING2_MINI.md`. Recette retenue : `RECETTE_LUMEN_INKLING.md`.
Résultats bruts : `results/`. Journal des écarts : `results/decisions_log.md`. Branche git `formic-inkling` (rien n'a été poussé).

> Vocabulaire : **mesuré** (cette campagne, fichiers bruts dans `results/`), **publié** (par un tiers, non reproduit), **estimé**,
> **non testé**. KL et accord top-1 ne sont **pas** des pourcentages de capacités.

## Réponse en une page

> **Mise à jour phase 2 (22/11, 19h40).** Après le gel de phase 1 et à la demande de l'opérateur, la variante **V4** (attention,
> embeddings et tête +1 bit, experts compensés, octets égaux) a été sélectionnée sur P-I dev puis confirmée sur un jeu neuf scellé
> (P-I2, 48 fenêtres) : **KL 0,0584 contre 0,0779 pour UD-IQ2_XXS (−25,0 %, IC 95 % [−39,6 ; −11,7])** et −20,7 % contre L0. **Formic
> V1.5 = V4** ; paquet de **8 860 059 425 octets**, vérifié et relu. Seul l'agentique reste derrière UD-IQ2_XXS. Détails : § 11. La
> réponse ci-dessous est celle de la phase 1 (L0), conservée telle quelle.

**Ce qui a été produit (mesuré).** Formic V1.5 = recette F1 transposée (« L0 ») : E8 doublé, pas FP16 par ligne = 0,757082 × RMS,
LDLQ par blocs de 8 avec 7 candidats, Gram par expert sur les entrées de l'élève, aucun apprentissage. Le paquet complet existe :
**8 860 641 865 octets** (texte 8 474 167 056, Vision 196 Mo à 4,93 bpw, drafter 158 Mo à 3,98 bpw, annexes), `package.py verify` OK.
Il a été reconstruit **depuis ses fichiers** par un processus séparé : KL dev 0,080236, identique à 6 décimales au modèle en mémoire.

**Comparaison dans notre graphe, même professeur, P-I (moitié conf, 52 fenêtres, scellée jusqu'au gel) :**

| Modèle | octets texte | KL conf | top-1 | routage (8 experts communs) | Formic contre ce modèle [IC 95 %] |
|---|---:|---:|---:|---:|---|
| UD-IQ2_XXS (octets égaux) | 8 474 257 664 | 0,0819 | 91,07 % | 85,11 % | **−4,2 %** [−13,4 ; +5,0], 38/52 fenêtres |
| UD-Q2_K_XL (+652 Mo) | 9 126 736 640 | 0,0691 | 92,03 % | 86,04 % | +13,6 % [+4,1 ; +23,0] |
| UD-IQ3_XXS (+3,2 Go) | 11 651 330 816 | 0,0370 | 94,85 % | 90,51 % | +112 % |
| **Formic V1.5 (L0)** | **8 474 167 056** | **0,0785** | **92,15 %** | **86,80 %** | |

Sur les 106 fenêtres : Formic 0,0794 contre 0,0841 (−5,6 %, IC [−11,5 ; +0,2], 81/106 fenêtres). Formic est meilleur que
UD-IQ2_XXS sur 7 domaines sur 9 (code, math, encyclopédique, multilingue, long, instructions, web : −8 à −28 %) et **nettement moins bon
sur l'agentique** (+34 % en conf) et le raisonnement (+13 %), les deux domaines dont le texte est généré par le professeur.

**Critère de réussite (§ 7, recopié sans modification) : niveau « Échec »** — le gain contre UD-IQ2_XXS sur P-I conf est de −4,2 %
mais son IC 95 % contient 0. Le critère « Contrôle » (llama.cpp) est **non testé** (volet abandonné sur instruction de l'opérateur).
Mesures secondaires (recoupement) : P-W (Wikitext-2, 40 × 512) KL 0,0647 contre 0,0948 (UD-IQ2_XXS), 0,0763 (UD-Q2_K_XL) ;
P-D32 (283 invites, 32 tokens gloutons) séquences identiques 15,2 % contre 10,6 % et 13,1 %.

**Deux constats techniques importants (mesurés).** (1) Transformers 5.24 charge mal les experts NVFP4 d'Inkling (facteur par expert
arrondi en BF16 : KL 0,0188 avec le checkpoint exact) ; corrigé, le professeur est identique bit à bit à UD-Q8_K_XL (KL 0,00000).
(2) Le graphe d'Inkling a un plancher de sensibilité numérique : deux exécutions mathématiquement équivalentes (`grouped_mm` contre
boucle d'origine, mêmes poids BF16) diffèrent de KL ≈ 0,019. Les KL ci-dessus sont donc à lire comme des écarts dans un graphe fixe.

**Octets par bloc et débit réel (paquet).** Prélude 1,906 Go (2,493 bpw), cœur 3,807 Go (2,490), coda 1,905 Go (2,492), attention
0,147 Go (2,482), vocabulaire 0,291 Go (2,523), petits tenseurs texte 0,419 Go (dont `gate_attn` BF16 0,34 Go), Vision 0,196 Go
(4,927), drafter 0,158 Go (3,984). **Coûts mesurés** : passe L0 4 328 s, V1a 4 210 s, évaluation P-I complète ~250 s, reconstruction
depuis les fichiers 100 s, pic VRAM 77,1 Gio (A100 80 Go).

**Variantes.** V1a (cœur +0,2 bit, bords −0,2 bit) : −2,2 % contre L0 sur dev, IC [−5,7 ; +1,5] ; départage par domaines → L0 retenu.
V2–V5 non testées en phase 1. Une **phase 2** (demandée par l'opérateur après le gel, crédit ajouté) teste V4 et la combinaison V1a+V4
avec un nouveau jeu de confirmation scellé (§ 11).

**Limites.** Protocole maison (106 fenêtres) ; corpus d'Unsloth inconnu (leurs KL publiées 0,0726 / 0,0595 / 0,0269 ne sont que des
repères, protocole inconnu) ; aucune tâche exécutée ; Vision partielle (96 invites, KL) ; drafter non évalué ; une seule graine.

## 1. Méthode de comparaison

* **Professeur.** Le checkpoint publié (`thinkingmachines/Inkling-2-Mini`, révision `b85c3cf`, SHA-256 identiques à ceux du dépôt)
  déquantifié exactement : W4A16, aucun `input_scale` dans le fichier. **Correctif du chargeur** : Transformers 5.24 arrondissait en
  BF16 le facteur FP32 par expert NVFP4 avant la déquantification (les 40 tenseurs NVFP4 correspondent exactement à cette erreur ;
  22,5 % des poids changent après correction). Le professeur corrigé a des experts NVFP4 **identiques bit à bit** à ceux de
  UD-Q8_K_XL d'Unsloth (déquantifiés en FP32 puis arrondis en BF16).
* **Graphe.** `Inkling2LoopForConditionalGeneration` de Transformers 5.24.0 (sans `trust_remote_code`), experts en `grouped_mm`.
  Professeur, Formic et concurrents passent dans **le même graphe**.
* **Concurrents exécutés dans notre graphe.** Les tenseurs quantifiés des GGUF d'Unsloth sont déquantifiés (lecteur d'en-têtes
  maison ; types K/IQ par `gguf` 0.19 ; NVFP4, Q2_0, F8_E4M3 par une implémentation maison écrite d'après la référence de
  llama.cpp) et écrits dans le modèle Transformers. L'échelle FP32 par expert (`ffn_*_exps.scale`) est appliquée. Le rangement est
  vérifié tenseur par tenseur avant toute mesure (cosinus, rapport des normes, égalité bit à bit). Les tenseurs F32 des GGUF
  (normes, convolutions, contrôleurs) ne sont pas chargés : ce sont ceux du professeur.
* **Protocole principal P-I** : 106 fenêtres de 4096 tokens lues depuis un contexte vide (`<|begin_of_text|>` en tête), positions
  2048–4095 scorées (217 088 positions), 9 domaines. KL(professeur ‖ candidat) sur les 200 058 entrées utiles du vocabulaire
  (l'unembed est chargé non rembourré), logits `30·tanh(z/30)` pour les deux, en nats, moyenne par token ; log-softmax par
  morceaux de 1024 positions ; professeur rejoué depuis ses états finaux mis en cache avec sa tête d'origine. Accord top-1 ; accord
  de routage (part des 8 experts communs et part des tokens dont l'expert top-1 change, pour chacune des 48 applications par
  token) ; IC 95 % appariés par bootstrap des fenêtres (méthode V7).
* **Moitiés.** Alternance par domaine : dev 54 fenêtres, conf 52 fenêtres, **scellée** (`/workspace/work_ink/sealed/`) jusqu'au gel.
* **Budget égal.** Octets texte de UD-IQ2_XXS **mesurés sur fichier** : 8 474 257 664 octets de tenseurs (métadonnées exclues).
  Formic : octets réels de ses fichiers texte (E8 + petits tenseurs), mesurés sur disque.
* **Disjonction.** Calibration et P-I disjointes par identifiant de document (intersection vide, `protocol_ink.json`) et par le
  filtre de 13 mots de `build_cal2.py` (bloc `tool_declare` exclu). P-W et P-D32 utilisent des plages d'indices distinctes.
* **Gel.** Règle de sélection écrite et commitée avant toute variante (`results/selection_rule.lock.json`, commit `9829a60`) ; choix
  figé avant lecture de la moitié conf (`results/selection.lock.json`).
* **KL publiées par Unsloth** (0,0726 / 0,0595 / 0,0269) : corpus, domaines et commande inconnus. Elles ne sont citées que comme
  repères « publié, protocole inconnu » et ne sont jamais comparées à nos chiffres.

## 2. Professeur et contrôles (C0, C1) — mesuré

* **Identité.** `thinkingmachines/Inkling-2-Mini`, révision `b85c3cfa615fc413cc9af3379f491b963f3b586a` ; SHA-256 de
  `model-00001-of-00001.safetensors` (19 526 019 168 octets) et `mtp.safetensors` (635 801 416 octets) identiques aux SHA LFS du dépôt.
  1 244 tenseurs : 994 BF16, 154 F32, 48 F8_E4M3, 48 U8. **Aucun `input_scale`** : professeur W4A16 (`teacher_definition.md`).
* **Inventaire** (`checkpoint_headers_inkling.json`) : experts **24,461 G** poids (prélude 6,115 / cœur 12,231 / coda 6,115),
  attention et petits tenseurs 0,676 G, vocabulaire 0,926 G, routeurs 0,007 G, chemin texte **26,070 G** ; Vision 0,318 G ; drafter
  0,318 G. Conforme à l'étude (24,46 / 26,06 G).
* **Disposition.** NVFP4 : deux codes par octet, élément pair dans le quartet bas, échelle E4M3 par bloc de 16, facteur FP32 par
  expert `amax / outer_scale_denominator` ; `w13` entrelacé (gate₀, up₀, …) dans le checkpoint, `[gate ; up]` dans le graphe et dans
  les GGUF ; MXFP8 (couches 20–23) : E4M3 × 2^(s−127) par bloc de 32.
* **Contrôle croisé UD-Q8_K_XL** : experts NVFP4 des couches 0 et 13 (w13, w2) et MXFP8→BF16 de la couche 21 **identiques bit à bit
  en FP32** à notre déquantification du checkpoint.
* **Défaut du chargeur de Transformers 5.24 (trouvé et corrigé).** Sans quantifieur, le chargeur convertit toutes les sources en BF16
  avant `Inkling2LoopDequantizeExperts` : le facteur FP32 par expert est arrondi en BF16. Les 40 tenseurs NVFP4 chargés sont
  exactement ceux d'une déquantification au facteur arrondi ; la correction (`lab/ink_model.py`) modifie 22,5 % des poids NVFP4.
  Sans elle, le « professeur » s'écarte du checkpoint de **KL 0,0188** (top-1 97,6 %) sur P-I dev. Une première passe L0 lancée
  avec le professeur fautif a été interrompue et jetée.
* **Règle FP32/BF16 (§ 4.1.3).** Mesurée en isolant les poids (calcul FP32 des experts dans les deux cas, 18 fenêtres dev) :
  KL(poids FP32 ‖ poids BF16) = **0,0166** → la règle (≤ 1·10⁻⁴) n'est **pas** satisfaite. Mais deux exécutions **aux poids
  identiques** diffèrent autant : boucle d'origine BF16 contre `grouped_mm` BF16 = **0,0188** ; FP32 contre `grouped_mm` BF16 =
  0,0152. **Le graphe d'Inkling a un plancher de sensibilité numérique d'environ 0,015–0,019 de KL entre exécutions
  mathématiquement équivalentes** (ordre des sommes) ; l'effet de l'arrondi BF16 des poids y est indiscernable. Appliquer la
  branche « FP32 à la volée » de la règle n'abaisserait pas ce plancher (FP32 contre boucle BF16 : 0,0188) et doublerait le coût de
  chaque évaluation. **Décision** : professeur = experts exacts arrondis en BF16, exécutés en `grouped_mm` ; tous les modèles
  comparés passent dans ce même graphe déterministe (contrôle : UD-Q8_K_XL, poids identiques, doit donner une KL ≈ 0).
  Conséquence pour la lecture : les KL de ce rapport mesurent l'écart au professeur **dans ce graphe** ; une KL de l'ordre de 0,017
  n'est pas distinguable du bruit d'exécution entre deux moteurs.

## 3. Concurrents dans notre graphe (C2) — mesuré

* Fichiers `unsloth/Inkling-2-Mini-GGUF`, révision `45fd08764f2b0bf16b03a5483a80e5c7aaec8772` (fork « looped-layer »). Octets des
  tenseurs texte (métadonnées exclues), lus sur fichier : UD-IQ2_XXS **8 474 257 664**, UD-Q2_K_XL 9 126 736 640, UD-IQ3_XXS
  11 651 330 816, UD-Q8_K_XL 22 855 473 152.
* Chargement : 218 tenseurs par fichier ; types rencontrés IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ4_XS, IQ4_NL, Q2_0, Q4_K, Q5_K,
  Q6_K, Q8_0, NVFP4, MXFP4, BF16 ; échelle FP32 par expert appliquée. **UD-Q8_K_XL : 218/218 tenseurs identiques bit à bit au
  professeur corrigé, KL 0,00000, top-1 et routage 100,00 %** (contrôle du graphe et de la déquantification).
* Cosinus minimal avec le professeur : 0,874 (UD-IQ2_XXS et UD-Q2_K_XL : `ffn_gate_up_exps` en IQ1_S des couches 1, 2, 6), 0,936
  (UD-IQ3_XXS, IQ2_XXS couche 0). Le seuil de 0,99 du mandat pour « les 2 bits » est inatteignable pour des tenseurs à 1,56 bpw
  (IQ1_S) : un rangement erroné donne un cosinus proche de 0 et un rapport de normes éloigné de 1, ce qui n'est le cas d'aucun tenseur
  (contrôle automatique : cosinus ≥ 0,80 et rapport des normes à ±20 %). Le rangement est en outre prouvé par l'égalité bit à bit de
  UD-Q8_K_XL.
* Durées : chargement 66–83 s, évaluation P-I complète 243–250 s par GGUF.

## 4. Calibration (C3) — mesuré

* Corpus construit : 1 025 documents × 2048 (2 099 200 tokens) ; **utilisé : préfixe entrelacé de 576 documents = 1 179 648 tokens**
  (vitesse mesurée 2 933 tokens/s ; règle de volume écrite avant la passe). Composition du préfixe : 35,1 % conversations, outils et
  raisonnement (UltraChat, ToolACE, glaive-function-calling-v2, OpenThoughts-114k, glaive reasoning-v1-20m, rendus avec le gabarit
  d'Inkling, effort par défaut), 20,0 % code (Python, Rust, C/C++, JS), 15,1 % Wikipedia en, 14,9 % Wikipedia fr/de/es/ru/zh/ja/ar,
  9,9 % MetaMathQA, 5,0 % FineWeb-Edu. Un document brut sur deux commence par `<|begin_of_text|>`.
* Couverture des experts, mesurée sur le **professeur** (tokens × applications) : **6 experts sur 3 072 (0,20 %)** sous le seuil
  n_min = 9 216 pour w13 (couche 0 : 39, 113 ; couche 1 : 114 ; couche 3 : 59 ; couche 18 : 70 ; couche 20 : 33), aucun pour w2.
  Le seuil d'arrêt (5 %) n'est pas atteint : pas de passage à 4,2 M tokens. Les 6 experts sont quantifiés au plus proche.
  Forward du professeur sur la calibration : 383 s.
* Disjonction : intersection vide des identifiants avec P-I ; filtre de 13 mots appliqué (bloc `tool_declare` exclu).

## 5. Passe L0 (C4) — mesuré

* Pilote (Gram du professeur, couches 0 et 5) : rapport bits LDLQ / bits au plus proche = 1,0016 (attention), 1,0028 (w13),
  1,0007 (w2). **Multiplicateur global 0,757082** ; tête au multiplicateur d'ajustement 0,717596.
* Octets texte réels **8 474 167 056** (budget UD-IQ2_XXS 8 474 257 664 : **−0,001 %**), dont petits tenseurs 417 836 611 octets
  (208,7 M poids, bit à bit). Débit idéal E8 : embed 2,485, attention 2,486, w13 2,493, w2 2,493, unembed 2,562 bpw ; moyenne 2,494.
* Durée : **4 328 s** (groupes 265 / 807 / 880 / 955 / 981 s, Gram final + tête ~440 s) ; évaluation P-I complète 252 s ; pic VRAM
  **77,1 Gio**.

## 5 bis. Tableaux P-I (dev, conf) — mesuré


#### P-I dev (54 fenêtres)

| Modèle | octets texte | KL | p99 | top-1 | routage (8 experts communs) | top-1 expert changé | contre UD-IQ2_XXS [IC 95 %] |
|---|---:|---:|---:|---:|---:|---:|---|
| UD-IQ2_XXS | 8 474 257 664 | 0,0862 | 1,120 | 90,50 % | 85,80 % | 17,54 % |  |
| UD-Q2_K_XL | 9 126 736 640 | 0,0712 | 0,947 | 91,65 % | 86,78 % | 16,28 % | -17,4 % [-19,4 % ; -15,4 %], 54/54 fenêtres |
| UD-IQ3_XXS | 11 651 330 816 | 0,0376 | 0,553 | 94,43 % | 91,04 % | 10,99 % | -56,3 % [-61,8 % ; -51,2 %], 54/54 fenêtres |
| Formic L0 | 8 474 167 056 | 0,0802 | 1,029 | 91,61 % | 87,52 % | 15,20 % | -6,9 % [-14,8 % ; +0,7 %], 43/54 fenêtres |
| Formic V1a | 8 474 163 580 | 0,0785 | 1,002 | 91,79 % | 87,88 % | 14,67 % | -8,9 % [-15,8 % ; -2,3 %], 43/54 fenêtres |

#### Par domaine, P-I dev (KL)

| Modèle | agentic | code | math | reasoning | encyclopedic | multilingual | long | instructions | web |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UD-IQ2_XXS | 0,1365 | 0,0299 | 0,0260 | 0,1750 | 0,0673 | 0,0713 | 0,0856 | 0,1234 | 0,0606 |
| UD-Q2_K_XL | 0,1182 | 0,0237 | 0,0200 | 0,1547 | 0,0521 | 0,0569 | 0,0663 | 0,1012 | 0,0473 |
| UD-IQ3_XXS | 0,0901 | 0,0109 | 0,0076 | 0,0992 | 0,0185 | 0,0193 | 0,0237 | 0,0520 | 0,0172 |
| Formic L0 | 0,1638 | 0,0237 | 0,0161 | 0,2005 | 0,0479 | 0,0495 | 0,0588 | 0,1152 | 0,0468 |
| Formic V1a | 0,1634 | 0,0241 | 0,0153 | 0,1797 | 0,0516 | 0,0505 | 0,0598 | 0,1157 | 0,0460 |

#### P-I conf (52 fenêtres, scellée jusqu'au gel)

| Modèle | KL | p99 | top-1 | routage | Formic contre ce modèle [IC 95 %] |
|---|---:|---:|---:|---:|---|
| UD-IQ2_XXS | 0,0819 | 1,074 | 91,07 % | 85,11 % | -4,2 % [-13,4 % ; +5,0 %], 38/52 |
| UD-Q2_K_XL | 0,0691 | 0,944 | 92,03 % | 86,04 % | +13,6 % [+4,1 % ; +23,0 %], 29/52 |
| UD-IQ3_XXS | 0,0370 | 0,534 | 94,85 % | 90,51 % | +112,1 % [+100,9 % ; +126,7 %], 0/52 |
| **Formic L0** | **0,0785** | 0,993 | 92,15 % | 86,80 % | |

#### Par domaine, P-I conf (KL)

| Modèle | agentic | code | math | reasoning | encyclopedic | multilingual | long | instructions | web |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UD-IQ2_XXS | 0,1559 | 0,0362 | 0,0257 | 0,1609 | 0,0608 | 0,0876 | 0,0812 | 0,0971 | 0,0469 |
| UD-Q2_K_XL | 0,1394 | 0,0301 | 0,0198 | 0,1440 | 0,0472 | 0,0709 | 0,0666 | 0,0790 | 0,0381 |
| UD-IQ3_XXS | 0,1103 | 0,0136 | 0,0076 | 0,0952 | 0,0169 | 0,0248 | 0,0218 | 0,0426 | 0,0134 |
| Formic | 0,2086 | 0,0290 | 0,0174 | 0,1815 | 0,0444 | 0,0628 | 0,0587 | 0,0896 | 0,0378 |

#### Octets par bloc (paquet complet)

| Bloc | fichiers | octets | poids E8 | bpw réel E8 |
|---|---:|---:|---:|---:|
| attention | 120 | 147 115 974 | 474 218 496 | 2,4818 |
| drafter | 30 | 158 291 917 | 317 865 984 | 3,9839 |
| experts_coda | 1536 | 1 904 872 893 | 6 115 295 232 | 2,4919 |
| experts_coeur | 3072 | 3 806 888 431 | 12 230 590 464 | 2,4901 |
| experts_prelude | 1536 | 1 906 015 871 | 6 115 295 232 | 2,4934 |
| petits_tenseurs_texte | 827 | 418 622 747 | 0 | — |
| vision | 142 | 195 791 532 | 317 898 752 | 4,9271 |
| vocabulaire | 2 | 291 437 276 | 924 092 928 | 2,5230 |

Total sur disque : 8 860 641 865 octets


## 6. Variantes (C5) et sélection — mesuré

| Variante | Changement | Octets texte | P-I dev | contre L0 [IC 95 %] | domaines gagnés contre L0 | Durée de passe |
|---|---|---:|---:|---|---:|---:|
| L0 | F1 transposée, multiplicateur unique 0,757082 | 8 474 167 056 | 0,08024 | — | — | 4 328 s |
| V1a | cœur +0,2 bit (0,659841), prélude et coda −0,2 bit (0,870665), à octets égaux | 8 474 163 580 | 0,07846 | −2,2 % [−5,7 ; +1,5], 28/54 fenêtres | 4/9 (agentique, math, raisonnement, web) | 4 210 s |

* V1a réduit surtout le raisonnement (0,2005 → 0,1797) ; il dégrade légèrement l'encyclopédique (0,0479 → 0,0516). L'écart global
  est dans l'IC 95 %.
* **Garde-fous de la règle gelée** : aucun candidat ne passe le garde-fou « aucun domaine au-dessus de 1,10 × UD-IQ2_XXS » : le
  domaine agentique est à 1,200 × (L0) et 1,197 × (V1a). Le contrôle provisoire de taille (octets texte + 0,6 Go) était trop prudent
  et échouait aussi pour les deux ; la taille réelle est vérifiée au § 7. Routage : les deux passent. La règle ne prévoyant pas le cas
  « aucun candidat admissible », elle a été appliquée à tous les candidats (décision notée dans `selection.lock.json`).
* **Sélection** (`results/selection.lock.json`, commit `8f0adfa`, avant toute lecture de la moitié conf) : V1a a la KL la plus basse,
  mais l'écart à L0 est inférieur à son IC 95 % ; départage par domaines gagnés : **L0 5, V1a 4 → L0 retenu**.
* Non exécutées faute de crédit : **V2** (second balayage du cœur), **V3** (Gram pondéré par le routage), **V4** (attention et tête
  plus fines), **V5** (calibration doublée) : **non testé**. Vu les résultats par domaine (retard sur l'agentique, où UD-IQ2_XXS garde
  l'attention en Q6_K/Q8_0 et le vocabulaire en IQ4_XS), V4 est la piste la plus prometteuse (estimé, non testé).

## 7. Gel, confirmation et paquet complet (C6) — mesuré

* **Paquet** `/workspace/work_ink/packages/L0` : 7 265 entrées, **8 860 641 865 octets** sur disque (dans 8–9 Go), `verify` OK
  (2,6754 bits par poids codé, tout compris). Manifeste : `results/package_manifest_ink.json`. 6 408 matrices E8 chargées (texte 6 266 +
  Vision 142) et 30 matrices du drafter (non chargées dans ce graphe) ; en-tête par fichier d'expert : 32 o + pas FP16 (4,6 ko) + tables.
* **Reconstruction depuis les fichiers** (décodeur CPU séparé, tous les paramètres mis à NaN puis remplis) : 6 408 matrices et
  803 petits tenseurs en 100 s ; **KL dev 0,080236 = 0,080236 en mémoire** (6 décimales).
* **Vision** (96 invites : documents, graphiques, captures d'écran, photos ; réponse du professeur de 64 tokens, KL aux 64 positions) :
  tour du paquet à 4,93 bpw avec texte du professeur : **KL 0,0109, top-1 97,3 %** (graphiques 0,0074, documents 0,0100, photos
  0,0095, captures d'écran 0,0168) ; paquet complet (Vision + texte quantifiés) : KL 0,0650, top-1 92,7 %.
* **Drafter DSpark** : 317,9 M poids à 3,98 bpw (158 Mo) ; **non évalué** (pas d'exécution du drafter dans Transformers 5.24).
* **Mesures secondaires** (après le gel ; concurrents mesurés et scellés avant) :

  | Modèle | P-W KL | P-W top-1 | P-D32 séquences identiques | P-D32 1re divergence |
  |---|---:|---:|---:|---:|
  | UD-IQ2_XXS | 0,0948 | 87,15 % | 10,6 % | 11,8 |
  | UD-Q2_K_XL | 0,0763 | 88,20 % | 13,1 % | 12,2 |
  | UD-IQ3_XXS | 0,0297 | 92,81 % | 21,2 % | 15,9 |
  | **Formic V1.5 (L0)** | **0,0647** | **89,61 %** | **15,2 %** | **13,9** |

* **Critères de réussite (§ 7 du mandat, recopiés sans modification)** :

  | Niveau | Critère, sur `P-I conf`, à octets texte égaux à UD-IQ2_XXS, dans notre graphe | Résultat |
  |---|---|---|
  | Échec | KL Formic ≥ KL UD-IQ2_XXS, ou IC 95 % du gain qui contient 0 | **oui** : −4,2 %, IC [−13,4 ; +5,0] |
  | Atteint | gain ≥ 10 %, IC 95 % entièrement sous 0 | non |
  | Renforcé | gain ≥ 25 % (hypothèse de l'étude : 25–45 %, **estimée**) | non |
  | Contrôle | même conclusion dans llama.cpp si l'export GGUF BF16 du § 5.3 a été fait | non testé |

## 8. Lecture

* L'hypothèse de l'étude (gain de 25 à 45 % contre Unsloth à octets égaux, **estimée**) n'est pas confirmée sur ce protocole : le gain
  mesuré est de −4 à −7 % selon la moitié, statistiquement non distinct de 0 en confirmation.
* Le gain est réel sur le texte « naturel » (code, math, encyclopédique, multilingue, long, web : −8 à −28 % par domaine) et s'inverse sur
  les textes générés par le professeur (agentique, raisonnement). UD-IQ2_XXS y est aidé par (i) une imatrix calculée sur des
  conversations (`imx_s0.conv.jsonl`, métadonnées du fichier), (ii) l'attention en Q6_K/Q8_0 et le vocabulaire en IQ4_XS, quand
  Formic L0 met tout à ~2,49 bpw. C'est l'objet de la phase 2 (V4).
* Formic ne bat pas UD-Q2_K_XL, plus gros de 652 Mo de texte (+13,6 % de KL en conf).

## 9. Écarts au mandat et limites

Liste complète et raisons : `results/decisions_log.md`. Principaux : volet llama.cpp abandonné (instruction de l'opérateur, disque) ;
P-I à 106 fenêtres au lieu de 216 ; calibration de 1,18 M tokens au lieu de 2,1 M ; capture des Gram par groupes de couches au lieu
de couche par couche ; une seule variante sur trois autorisées ; réponses de P-I générées par le professeur avant le correctif du
chargeur.

Limites : protocole maison ; corpus de la KL publiée par Unsloth inconnu (leurs 0,0726 / 0,0595 / 0,0269 ne sont que des repères
« publié, protocole inconnu ») ; aucune tâche exécutée ; Vision mesurée en KL sur 96 invites seulement ; drafter quantifié mais non
évalué ; une seule graine ; plancher de sensibilité numérique du graphe d'environ 0,017 de KL entre exécutions équivalentes.

## 10. Ce qui n'a pas été exécuté

- Volet llama.cpp : `llama-perplexity`, export GGUF BF16 de Formic, vérification `llama-tokenize` (critère « Contrôle » : non testé).
- Variantes V2, V3, V4, V5 et combinaison.
- Taux d'acceptation du drafter DSpark (non évalué : pas de chemin d'exécution du drafter dans Transformers 5.24).
- Benchmarks de tâches (hors mandat sans accord de l'opérateur) ; runtime d'inférence (projet séparé).

## 11. Phase 2 (après le gel, à la demande de l'opérateur)

### 11.1 Cadre

* Demande de l'opérateur à 16h25 (après le gel de phase 1, moitié conf de P-I déjà lue) : « trouver des solutions pour que la KL de
  Formic V1.5 soit plus basse », crédit ajouté.
* Règle écrite et commitée **avant** tout résultat (`results/selection_rule_phase2.lock.json`, commit `88bde90`) : candidats L0, V1a,
  V4, C1 = V1a + V4 (C1 seulement si V4 bat L0 sur dev) ; sélection sur P-I dev ; **confirmation sur P-I2**, jeu neuf scellé
  (48 fenêtres : 36 brutes code / math / encyclopédique / multilingue / long / web + 6 agentiques glaive + 6 instructions UltraChat
  test, réponses des jeux de données ; disjoint par identifiant ; gelé avant tout résultat, commit `1eea8e1`) ; **remplacement de
  Formic V1.5 seulement si le gain contre L0 sur P-I2 a un IC 95 % entièrement sous 0**.
* La moitié conf de P-I n'est plus aveugle pour la phase 2 : ses chiffres de phase 2 sont donnés à titre d'information seulement.

### 11.2 V4 — mesuré (P-I dev)

| | L0 | V4 |
|---|---:|---:|
| Multiplicateur experts | 0,757082 | 0,787197 |
| Multiplicateur attention, embed (et tête visée) | 0,757082 | 0,393599 (tête ajustée 0,365758) |
| bpw idéal : experts / attention / embed / unembed | 2,49 / 2,49 / 2,48 / 2,56 | 2,44 / 3,44 / 3,41 / 3,52 |
| Octets texte réels | 8 474 167 056 | 8 473 584 587 (−0,008 %) |
| KL P-I dev | 0,08024 | **0,06117** |
| top-1 / routage / top-1 expert changé | 91,61 % / 87,52 % / 15,20 % | 93,04 % / 89,33 % / 13,02 % |
| contre L0 [IC 95 %] | — | **−23,8 %** [−26,7 ; −20,9], 53/54 fenêtres |
| contre UD-IQ2_XXS [IC 95 %] | −6,9 % [−14,8 ; +0,7] | **−29,0 %** [−36,5 ; −21,9], 51/54 |

Par domaine (V4) : agentique 0,1389, code 0,0167, math 0,0121, raisonnement 0,1524, encyclopédique 0,0328, multilingue 0,0336,
long 0,0414, instructions 0,0924, web 0,0303 — tous meilleurs que L0. V4 est aussi en dessous de UD-Q2_K_XL (0,0712), plus gros de
652 Mo de texte. Lecture : ce que l'allocation uniforme de F1 coûte sur Inkling vient surtout de l'attention et du vocabulaire
(3,5 % des poids), que le cœur MoE de 24 G poids ne compense pas : retirer 0,05 bit aux experts pour donner +1 bit à ces chemins
paie massivement. C'est l'inverse du résultat de Qwen, où les allocations non uniformes essayées avaient perdu.

### 11.3 C1 = V1a + V4 — mesuré (P-I dev)

KL 0,06269 (contre L0 −21,9 % [−25,6 ; −17,7] ; contre UD-IQ2_XXS −27,2 % [−35,0 ; −19,3]) : légèrement moins bonne que V4 seul
(0,06117). Octets texte 8 473 641 336 (−0,007 %). Passe 4 236 s.

### 11.4 Sélection et confirmation sur P-I2 — mesuré

Sélection gelée sur P-I dev (`results/selection_phase2.lock.json`, avant lecture de P-I2) : **V4** (L0 0,08024 ; V1a 0,07846 ;
V4 0,06117 ; C1 0,06269). Lecture unique de P-I2 (48 fenêtres, 98 304 positions, scellé) :

| Modèle | octets texte | KL P-I2 | p99 | top-1 | routage | top-1 expert changé |
|---|---:|---:|---:|---:|---:|---:|
| UD-IQ2_XXS | 8 474 257 664 | 0,07786 | 0,907 | 91,18 % | 85,56 % | 17,72 % |
| Formic L0 (V1.5 phase 1) | 8 474 167 056 | 0,07359 | 0,805 | 92,34 % | 87,54 % | 15,04 % |
| **Formic V4** | **8 473 584 587** | **0,05837** | **0,634** | **93,57 %** | **89,34 %** | **12,90 %** |

* **V4 contre L0 : −20,7 %**, IC 95 % [−27,5 ; −14,7], 46/48 fenêtres → la règle de remplacement est remplie : **Formic V1.5 = V4**.
* **V4 contre UD-IQ2_XXS (octets égaux) : −25,0 %**, IC 95 % [−39,6 ; −11,7], 43/48 fenêtres. (L0 contre UD-IQ2_XXS sur P-I2 :
  −5,5 %, IC [−18,7 ; +6,6] — même conclusion qu'en phase 1.)
* Par domaine sur P-I2 (V4 / UD-IQ2_XXS) : code 0,0136 / 0,0242, math 0,0096 / 0,0244, encyclopédique 0,0251 / 0,0497, multilingue
  0,0384 / 0,0769, long 0,0421 / 0,0921, web 0,0445 / 0,0833, instructions 0,0618 / 0,0843, **agentique 0,2318 / 0,1880** (seul domaine
  perdu, voir § 11.6).
* Lecture au regard des critères du mandat (définis sur P-I conf, qui n'est plus aveugle en phase 2) : sur le jeu de confirmation neuf
  et scellé, le gain de −25,0 % avec un IC entièrement sous 0 correspondrait au niveau « Renforcé » (≥ 25 %) à la limite, et au moins
  « Atteint » (≥ 10 %). **Cette lecture porte sur P-I2, pas sur P-I conf** ; elle est donnée comme telle.

### 11.5 Nouveau paquet Formic V1.5 (V4) — mesuré

* `/workspace/work_ink/packages/V4` : **8 860 059 425 octets** sur disque (dans 8–9 Go), `verify` OK ; texte 8 473 584 587 octets,
  Vision 195,8 Mo (4,93 bpw), drafter 158,3 Mo (3,98 bpw, non évalué), annexes. Manifeste : `results/package_manifest_ink_phase2.json`.
* **Reconstruit depuis les fichiers** (6 408 matrices, 803 petits tenseurs, 110 s) : KL P-I dev **0,061174 = 0,061174 en mémoire**.
* P-I conf (**non aveugle en phase 2**, information seulement) : KL 0,0598, top-1 93,38 %, routage 88,72 % ; contre UD-IQ2_XXS −27,0 %
  [−37,2 ; −16,4] (48/52) ; contre UD-Q2_K_XL (+652 Mo) −13,5 % [−24,0 ; −2,4] (46/52) ; contre UD-IQ3_XXS +61,5 %.
* P-W : KL 0,0464 (L0 0,0647 ; UD-IQ2_XXS 0,0948 ; UD-Q2_K_XL 0,0763 ; UD-IQ3_XXS 0,0297). Vision, paquet complet : KL 0,0520,
  top-1 93,4 % (L0 : 0,0650). P-D32 non refait pour V4 (temps).

### 11.6 Le domaine agentique

Seul domaine où UD-IQ2_XXS reste meilleur (P-I2 : 0,232 contre 0,188 ; P-I conf : 0,187 contre 0,156). Diagnostic par type de
segment : `results/diag_agentic.json` (§ 11.7).

### 11.7 Diagnostic agentique — mesuré (P-I, 22 fenêtres agentiques, 22 528 positions)

KL moyenne par type de segment du jeton prédit. « Experts seuls » = attention, embed et unembed remis au professeur.

| Segment | positions | Formic V4 | V4 experts seuls | UD-IQ2_XXS | IQ2 experts seuls |
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

Lecture :
* **Sur tout ce que le modèle génère lui-même** (réponse, raisonnement, appels d'outil), Formic V4 est **2 à 3 fois plus fidèle**
  qu'UD-IQ2_XXS. L'écart défavorable vient entièrement de jetons que le modèle **ne génère jamais en usage agentique** : réponses
  d'outil (≈ 470 des ≈ 690 unités de KL en excès), message système, déclarations JSON, jetons spéciaux. La KL « tout jeton » du
  domaine agentique mesure donc surtout la prédiction de textes d'entrée à haute entropie, pas la qualité des actions de l'agent.
* Sur ces segments, remettre l'attention et le vocabulaire du professeur n'enlève que 14 % (réponses d'outil) à 37 % (jetons spéciaux)
  de la KL de V4 : l'essentiel vient des **experts**, donc vraisemblablement de la **calibration** (≈ 130 k tokens d'outils dans nos
  1,18 M, réponses d'outil courtes et synthétiques ; imatrix d'Unsloth sur un corpus conversationnel, `imx_s0.conv.jsonl`).
* Chez UD-IQ2_XXS, remettre l'attention du professeur **augmente** la KL sur plusieurs segments (compensations croisées entre tenseurs
  quantifiés) : la décomposition n'est pas additive.

Plan d'amélioration (non testé) : (1) déclarer une métrique « positions générées par le modèle » (réponse, raisonnement, appels) à
côté de la KL tout jeton ; (2) calibration agentique enrichie (2 à 3 × plus de conversations outillées avec des réponses d'outil
réelles et longues, rendues au gabarit, réponses du professeur) — **hors mandat, accord de l'opérateur requis** ; (3) lignes des
jetons spéciaux de `embed`/`unembed` à +2 bits (coût < 1 Mo) ; (4) V3 (Gram pondéré par le routage). Coût estimé : une passe
de 75 min + confirmation par variante.
