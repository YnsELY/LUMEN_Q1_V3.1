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
