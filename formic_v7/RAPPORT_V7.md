# FORMIC-Max — Campagne V7 : comparaison aux concurrents et recette retenue

**13 novembre 2026 — pod RunPod A100 80 Go.** Mandat : `PROMPT_AGENT_FORMIC_COOPTIMISATION_KL_V7.md`.
Recette retenue : `RECETTE_LUMEN_V7_EXP.md`. Audit du pipeline : `docs/AUDIT_PIPELINE.md`. Résultats bruts : `results/`.

> Vocabulaire : **mesuré** (cette campagne, fichiers bruts dans `results/`), **publié** (par un tiers, non reproduit), **estimé**, **non testé**.

## Réponse en une page

**Qu'avons-nous amélioré ?** La recette Lumen elle-même n'a presque pas changé : architecture d'origine, E8 + LDLQ, allocation
uniforme. Deux choses ont changé : les **données de calibration** (ajout de conversations, d'appels d'outils et de traces de
raisonnement, 409 600 tokens au lieu de 147 456) et l'existence d'un **paquet réel** (codec rapide, fichiers relus et réévalués).
La calibration apporte −11,6 % de KL sur le protocole public et −8,1 % sur nos 600 documents, à octets égaux.

**Quel concurrent battons-nous, dans quelles conditions ?** À budget texte égal (≈ 9,45 Go pour les mêmes 498 matrices), dans le même
graphe d'exécution, avec le même professeur BF16 et la même KL plein vocabulaire :

| Protocole | Arki05 2,82 bpw | Axolotl XQD Q2 | **Formic F1** | F1 contre Arki05 [IC 95 %] |
|---|---:|---:|---:|---|
| **Protocole public d'Arki05, moitié de confirmation** (52 fenêtres scorées, scellée jusqu'au gel) | 0,0915 | 0,1292 | **0,0613** | **−33,1 %** [−36,6 ; −28,7], meilleur sur 51 fenêtres / 52 |
| Protocole public, 108 fenêtres sur 128 | 0,0868 | 0,1200 | **0,0596** | −31,3 % [−33,8 ; −28,5], meilleur sur 102 / 108 |
| `conf2` : 300 documents neufs × 1024 tokens (même distribution que notre calibration) | 0,2095 | 0,2112 | **0,1041** | −50,3 % [−53,3 ; −47,6], meilleur sur 297 / 300 |

Accord top-1 sur les 108 fenêtres : 89,17 % (Arki05), 87,32 % (XQD), **90,53 %** (F1). F1 est meilleur dans les huit domaines du
protocole public (de −13 % en math à −45 % en encyclopédique, −23 % en appels d'outils). L'objectif du mandat (≥ 5 %, renforcé ≥ 10 %,
confirmé par une analyse appariée avec garde-fous par domaine) est **atteint sur la fidélité textuelle**.

**Que manque-t-il pour livrer Formic-Max complet sous 10 Go ?** Le paquet complet existe : **9 974 484 649 octets**, texte, embeddings,
tête, Vision, MTP, petits tenseurs, tokenizer et configurations compris, vérifié fichier par fichier. Il manque : (1) un **moteur
d'inférence** pour ce format (aujourd'hui on décode vers des poids denses) ; (2) une **évaluation de capacités** (aucun benchmark de
tâche) ; (3) l'évaluation du **MTP**, quantifié mais jamais exécuté ; (4) une évaluation Vision au-delà de 24 invites ; (5) la
comparaison dans le moteur des concurrents (llama.cpp) et sur leurs 20 fenêtres non redistribuées.

Jalons internes du mandat (KL sur protocole Formic figé) : nos 600 documents donnent 0,1015 pour F1 (0,1104 avant) ; le jalon
« ≤ 0,100 » n'est **pas** atteint sur ce protocole, et « ≤ 0,080 » n'est atteint que sur le protocole public (0,0596), qui est une autre mesure.

## 1. Méthode de comparaison

* **Concurrents exécutés dans notre graphe.** Les tenseurs quantifiés des GGUF sont déquantifiés avec la bibliothèque de référence
  `gguf-py` et écrits dans le modèle Transformers (`lab/v7.py : load_gguf`). Les têtes de valeur des couches Gated DeltaNet sont
  stockées dans un autre ordre par llama.cpp ; la permutation a été identifiée et vérifiée (cosinus avec le BF16 ≥ 0,86 sur tous les
  tenseurs après correction, 0,04 avant). Contrôle : Arki05 2,82 bpw obtient 0,0868 sur les 108 fenêtres disponibles, pour 0,0823
  **publié** sur les 128. Les deux nombres ne portent pas sur le même ensemble ; l'ordre de grandeur est cohérent.
* **Protocole public P-A108** : les 108 fenêtres redistribuées (sur 128) d'`Arki05/Qwen3.8-27B-GGUF/eval`, 4096 tokens lus depuis un
  contexte vide, positions 2048–4094 scorées selon leur masque (191 486 positions), KL(BF16 ‖ candidat) plein vocabulaire en nats,
  moyenne par token. Les 20 fenêtres manquantes (science, une partie du code) ne sont pas redistribuables.
* **Budget égal** : mêmes octets pour les mêmes 498 matrices du chemin texte. Arki05 : 9,456 Go ; XQD : 9,400 Go (+ davantage de
  tenseurs laissés en 32 bits, fichier de 9,469 Go) ; F1 : 9,449 Go **mesurés sur fichiers**.
* **Séparation des données.** Calibration (`cal`, `cal2`), développement (72 + 72 + 600 documents, 12 documents longs, moitié « dev »
  du protocole public) et confirmation (moitié « conf » du protocole public, scellée ; `conf2`, 300 documents jamais évalués avant le gel)
  sont disjoints par document. `cal2` exclut tous les identifiants de documents des fenêtres d'Arki05 et toute séquence de 13 mots commune
  hors texte fixe du gabarit de conversation.
* **Gel.** La règle de sélection a été déclarée avant les résultats de F1 et de la cible asymétrique (`results/v7_selection_rule.lock.json`,
  commit `41418f9`) ; le choix a été figé avant toute lecture de la confirmation (`results/v7_selection.lock.json`, commit `9583236`).

## 2. Résultats de développement (moitié « dev » du protocole public, 600 documents, contexte long)

| Variante | bpw texte | P-A108-dev | contre Arki05 [IC 95 %] | contre L0 [IC 95 %] | 600 documents | contre L0 [IC 95 %] | 8192 tokens |
|---|---:|---:|---|---|---:|---|---:|
| Arki05 2,82 bpw (concurrent) | 2,813 | 0,0824 | | | 0,2116 | | 0,188 |
| Axolotl XQD Q2 sans MTP (concurrent) | 2,799 | 0,1113 | +35,2 % [+23,1 ; +45,8] | | 0,2181 | | 0,164 |
| `L0` : recette Lumen inchangée, calibration `cal` | 2,808 | 0,0657 | −20,2 % [−24,3 ; −15,4] | | 0,1104 | | 0,111 |
| `L1` : allocation par tenseur d'Arki05 sur E8-LDLQ | 2,803 | 0,0710 | −13,8 % [−17,3 ; −9,7] | +8,1 % [+3,4 ; +12,4] | 0,1305 | +18,2 % [+10,7 ; +27,1] | 0,130 |
| `L2` : calibration `cal2` seule | 2,808 | 0,0567 | −31,1 % [−33,8 ; −28,0] | −13,7 % [−16,9 ; −10,8] | 0,1115 | +1,0 % [−4,1 ; +7,0] | 0,116 |
| **`F1` : calibration `cal` + `cal2`, paquet réel** | 2,811 | 0,0581 | −29,4 % [−32,4 ; −26,2] | −11,6 % [−15,4 ; −8,1] | 0,1015 | −8,1 % [−13,0 ; −3,1] | 0,109 |

Par domaine sur P-A108-dev (Arki05 → L0 → F1) : code 0,026 → 0,028 → 0,022 ; encyclopédique 0,099 → 0,070 → 0,063 ; instructions
0,072 → 0,057 → 0,049 ; contexte long 0,069 → 0,062 → 0,054 ; math 0,027 → 0,026 → 0,023 ; multilingue 0,170 → 0,127 → 0,114 ;
**outils 0,023 → 0,029 → 0,021** ; web 0,094 → 0,073 → 0,068.

Ce que ces mesures établissent :

1. **Le quantifieur Lumen était déjà devant à octets égaux** (`L0` : −20 % contre Arki05), sauf sur les appels d'outils (+26 %) et le
   code (+8 %), domaines absents ou peu représentés dans la calibration d'origine.
2. **Leur avantage n'était pas l'allocation.** Leur propre allocation par tenseur, appliquée à notre quantifieur, le dégrade (+8 % et
   +18 %). Une allocation non uniforme n'a d'intérêt que si elle est mesurée avec le quantifieur auquel elle s'applique.
3. **La calibration est le levier.** `cal2` seule efface le retard sur les outils et le code (0,029 → 0,023 ; 0,028 → 0,020) et gagne
   partout sur le protocole public, sans rien coûter sur nos 600 documents. Réunir les deux jeux (`F1`) garde ce gain et améliore aussi
   les 600 documents (−8 %) et le contexte long.
4. **Aucun apprentissage** n'entre dans ces résultats : une seule passe de quantification.

### Essais écartés

| Essai | Mesure | Verdict |
|---|---|---|
| Pondération Fisher bornée des pas par ligne (sans rétroaction, 2,808 bpw) | référence 0,1124 (P-A108-dev) / 0,290 (36 documents) ; corps ±0,5 bit : 0,1258 / 0,314 ; corps ±1 bit : 0,1347 / 0,319 ; écrivains ±1 bit : 0,1187 / 0,355 ; tête ±1 bit : 0,1054 / 0,302 ; tête ±2 bits : 0,1510 / 0,329 | écartée ; seule la tête ±1 bit gagne sur un des deux jeux |
| Allocation par modèle Fisher × pivots LDL | corrélation 0,65 avec l'allocation d'Arki05 ; le modèle prédit −21 % pour l'allocation d'Arki05, qui mesure +8 % | modèle non prédictif, aucune passe lancée |
| Cible de calibration asymétrique (recette v3.1 §4), pilote sur les 12 premières couches | symétrique 0,00848 (P-A108-dev) / 0,01587 (72 documents) ; ridge 0,01 : 0,00827 / 0,01671 ; ridge 0,1 : 0,00788 / 0,01633 | −7,1 % sur un jeu, +2,9 % sur l'autre ; critère de promotion déclaré non rempli ; passe 1,5 fois plus longue |

## 3. Sélection

Règle déclarée : KL la plus basse sur P-A108-dev parmi les candidats respectant trois garde-fous (600 documents ≤ 1,05 × L0 ; contexte
long ≤ 1,10 × L0 ; aucun domaine de P-A108-dev à plus de 10 % au-dessus d'Arki05).

* `L0` échoue au garde-fou par domaine (outils +26 %). `L2` et `F1` passent les trois.
* Application littérale : `L2` (0,0567 contre 0,0581). **Écart non résolu statistiquement** : F1 contre L2 = +2,4 %, IC 95 % [−0,4 ; +5,4].
* La règle ne prévoyait pas de départage. **Amendement, écrit et commité avant toute lecture de la confirmation** : départage sur le plus
  grand jeu de développement, où F1 est meilleur de 9,0 % [−14,0 ; −4,3] et dans les six domaines. **F1 est retenu.**
* La confirmation donne raison à l'égalité sur le protocole public : F1 contre L2 = −0,1 % [−2,1 ; +2,2] sur la moitié scellée.

## 4. Confirmation (lue après le gel)

**Moitié scellée du protocole public** (52 fenêtres scorées) :

| Domaine | Arki05 | XQD | **F1** | F1 contre Arki05 | Top-1 Arki05 → F1 |
|---|---:|---:|---:|---:|---|
| code | 0,0262 | 0,0209 | 0,0191 | −27 % | 96,5 → 97,1 |
| encyclopédique | 0,1382 | 0,1522 | 0,0764 | −45 % | 85,4 → 88,2 |
| instructions | 0,0792 | 0,0840 | 0,0536 | −32 % | 92,1 → 93,4 |
| contexte long | 0,0646 | 0,0825 | 0,0513 | −21 % | 89,2 → 90,1 |
| math | 0,0281 | 0,0277 | 0,0246 | −13 % | 95,7 → 95,7 |
| multilingue | 0,2078 | 0,3103 | 0,1293 | −38 % | 79,4 → 83,5 |
| outils | 0,0463 | 0,0472 | 0,0355 | −23 % | 94,8 → 95,0 |
| web | 0,0863 | 0,2122 | 0,0615 | −29 % | 86,1 → 87,1 |
| **ensemble** | **0,0915** | **0,1292** | **0,0613** | **−33,1 %** [−36,6 ; −28,7] | 89,03 → 90,50 |

F1 contre XQD : −52,6 % [−61,4 ; −43,9], meilleur sur 49 fenêtres / 52. `L2` : 0,0613 également (−33,0 % contre Arki05).
Sur les 108 fenêtres : F1 0,0596, 99ᵉ centile 0,548 (Arki05 0,910 ; XQD 1,239).

**`conf2`** (300 documents neufs, F1 **reconstruit depuis ses fichiers**) :

| | chat | code | prose | math | multilingue | science | **ensemble** | top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arki05 2,82 bpw | 0,410 | 0,386 | 0,113 | 0,094 | 0,144 | 0,109 | 0,2095 | 85,63 % |
| XQD Q2 | 0,447 | 0,273 | 0,124 | 0,125 | 0,186 | 0,112 | 0,2112 | 85,30 % |
| **F1** | 0,191 | 0,212 | 0,061 | 0,042 | 0,072 | 0,047 | **0,1041** | **89,64 %** |

F1 contre Arki05 : −50,3 % [−53,3 ; −47,6] (297 documents sur 300) ; contre XQD : −50,7 % [−54,2 ; −47,4] (299 sur 300).
**Ce jeu avantage Formic** : il est tiré des mêmes jeux de données que notre calibration (documents différents). Le protocole public,
construit par un concurrent, est la comparaison la plus défendable.

## 5. Le paquet complet

| Contenu | Fichiers | Octets | Débit réel |
|---|---:|---:|---|
| Texte — 498 matrices E8 (couches, `embed_tokens`, `lm_head`) | 498 | 9 448 769 144 | 2,8107 bpw sur 26 893 M poids |
| Texte — petits tenseurs en FP16 | 353 | 5 300 601 | |
| Vision — 112 matrices E8 | 112 | 284 276 987 | 4,9405 bpw sur 460,3 M poids |
| Vision — petits tenseurs en FP16 | 221 | 824 181 | |
| MTP — 8 matrices E8 | 8 | 212 044 538 | 3,9945 bpw sur 424,7 M poids |
| MTP — petits tenseurs en FP16 | 7 | 52 399 | |
| Tokenizer, gabarit, configurations | 9 | 22 909 615 | |
| Manifeste (tailles, formes, SHA-256) | 1 | 307 184 | |
| **Total sur disque** | | **9 974 484 649** | marge sous 10 Go : 25,5 Mo |

* **Fichiers réels.** Le coût « idéal » utilisé jusqu'ici annonçait 2,8119 bpw pour le texte ; les fichiers font 2,8107 bpw.
* **Relecture.** Un processus séparé décode les 498 fichiers ; le modèle reconstruit redonne les mêmes KL à six décimales
  (600 documents : 0,101506 ; P-A108-dev : 0,058115). `package.py verify` : 1199 entrées, 27 778 347 008 poids codés, sommes de contrôle correctes.
* **Vision.** Sur 24 invites image + texte (64 tokens générés par le professeur) : KL 0,0626 avec la tour en BF16, **0,0636** avec la
  tour du paquet (accord top-1 93,7 % → 93,4 %). À 5 bpw, la tour ajoute donc ~0,001 ; l'essentiel du dommage vient du chemin texte.
* **MTP.** Quantifié depuis le checkpoint et compté, **jamais exécuté** : le runtime Transformers utilisé ne le charge pas.
* Le paquet (9,97 Go) n'est pas dans le dépôt Git ; seul son manifeste l'est (`results/package_manifest_F1.json`).

## 6. Coûts mesurés

| Poste | Valeur |
|---|---|
| Passe F1 (64 couches + tête, 409 600 tokens de calibration, écriture du paquet en parallèle) | ≈ 29 min sur A100 ; pic 76,9 Gio de VRAM |
| Codec | ≈ 110 M poids/s par cœur à l'encodage, ≈ 70 M poids/s au décodage |
| Décodage des 498 fichiers, chargement et réévaluation (600 documents + P-A108) | 11 min au total (12 processus de décodage) |
| Chargement d'un GGUF concurrent dans notre graphe | ≈ 7–8 min |
| Évaluation P-A108 (108 fenêtres × 4096 tokens) | ≈ 4 min ; 600 documents : ≈ 5 min |

## 7. Limites

* **Fidélité, pas capacités.** KL et accord top-1 ne sont pas des scores de benchmark. Aucune tâche n'a été exécutée, ni pour Formic ni pour les concurrents.
* **Pas le moteur des concurrents.** Leurs poids sont exécutés dans Transformers après déquantification, en BF16, et non dans llama.cpp.
  La mesure isole la qualité des poids quantifiés, pas un produit en fonctionnement.
* **108 fenêtres sur 128.** Le tableau publié par Arki05 (0,0823) porte sur 128 fenêtres ; nos chiffres ne lui sont pas directement comparables.
* **Deux concurrents seulement**, à leurs révisions du 13 novembre 2026 (`cdcbea9996`, `d7110adadd`). GSQ-RCO, Unsloth et BaseQuant n'ont pas été mesurés.
  XQD déclare viser les tokens d'assistant et non la prose : le protocole public, en partie fait de texte brut, ne lui est pas favorable.
* **Pas de moteur pour le format Formic.** Le paquet se décode vers des poids denses ; rien n'est dit de la mémoire ou de la vitesse d'exécution.
* **Poids évalués en BF16** après décodage ; le type d'exécution du produit reste à fixer.
* **Vision** : 24 invites ; tour quantifiée sans rétroaction. **MTP** : non évalué. Génération avec cache et contextes > 8192 tokens : non évalués.
* **Parenté des données.** `cal` contient MetaMathQA, dérivé de GSM8K et MATH dont viennent les invites « math » du protocole public
  (c'est le domaine où le gain est le plus faible). `conf2` partage la distribution de la calibration.
* **Une seule passe, une seule graine**, pas de mesure de la variance entre exécutions. La tête de sortie a été quantifiée après une
  reprise sur point de sauvegarde (saturation mémoire au premier essai) ; le paquet relu reproduit les mesures.
* **Sélection amendée** (section 3) : le départage entre F1 et L2 n'était pas prévu par la règle initiale.
* **Révision du checkpoint** : toujours non attestée par un fichier local (déclarée par l'opérateur).

## 8. Ce qui reste ouvert (non lancé, conformément à la consigne d'arrêt)

1. Cible de calibration asymétrique sur les 64 couches : signal mitigé sur 12 couches, à trancher sur une passe complète.
2. Tête de sortie à ±1 bit de pondération : gain sur un seul des deux jeux, sans rétroaction ; à mesurer avec LDLQ.
3. Composition et taille de la calibration : seul levier ayant payé ; ni la proportion des domaines ni le volume n'ont été explorés.
4. Moteur d'inférence, évaluation de capacités, MTP, Vision à plus grande échelle, comparaison dans llama.cpp.
