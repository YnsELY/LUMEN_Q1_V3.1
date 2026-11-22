# Étude de faisabilité — appliquer LUMEN (recette F1) à Inkling 2 Mini

**22 novembre 2026.** Étude documentaire et calculs légers, sans GPU. **Aucun poids d'Inkling 2 Mini n'a été téléchargé, quantifié
ni évalué.** Calculs : `calculs_inkling.py`, sortie intégrale dans `calculs_inkling.out.txt`.

> Niveaux de preuve utilisés : **publié** (par Thinking Machines, Hugging Face, Unsloth ou Artificial Analysis, non reproduit) ;
> **mesuré** (par Formic, sur Qwen3.8-27B, campagne V7) ; **calculé** (arithmétique sur des champs publiés) ; **estimé**
> (extrapolation sous hypothèse explicite) ; **non testé**.

---

## 0. Réponse en une page

**Peut-on reproduire la quantification LUMEN sur Inkling 2 Mini ?** Oui. Plusieurs briques se transposent sans modification : le
réseau E8, le pas FP16 par ligne, la compensation LDLQ par blocs de 8 colonnes, le codec, le traitement de la Vision et toute la
méthode (jeux disjoints, gel, confirmation scellée, concurrents exécutés dans notre graphe). Ces briques ne dépendent pas de
l'architecture : elles opèrent matrice par matrice.

Le modèle diffère pourtant de Qwen3.8-27B sur cinq points. Chacun impose une adaptation du pipeline (§ 4) :

1. **Il n'y a pas de professeur BF16.** Le checkpoint publié est le modèle, en NVFP4 et MXFP8 natifs.
2. **12 couches sur 24 s'exécutent trois fois** avec les mêmes poids. Leur erreur est injectée trois fois.
3. **Le modèle est un MoE de 128 experts dont 8 actifs.** Il faut des Gram par expert, une calibration 5 à 10 fois plus grande et
   un routeur protégé.
4. **L'architecture contient des éléments absents du pipeline** : résidus d'attention, fenêtres glissantes, puits, convolutions
   courtes, contrôleurs de boucle.
5. **Le runtime Formic R0 v3 ne sert pas tel quel.** Il est écrit pour le graphe Qwen3.5 (Gated DeltaNet).

**Jusqu'où descendre ?** Ces paliers sont **estimés**, pas mesurés. Les tailles comprennent Vision, drafter et fichiers annexes.

| Palier | Experts | **Paquet complet** | KL estimée* | Repère publié (Unsloth, texte seul) |
|---|---:|---:|---|---|
| Quasi exact : NVFP4 conservé bit pour bit | ~4,3 bpw | **~14,4 Go** | ≈ 0 sur 83 % des experts | UD-Q4_K_XL 17,6 Go : 0,0107 |
| Équilibre | 3,2 bpw | **~11,0 Go** | 0,022–0,030 | UD-IQ3_XXS 11,7 Go : 0,0269 |
| **Sous 10 Go (transposition de F1)** | 2,8 bpw | **~9,7 Go** | 0,032–0,043 | UD-Q2_K_XL 9,1 Go : 0,0595 |
| Plancher recommandé | 2,3 bpw | **~8,2 Go** | 0,051–0,068 | UD-IQ1_S 7,8 Go : 0,0911 |
| Déconseillé | 2,0 bpw | ~7,2 Go | 0,067–0,090 | |

\* KL estimée = KL d'Unsloth au même débit × g, avec g ∈ [0,56 ; 0,75] pour le gain de LUMEN à octets égaux. Ce g est transposé
de Qwen (§ 5.3), sur le protocole d'Unsloth, qui n'est pas documenté. Ces chiffres ne sont pas comparables aux KL de F1 sur Qwen.

**Recommandation :**
- **Cible principale : ~9,7 Go**, experts à 2,8 bpw, l'équivalent direct de F1.
- **Plancher : ~8 Go.** En dessous, la courbe publiée se dégrade vite.
- **Palier quasi exact : ~14,4 Go**, propre à ce modèle. Aucun concurrent ne l'occupe : les GGUF 4 bits requantifient un modèle
  déjà en 4 bits et plafonnent à 0,0107 de KL (publié).

À cette taille, on obtiendrait un modèle de même indice Artificial Analysis que Qwen3.8-27B (20,1, publié). Il ne lirait
qu'environ 1,1 à 1,8 Go de poids par token, contre 9,45 Go pour Qwen F1 (calculé).

---

## 1. Le modèle cible (publié)

`thinkingmachines/Inkling-2-Mini`, publié le 20 novembre 2026, licence Apache 2.0. Il accepte texte et images en entrée et produit
du texte. Contexte d'un million de tokens.

### 1.1 Architecture (`config.json`, champs verbatim)

| Élément | Valeur | Comparaison Qwen3.8-27B |
|---|---|---|
| Paramètres | 26,2 G (fiche HF) / 26,4 G (annonce) au total ; 3,4 G actifs ; 5,4 G « activations de poids » par token en comptant les passes | 27,8 G, dense |
| Couches | `num_hidden_layers` 24, `hidden_size` 2304 | 64 couches, largeur 5120 |
| Boucle | `loop_config` : `core_start` 6, `core_end` 18, `num_passes` 3, `gate_rank` 64, `is_varloop` false → couches 0–5 une fois, **6–17 trois fois**, 18–23 une fois : 48 applications par token | aucune |
| MoE | `n_routed_experts` 128, `num_experts_per_tok` 8, `n_shared_experts` 0, `intermediate_size` 1152, `route_scale` 8, `use_gate_bias` true, softcaps `mlp_gate` 4 et `mlp_up` 25 | MLP dense 17 408 |
| Attention | 24 têtes Q, 8 têtes KV, `head_dim` 128 ; 20 couches à fenêtre glissante de 1024 et 4 couches globales (5, 11, 17, 23) ; logits de position relatifs appris, puits, porte de sortie, convolutions courtes (`sconv`, noyau 4) | 48 GDN + 16 attention complète |
| Résidus d'attention | `n_layers_per_attn_res_block` 6 : chaque couche lit un mélange appris des sorties des blocs précédents | résiduel classique |
| Vocabulaire | 201 024 (200 058 utiles), `embed` et `unembed` séparés, `final_logit_softcapping` 30 | 248 320, séparés |
| Vision | ViT 23 blocs, largeur 1024, patch 20, fusion 2×2, projection vers 2304 | 27 blocs, largeur 1152 |
| Drafter | DSpark 3 couches (`mtp.safetensors`, 636 Mo), lit les états 5, 23, 41 et 48 | bloc MTP d'une couche |

Pourquoi boucler le milieu : selon l'annonce, boucler les 12 couches du milieu bat le fait de boucler les 12 dernières sur 27
évaluations sur 28. Un modèle non bouclé coûtait 7,5 % de calcul d'entraînement en moins, mais demandait 140 % de calcul
d'inférence en plus à perte égale (publié). Les passes 2 et 3 reçoivent un « message » de la passe précédente par des portes de
rang 64. Les passes s'ouvrent progressivement pendant l'entraînement.

Histoire des experts (publié) : l'entraînement a commencé avec 384 experts dont 8 actifs. Au milieu de l'entraînement, les deux
tiers ont été élagués pour garder les 128 plus utilisés par couche. Le modèle a ensuite été « soigné » par environ 1 T tokens de
distillation depuis un modèle Inkling plus grand.

### 1.2 Précision du checkpoint publié

| Tenseurs | Format | Source |
|---|---|---|
| Experts des couches 0–19 | **NVFP4** (E2M1, échelle FP8 E4M3 par bloc de 16, échelle globale FP32) | `hf_quant_config.json` (partie résumée par l'outil de lecture) |
| Experts des couches 20–23 | **MXFP8** (E4M3, bloc de 32) | idem |
| Attention, embeddings, unembed, normes, contrôleurs de boucle, Vision, drafter | BF16 | idem, partie verbatim |
| Biais des routeurs, requêtes des résidus d'attention, requêtes inter-passes | FP32 | idem |

L'annonce précise que la plupart des poids ont été entraînés et sont déployés nativement en NVFP4, les quatre dernières couches en
MXFP8, si bien qu'il n'y a « aucune perte de conversion ». **Aucun checkpoint BF16 n'est publié.** Le fichier principal pèse
19,5 Go ; le dépôt fait 20,2 Go.

La reconstitution de la taille (calculé, § A.1) donne 19,08 Go. L'écart avec 19,5 Go correspond à environ 0,21 G poids BF16 non
identifiés (`wr_du`, portes, convolutions…). Le total retrouvé est de 26,38 G, cohérent avec les 26,4 G de l'annonce.

### 1.3 Capacités (publié par le laboratoire ; aucune mesure de notre part)

- **Ensemble.** Indice Artificial Analysis v5.0 : **20,1, identique à Qwen3.8-27B**.
- **Où Inkling 2 Mini est devant.** DeepSWE (48,2 % contre 42,2 %), SWE-bench Pro Hard (77,9 % contre 73,8 %), BrowseComp
  (79,0 % contre 61,6 %), LongBench v2 (66,9 % contre 52,3 %), MCP Atlas (82,7 % contre 77,3 %).
- **Où il est derrière.** GDPVal-AA (1319 contre 1427), AutomationBench (41,8 % contre 48,2 %), ARC-AGI v1 (84,5 % contre
  87,5 %). Surtout le **rappel factuel** : non-hallucination AA-Omniscience 63,4 % contre 71,4 %. Le laboratoire signale cette
  faiblesse.
- **Coût d'exécution.** Artificial Analysis mesure environ 6 fois moins de coût et 4 fois moins de temps par tâche que
  Qwen3.8-27B, avec 38 % de tokens de sortie en plus.

Conséquence pour la quantification : sur Qwen, les domaines les plus dégradés par F1 étaient l'encyclopédique (0,0764) et le
multilingue (0,1293) (mesuré, moitié scellée). C'est précisément là qu'Inkling est déjà le plus faible.

### 1.4 Quantifications concurrentes publiées (Unsloth)

KL moyenne par rapport à UD-Q8_K_XL. Cette référence conserve les experts NVFP4 d'origine et passe le MXFP8 en BF16 : elle vaut
donc le modèle officiel aux arrondis de llama.cpp près. Le protocole (données, contexte, positions) **n'est pas documenté** dans
les pages lues.

| GGUF | Taille | bpw texte (calculé) | KL | Même top-1 |
|---|---:|---:|---:|---:|
| UD-IQ1_S | 7,76 Go | 2,38 | 0,0911 | 89,4 % |
| UD-IQ1_M | 8,02 Go | 2,46 | 0,0867 | 90,2 % |
| UD-IQ2_XXS | 8,49 Go | 2,61 | 0,0726 | 90,6 % |
| UD-Q2_K_XL | 9,14 Go | 2,81 | 0,0595 | 91,6 % |
| UD-IQ3_XXS | 11,66 Go | 3,58 | 0,0269 | 94,7 % |
| UD-Q3_K_XL | 13,63 Go | 4,18 | 0,0174 | 96,3 % |
| UD-IQ4_XS | 15,73 Go | 4,83 | 0,0109 | 97,7 % |
| UD-Q4_K_XL | 17,63 Go | 5,41 | 0,0107 | 98,0 % |

Ces tailles sont hors Vision (projecteur `mmproj` de 0,64 Go) et hors drafter (0,40 Go). Le laboratoire publie lui-même des GGUF
`IQ4_NL` et `Q8_0` (`ggml-org`). Aucun chiffre de KL n'est publié pour eux.

**Plateau à 4 bits.** De 15,7 à 17,6 Go, la KL reste à environ 0,011. L'explication la plus probable (non vérifiée) : les formats
GGUF à 4 bits ne représentent pas exactement la grille NVFP4 (blocs de 16 à échelle FP8). Ils requantifient donc un modèle déjà
quantifié, et la perte de conversion domine.

---

## 2. Rappel : ce que F1 a fait sur Qwen3.8-27B (mesuré)

Recette complète : `04_projet_F1_et_campagnes/formic_v7/RECETTE_LUMEN_V7_EXP.md` ; résultats : `RAPPORT_V7.md`.

- **Quantifieur.** 498 matrices du chemin texte en E8 à codes doublés. Un pas FP16 par ligne, égal à un multiplicateur global
  unique (0,598541) × RMS de la ligne : l'allocation est **uniforme**.
- **Compensation.** LDLQ par blocs de 8 colonnes, 7 candidats par bloc. Objectif local symétrique sur les entrées de l'élève,
  couche par couche. `embed_tokens` est quantifié au plus proche ; `lm_head` passe par LDLQ.
- **Calibration.** 409 600 tokens, `cal` + `cal2` : conversations, outils, raisonnement rendus avec le gabarit. C'est le seul
  levier qui a payé : −11,6 % de KL.
- **Paquet.** 9 974 484 649 octets : texte à 2,8107 bpw, Vision à 4,94 bpw, MTP à 3,99 bpw.
- **Résultat.** Sur la moitié scellée du protocole public d'Arki05 : KL **0,0613** contre 0,0915 pour Arki05 2,82 bpw (−33,1 %,
  IC [−36,6 ; −28,7]) et 0,1292 pour XQD Q2. Top-1 : 90,50 %.
- **Échecs utiles.** Une allocation non uniforme transplantée d'un autre quantifieur a dégradé la KL (+8 % et +18 %). Le modèle
  Fisher × pivots n'était pas prédictif. Seule une allocation **mesurée avec E8-LDLQ** a une valeur.

---

## 3. Transposabilité brique par brique

| Brique F1 | Reprise sur Inkling | Adaptation nécessaire | Risque |
|---|---|---|---|
| E8 doublé, pas par ligne, multiplicateur global | telle quelle | aucune ; un expert = 2304 lignes de 2304 (`w13`) ou de 1152 (`w2`), coût du pas 0,007–0,014 bpw | faible |
| LDLQ par blocs de 8 | algorithme identique | Gram **par expert** ; Gram des couches bouclées = somme des 3 passes | moyen (§ 4.2, 4.3) |
| Passe séquentielle couche par couche (`seq.py`) | non | ordre circulaire dans le cœur bouclé, routage, résidus d'attention | **élevé** |
| Calibration `cal` + `cal2` | contenus réutilisables | nouveau tokenizer (201 024), nouveau gabarit (effort de raisonnement, outils), volume × 5 à 10 | moyen |
| `embed` au plus proche, tête par LDLQ | oui | la tête produit des logits passés par un softcap de 30 ; le Gram vient des états finaux de l'élève | faible |
| Codec par intervalles | oui | environ 6 150 matrices au lieu de 498 : regrouper par couche pour limiter les en-têtes | faible |
| Vision E8 à 5 bpw sans rétroaction | oui | tour plus petite (~0,32 G contre 0,46 G) | faible |
| Bloc MTP à 4 bpw | à remplacer | drafter DSpark ; il ne sert qu'à proposer, la vérification reste exacte | faible |
| Petits tenseurs en FP16 | oui | **et en plus** : routeurs, contrôleurs de boucle, requêtes inter-passes, requêtes des résidus d'attention, puits, logits relatifs. Garder FP32 là où le checkpoint est en FP32 | faible |
| Méthode (disjonction, gel, confirmation scellée) | oui | construire un protocole public pour Inkling, qui n'en a pas | moyen |
| Harness Transformers 5.22.2 | non | il faut une version qui contient `inkling2_loop` (`Inkling2LoopForConditionalGeneration`) | à vérifier |
| Runtime Formic R0 v3 | non | nouveau graphe (§ 4.5) | **élevé** (ingénierie) |

---

## 4. Les cinq difficultés propres à Inkling

### 4.1 Le professeur est déjà quantifié

Il n'existe pas de BF16. Le professeur est donc le checkpoint publié, déquantifié en FP32. Ce choix est légitime : c'est le
modèle officiel, et Unsloth fait de même. Trois vérifications sont à faire avant toute passe :

1. **Activations quantifiées ?** `quant_algo: NVFP4` chez ModelOpt peut désigner W4A4 (activations en FP4). Si le checkpoint
   contient des `input_scale`, l'inférence de référence quantifie aussi les activations. Il faut alors fixer la définition du
   professeur. Proposition : W4A16, poids exacts et activations en BF16/FP32, comme llama.cpp, MLX et Unsloth. Il faut aussi
   mesurer l'écart avec W4A4 sur le protocole.
2. **Exactitude de la déquantification.** Une valeur NVFP4 multipliée par l'échelle FP8 puis par l'échelle globale FP32 n'est
   pas toujours représentable en BF16. Il faut garder le professeur et les cibles LDLQ en FP32, ou vérifier l'arrondi tenseur par
   tenseur (règle du projet : vérifier toute transformation exacte dans le vrai type).
3. **Requantifier un 4 bits.** Sous 4,3 bpw, E8 remplace la grille NVFP4. Le signal à coder est la grille elle-même : cela ne
   pose aucun problème théorique, et LDLQ minimise l'erreur de sortie par rapport à ces poids. Pour fixer les ordres de grandeur
   (calculé, signal gaussien) :
   - l'erreur de la grille NVFP4 est à environ −20,5 dB du signal ;
   - E8 entropique à 2,8 bpw serait vers −16 dB (6 dB par bit, moins environ 1 dB de perte granulaire). C'est un bruit plus fort
     que celui auquel l'entraînement a habitué le modèle, et ajouté à lui ;
   - le bruit de la grille est atteint vers 3,5–3,6 bpw.

   Hypothèse, **non testée** : un modèle entraîné sous bruit FP4 pourrait mieux tolérer une perturbation supplémentaire.

**Option propre à ce modèle : conserver le NVFP4 bit pour bit.** Le codage entropique des codes E2M1 donne environ 3,8–3,9
bit/poids. Avec les échelles, on obtient environ **4,30 bpw au lieu de 4,50** (calculé par simulation sur des lois gaussienne,
de Laplace et de Student, § A.2 ; à mesurer sur les vrais poids). Le gain sans perte est donc modeste (−4 %). Mais il donne une KL
nulle sur 83 % des experts, d'où le palier quasi exact à ~14,4 Go. Le codec actuel ne gère pas ce format : il faut un modèle de
code à 15 symboles avec contexte d'échelle.

### 4.2 Le cœur bouclé : 50 % des poids utilisés trois fois

- **Le Gram.** Les entrées d'une matrice du cœur viennent des trois passes. L'objectif local naturel est
  H = Σ_p X_pᵀ X_p, ce qui triple les tokens effectifs du Gram.
- **L'ordre est circulaire.** L'entrée de la couche 6 à la passe 2 dépend des couches 6 à 17 déjà quantifiées à la passe 1.
  Proposition :
  - premier balayage : Gram de la passe 1 sur l'élève, Gram des passes 2 et 3 sur l'état du professeur ;
  - second balayage : recalcul de tous les Gram du cœur sur l'élève entièrement quantifié, puis nouvelle passe LDLQ des 12
    couches.

  Coût : environ ×2 sur le cœur. Il faut mesurer si le second balayage paie.
- **L'amplification.** Une erreur sur une matrice du cœur touche trois positions du calcul, et les portes inter-passes la
  propagent d'une passe à l'autre. L'hypothèse naturelle est une **allocation non uniforme** : plus de bits au cœur, moins au
  prélude et à la coda, à octets égaux. Sur Qwen, l'uniforme avait gagné, mais les allocations essayées n'avaient pas de raison
  structurelle. Ici, il y en a une. **À mesurer avec E8-LDLQ**, par exemple cœur +0,2 / prélude-coda −0,2 bpw puis ±0,4, jamais
  à décréter.
- **Attention.** Le cœur contient la couche globale 11 et la couche 17 ; les 20 autres couches ont une fenêtre de 1024. Une
  erreur dans une couche globale touche tout le contexte. L'attention ne pèse que 0,45–0,66 G poids (≈ 2,5 %) : la garder à
  4,5–5 bpw coûte moins de 0,1 Go supplémentaire.

### 4.3 MoE : 3 072 experts, chacun ne voit qu'un seizième des tokens

- **Données par expert** (calculé, § A.6) :

  | Tokens de calibration | Tokens par expert en moyenne | Cœur bouclé (×3) | Rapport tokens / dimension 2304 |
  |---:|---:|---:|---:|
  | 409 600 (comme F1) | 25 600 | 76 800 | 11 |
  | 2 000 000 | 125 000 | 375 000 | 54 |
  | 4 000 000 | 250 000 | 750 000 | 108 |

  Ces chiffres sont des moyennes. Les experts rares en reçoivent beaucoup moins, même après l'élagage, qui a gardé les plus
  utilisés. Proposition : **2 à 4 M tokens**, avec un seuil minimal par expert. Sous ce seuil, un expert passe en quantification
  au plus proche ou reçoit un amortissement plus fort. Ce seuil est à déclarer avant les passes.
- **Routage.** Une erreur de quantification en amont change les experts choisis : c'est un mode d'erreur qui n'existait pas sur
  Qwen. Les routeurs restent en BF16, leurs biais en FP32 (7 M poids, 0,014 Go). Il faut mesurer l'**accord de routage** : la
  part des 8 experts identiques entre l'élève et le professeur, par couche et par passe.
- **Mémoire de la passe** (calculé) :
  - Gram FP32 `gate`/`up` : 128 × 2304² × 4 o = 2,7 Go par couche ; `down` : 0,68 Go. C'est acceptable sur A100 80 Go ;
  - le modèle déquantifié occupe environ 52 Go en BF16, comme Qwen.
- **Volume de LDLQ** (estimé, lignes × colonnes²) : environ 5·10¹³ pour les experts, contre environ 2·10¹⁴ pour Qwen. C'est
  moins, même avec le second balayage du cœur. La durée devrait être du même ordre que la passe F1 (≈ 30 min sur A100), hors
  construction de la calibration.

### 4.4 Éléments d'architecture absents du pipeline

Résidus d'attention, logits relatifs appris (`d_rel` 16, `rel_extent` 1024), puits, porte d'attention, convolutions courtes sur q,
k, v et la sortie, normes d'entrée des couches globales, softcaps du MLP, contrôleurs de boucle (`down/forget/gain_weight`) et
requêtes inter-passes.

Tous sont **petits**, moins de 0,3 G au total, et restent en FP16/BF16/FP32. Le risque n'est pas la taille, c'est la capture des
entrées de l'élève : le harness doit exécuter le vrai graphe Transformers et intercepter les entrées des linéaires d'experts
après routage. Les résidus d'attention multiplient les chemins par lesquels une erreur atteint une couche. L'objectif LDLQ
« entrée de l'élève » reste valide, car il ne fait aucune hypothèse sur le résiduel.

### 4.5 Le runtime

Le runtime R0 v3 est écrit pour le graphe Qwen3.5 (GDN, attention complète, MTP). Pour Inkling, il faut :

- un MoE avec sélection de 8 experts : la GEMV fusionnée « décodage + produit » par ligne s'y prête ;
- la boucle à 3 passes et ses portes ;
- les résidus d'attention ;
- la fenêtre glissante, les puits et les logits relatifs ;
- le drafter DSpark.

C'est un projet à part entière, à séparer de la campagne de quantification. Comme en V7, on évaluerait d'abord le paquet décodé en
poids denses. L'enjeu est important : **environ 1,1 à 1,8 Go lus par token à 2,8 bpw** (calculé ; borne haute si chaque passe
relit ses experts), contre 9,45 Go pour Qwen F1.

---

## 5. Taille : inventaire, modèle d'estimation, paliers

### 5.1 Inventaire (calculé depuis `config.json`, § A.1)

| Bloc | Poids |
|---|---:|
| Experts, 24 couches | **24,46 G** (93,9 % du texte) |
| — prélude 0–5 / cœur 6–17 / coda 18–23 | 6,12 / 12,23 / 6,12 G |
| — NVFP4 (0–19) / MXFP8 (20–23) | 20,38 / 4,08 G |
| Attention identifiée (wq, wk, wv, wo) + tenseurs non identifiés | 0,45 + ~0,21 G |
| Routeurs | 0,007 G |
| Embed + unembed | 0,93 G |
| **Chemin texte** | **26,06 G** |
| Vision (estimée) | ~0,32 G |
| Drafter DSpark | ~0,32 G |

### 5.2 Courbe concurrente

Sur 2,4–4,2 bpw, les points d'Unsloth suivent ln KL = −0,156 − 0,944 × bpw : la KL est multipliée par **0,39 par bit**
supplémentaire (calculé). Au-delà de 4,2 bpw, le plateau à 0,011 décrit au § 1.4 casse cette loi.

### 5.3 Hypothèse de transposition du gain LUMEN (estimé)

- **F1 contre Arki05**, même taille, moitié scellée : 0,0613 / 0,0915 = **0,670** (mesuré).
- **Arki05 contre Unsloth UD-Q2_K_XL** sur Qwen, même taille et même protocole : 0,0823 / 0,0983 = 0,837 (publié par Arki05).
- **Enchaînement** : g ≈ 0,56 contre Unsloth. C'est fragile : deux protocoles, et un chiffre publié non reproduit.
- **Borne prudente** : g = 0,75. Elle couvre le cas où les GGUF « UD » d'Unsloth, déjà adaptés aux MoE, sont plus difficiles
  à battre que sur Qwen, et le cas où les difficultés des § 4.2 et 4.3 coûtent.

À KL égale, g ∈ [0,56 ; 0,75] équivaut à **0,30 à 0,61 bpw économisés** (calculé à partir de la pente).

### 5.4 Paliers (estimé, § A.5)

Hypothèses de répartition :
- experts au débit indiqué ;
- attention et tenseurs annexes à 4,5 bpw ;
- vocabulaire au débit des experts + 0,5 bpw ;
- Vision à 5 bpw, drafter à 4 bpw, annexes 0,06 Go, soit 0,42 Go en tout.

| Experts | Texte | **Paquet complet** | KL estimée LUMEN | Top-1 estimé** | Unsloth au même débit (ajustement) |
|---:|---:|---:|---|---|---:|
| 2,0 bpw | 6,78 Go | 7,20 Go | 0,067–0,090 | 89,6–91,0 % | 0,120 |
| 2,3 bpw | 7,73 Go | **8,15 Go** | 0,051–0,068 | 90,9–92,2 % | 0,091 |
| 2,5 bpw | 8,37 Go | 8,79 Go | 0,042–0,057 | 91,8–92,9 % | 0,076 |
| **2,8 bpw** | 9,32 Go | **9,74 Go** | **0,032–0,043** | 92,9–94,0 % | 0,057 |
| 3,0 bpw | 9,96 Go | 10,37 Go | 0,027–0,036 | 93,6–94,7 % | 0,048 |
| 3,2 bpw | 10,59 Go | 11,01 Go | 0,022–0,030 | 94,3–95,4 % | 0,040 |
| 3,6 bpw | 11,86 Go | 12,28 Go | 0,015–0,021 | 95,7–96,7 % | 0,027 |
| NVFP4 conservé + MXFP8 → E8 à 4 bpw + reste à 5 bpw | | **14,41 Go** | ≈ 0 sur les couches 0–19 | | |
| idem, attention et vocabulaire en BF16 | | 16,60 Go | | | |

\** Top-1 interpolé sur la relation KL ↔ top-1 des points d'Unsloth : indicatif seulement.

**Lecture :**
- **À ~9,7 Go**, le paquet complet tient sous 10 Go avec 0,26 Go de marge. La KL estimée serait proche de celle d'UD-IQ3_XXS
  (11,7 Go, texte seul).
- **À ~8,2 Go**, on retrouverait à peu près la qualité d'UD-Q2_K_XL (9,1 Go, texte seul).
- **Sous ~2,3 bpw**, chaque 0,1 bpw retiré coûte environ 10 % de KL. Pour un modèle dont le point faible est déjà le rappel
  factuel, je ne recommande pas de descendre plus bas.

Ces estimations ne sont **pas comparables** aux KL de F1 sur Qwen (0,0613) : modèle, protocole et référence diffèrent. Elles ne
disent rien des capacités en tâches.

---

## 6. Plan de campagne proposé (non lancé)

> Mandat d'exécution détaillé pour l'agent du pod, budget de 8 à 9 Go : `../../PROMPT_AGENT_QUANTIFICATION_LUMEN_INKLING2_MINI.md`
> (rédigé le 22 novembre 2026). Il précise et remplace le tableau ci-dessous, notamment le protocole d'évaluation (§ 5) : le
> corpus de la KL publiée par Unsloth n'est pas public.

À ouvrir par un mandat et des critères numériques enregistrés **avant** les passes, comme pour V7 et R0 v3.

| Étape | Contenu | Critère de sortie |
|---|---|---|
| **C0 — Identité et inventaire** | Télécharger le dépôt (20,2 Go). Comparer les SHA-256 à l'arbre HF. Lire les en-têtes safetensors pour obtenir l'inventaire exact (il remplace le § 5.1). Chercher les `input_scale` (W4A4 ?). Identifier la version Transformers qui contient `inkling2_loop`. | inventaire exact ; définition du professeur écrite |
| **C1 — Professeur** | Déquantifier en FP32 et vérifier l'exactitude. Comparer les logits de notre graphe à ceux de llama.cpp UD-Q8_K_XL sur quelques fenêtres. | écart attribué aux arrondis, documenté |
| **C2 — Protocole** | Construire un protocole public : fenêtres de 4096, domaines dont rappel factuel, multilingue, outils, raisonnement avec le gabarit d'Inkling ; moitié scellée. Exécuter UD-Q2_K_XL, UD-IQ2_XXS et UD-IQ3_XXS dans notre graphe (chargeur GGUF au format bouclé d'Unsloth, échelles NVFP4 par expert). | concurrents mesurés sur notre protocole |
| **C3 — Calibration** | `cal` + `cal2` rendus au format Inkling, 2 à 4 M tokens. Statistiques de routage et couverture minimale par expert. | seuil par expert déclaré |
| **C4 — Passe L0** | F1 transposée sans autre changement : uniforme à 2,8 bpw, LDLQ par expert, Gram du cœur sommé sur les 3 passes, un puis deux balayages. Petits tenseurs et routeurs intacts. | KL et accord de routage sur la moitié dev |
| **C5 — Variantes mesurées** | Cœur ± Δ à octets égaux ; attention à 4,5 bpw ; tête ; paliers 2,3 et 3,2 bpw ; palier quasi exact avec un codec NVFP4 sans perte. | règle de sélection déclarée avant les résultats |
| **C6 — Confirmation et paquet** | Lecture de la moitié scellée. Paquet complet relu depuis ses fichiers. Vision sur plus de 24 invites. Drafter exécuté. | comme V7 |
| *Hors campagne* | Graphe Inkling dans le runtime Formic. | projet séparé |

Ressources : une A100 80 Go suffit (≈ 52 Go de modèle en BF16, Gram par couche < 4 Go). Le disque doit accueillir le checkpoint
(20 Go), la copie déquantifiée (~52 Go) et les GGUF concurrents (~30 Go).

---

## 7. Ce qui n'est pas établi

- **Aucune mesure sur Inkling.** Toutes les KL et tailles de ce dossier sont des estimations ou des calculs sur des champs publiés.
- **Le gain de LUMEN ne se transpose pas par principe.** g a été mesuré sur un modèle dense, contre Arki05, sur un autre
  protocole. Le MoE et la boucle peuvent l'éroder ou l'accroître.
- **La courbe d'Unsloth suit un protocole non documenté.** Sa référence est UD-Q8_K_XL, pas le checkpoint exécuté par
  Transformers.
- **Plusieurs lectures sont des résumés d'outil.** La partie de `hf_quant_config.json` qui répartit NVFP4 et MXFP8 par couche n'a
  pas été lue verbatim ; l'inventaire exact viendra de C0.
- **La Vision (~0,32 G) est estimée** à partir de la config, sans les tenseurs.
- **L'entropie NVFP4 de 4,30 bpw vient d'une simulation** sur des lois synthétiques. Les vrais poids, issus d'un entraînement
  natif en FP4, peuvent différer.
- **KL et top-1 ne sont pas des pourcentages de capacités.** Une campagne de tâches resterait nécessaire, en particulier sur le
  rappel factuel et l'agentique.
- **Écarts entre sources.** 26,2 G (fiche HF) contre 26,4 G (annonce) ; 3 G actifs (blog HF) contre 3,4 G (annonce).

---

## 8. Sources

| Réf. | Source | Contenu utilisé |
|---|---|---|
| S1 | https://huggingface.co/thinkingmachines/Inkling-2-Mini (fiche, `README.md`, `config.json`, `hf_quant_config.json`, arbre des fichiers) | architecture, précisions, tailles, benchmarks |
| S2 | https://thinkingmachines.ai/news/inkling-2-mini/ | boucle, élagage des experts, entraînement NVFP4/MXFP8 natif, ~20 Go |
| S3 | https://huggingface.co/blog/inkling2mini | mécanique de la boucle, attention, DSpark, GGUF IQ4_NL/Q8_0 |
| S4 | https://unsloth.ai/docs/models/inkling-2-mini et https://huggingface.co/unsloth/Inkling-2-Mini-GGUF | tailles et KL des GGUF, référence UD-Q8_K_XL |
| S5 | https://artificialanalysis.ai/articles/inkling-2-mini | indice 20,1, coût et temps par tâche |
| S6 | https://venturebeat.com/technology/thinking-machines-wows-with-inkling-2-mini-looped-transformer-model-that-runs-locally-on-laptops | contexte de sortie |
| P1 | `04_projet_F1_et_campagnes/formic_v7/RECETTE_LUMEN_V7_EXP.md`, `RAPPORT_V7.md` | recette et mesures F1 |
| P2 | `04_projet_F1_et_campagnes/DOSSIER_CONCURRENTS_KL_ET_LEVIERS_LUMEN_V2.md` | KL publiées par Arki05 (dont Unsloth UD-Q2_K_XL sur Qwen) |

Pages lues le 22 novembre 2026.

---

## Annexe A — Calculs

Script : `calculs_inkling.py` (numpy seul, graine 0). Sortie : `calculs_inkling.out.txt`. Sections :

- **A.1** inventaire des poids et reconstitution de la taille du checkpoint ;
- **A.2** entropie des codes NVFP4 par simulation (blocs de 16, échelle absmax/6 arrondie en E4M3) ;
- **A.3** courbe d'Unsloth convertie en bpw texte et ajustement log-linéaire ;
- **A.4** facteur g ;
- **A.5** paliers du paquet complet ;
- **A.6** tokens de calibration par expert et octets lus par token.

Lancer avec l'interpréteur qui a numpy, sortie en UTF-8 sous Windows :

```powershell
$env:PYTHONIOENCODING = 'utf-8'; & 'C:\Users\yanis\AppData\Local\Programs\Python\Python313\python.exe' -B calculs_inkling.py
```
