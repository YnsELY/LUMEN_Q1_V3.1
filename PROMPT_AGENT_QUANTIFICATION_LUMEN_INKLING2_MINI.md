# FORMIC-Ink — Quantifier Inkling 2 Mini avec LUMEN (recette F1 transposée), paquet complet de 8 à 9 Go

**Mandat d'exécution — 22 novembre 2026.** Destinataire : l'agent de codage qui travaille sur le pod GPU.
Étude préalable (à lire en premier) : `DOSSIER_REFLEXION_V8/06_etude_inkling2_mini/ETUDE_FAISABILITE_LUMEN_INKLING2_MINI.md`.
Recette de référence (Qwen) : `formic_v7/RECETTE_LUMEN_V7_EXP.md`, résultats : `formic_v7/RAPPORT_V7.md`.

> Vocabulaire obligatoire dans tout ce que tu écris : **mesuré** (par toi, fichiers bruts dans `results/`), **publié** (par un
> tiers, non reproduit), **estimé**, **non testé**. KL et accord top-1 ne sont **pas** des pourcentages de capacités.
> Documentation en français.

---

## 0. Mission en une page

**But.** Produire la version quantifiée LUMEN la plus fidèle possible de `thinkingmachines/Inkling-2-Mini`, sous forme d'un
**paquet complet réel de 8 000 000 000 à 9 000 000 000 octets**. Le paquet comprend le texte, les embeddings, la tête, la Vision,
le drafter DSpark, les petits tenseurs, le tokenizer et les configurations. Il doit être relu depuis ses fichiers et évalué.

**Méthode.** C'est la recette F1 de Qwen, appliquée sans changement de principe :
- E8 à codes doublés, un pas FP16 par ligne égal à un multiplicateur global × RMS de la ligne ;
- LDLQ par blocs de 8 colonnes avec 7 candidats, objectif local symétrique sur les entrées de l'élève, couche par couche ;
- calibration conversations + outils + raisonnement, aucun apprentissage, codec rapide.

Les seules différences autorisées sont celles qu'impose l'architecture : MoE, couches bouclées, professeur NVFP4 (§ 4). Toute
autre déviation est une **variante** déclarée et mesurée (§ 7), jamais un changement silencieux.

**Comparaison.** Elle se fait contre les GGUF d'Unsloth à **octets texte égaux**, mesurés **par toi**, sur **ton** protocole,
dans **ton** graphe et aussi dans llama.cpp (§ 5). Le corpus de la KL publiée par Unsloth n'est pas public (§ 5.1) : leurs
chiffres publiés servent de repère, jamais de point de comparaison direct.

**Budget.** Les octets du chemin texte de `UD-IQ2_XXS` (fichier de 8,49 Go, publié). Paquet complet estimé : ~8,9 Go, donc
sous 9 Go. Les concurrents sont limités à trois : `UD-IQ2_XXS`, `UD-Q2_K_XL` et `UD-IQ3_XXS` (§ 5.3).

**Livrables.**
1. `formic_inkling/RAPPORT_INKLING_F1.md` : rapport de campagne, au format de `RAPPORT_V7.md`.
2. `formic_inkling/RECETTE_LUMEN_INKLING.md` : recette retenue, au format de `RECETTE_LUMEN_V7_EXP.md`.
3. `formic_inkling/results/` : tous les JSON bruts, les verrous de sélection et le manifeste du paquet.
4. Le paquet sur le disque du pod (`/workspace/work_ink/packages/<VID>`) et son manifeste archivé dans `results/`.
5. Le code dans `formic_inkling/{lab,jobs,data,codec}` : versions dérivées et identifiées des fichiers Qwen.

**Interdits.**
- Toute forme d'apprentissage : codes, échelles, distillation, LoRA.
- Lire la moitié de confirmation avant le gel.
- Modifier les fichiers Qwen d'origine (`formic_arch/`, `formic_v7/`, `results/`, `lumen_ref/`). Tu copies, puis tu dérives.
- Changer de checkpoint sans le dire.
- Pousser quoi que ce soit hors du pod sans instruction de l'opérateur.

---

## 1. Ce que tu dois savoir sur le modèle (publié, à vérifier en C0)

| Élément | Valeur publiée |
|---|---|
| Dépôt | `thinkingmachines/Inkling-2-Mini`, publié le 20 novembre 2026, Apache 2.0. `model-00001-of-00001.safetensors` (19,5 Go), `mtp.safetensors` (636 Mo, drafter DSpark) |
| Classe | `Inkling2LoopForConditionalGeneration`, `model_type: inkling2_loop` ; chargement documenté par `AutoModelForMultimodalLM` |
| Texte | 24 couches, `hidden_size` 2304, vocabulaire 201 024 (200 058 utiles), `embed` et `unembed` séparés, `final_logit_softcapping` 30 |
| Boucle | `loop_config` : `core_start` 6, `core_end` 18, `num_passes` 3, `gate_rank` 64. Couches 0–5 une fois, **6–17 trois fois**, 18–23 une fois, soit 48 applications par token. Contrôleurs `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight` ; `middle_cycle_cross_pass_{queries,kv_queries}` |
| MoE | 128 experts routés, 8 actifs, 0 partagé, `intermediate_size` 1152, `route_scale` 8, biais de routeur, softcaps `mlp_gate` 4 et `mlp_up` 25. Tenseurs fusionnés `mlp.experts.w13_weight` (gate+up) et `w2_weight` |
| Attention | 24 têtes Q, 8 têtes KV, `head_dim` 128. Fenêtre glissante de 1024 sur 20 couches ; couches globales 5, 11, 17, 23. Logits relatifs appris, puits, porte, convolutions courtes `sconv`. Résidus d'attention (`attn_res`, blocs de 6) |
| Précision | Experts NVFP4 (couches 0–19, blocs de 16, échelle E4M3) et MXFP8 (couches 20–23, blocs de 32). Attention, embeddings, unembed, normes, contrôleurs, Vision et drafter en BF16. Biais de routeur et requêtes `attn_res` et inter-passes en FP32. **Aucun BF16 publié** |
| Taille estimée (étude) | Experts 24,46 G poids (93,9 % du texte) ; attention et annexes ~0,66 G ; vocabulaire 0,93 G ; Vision ~0,32 G ; drafter ~0,32 G |

Il existe aussi un modèle de recherche distinct, `thinkingmachines/TML-d24-varloop-research` : **ne pas l'utiliser**.

---

## 2. Environnement

- **GPU** : A100 ou H100 80 Go (le modèle décodé pèse ~52 Go en BF16 : **une carte de 24 ou 48 Go ne convient pas** sans refonte
  couche par couche). Un GPU Blackwell n'est pas nécessaire (nous déquantifions). **Disque** : ≥ 200 Go libres (150 Go suffisent si tu omets le volet llama.cpp et l'export GGUF BF16)
  (checkpoint 20 Go, copies déquantifiées, GGUF concurrents et référence ~52 Go, paquets, cache des professeurs).
- **Répertoires** : modèle `/root/models/inkling-2-mini` ; travail `/workspace/work_ink/` ; code `/workspace/LUMEN_Q1_V3.1/formic_inkling/`.
  **Ne réutilise pas `/workspace/work/`**, qui est celui de Qwen, pour ne rien écraser.
- **Transformers** : une version qui contient `inkling2_loop`. Note la version exacte dans `results/env.json`. Si aucune version
  publiée ne la contient, n'utilise pas `trust_remote_code` sans vérifier le code téléchargé : arrête-toi et signale-le.
- **llama.cpp** : le fork indiqué par Unsloth (release `b12588-mix-c0c3de8`, disposition « looped-layer » et échelles NVFP4 par
  expert) **et** llama.cpp principal avec les fichiers `Mainline_Support/`. Compile `llama-perplexity` et prends le `gguf-py` du
  fork pour la déquantification.
- **Règles du projet** :
  - `python -B` partout ;
  - jobs numérotés soumis au serveur de laboratoire, qui garde le modèle en VRAM ;
  - résultats scellés jamais imprimés ;
  - les scripts de calibration et de protocole écrivent dans `/workspace/work_ink/data/`, jamais dans `results/` de la release.
- **Serveur de laboratoire** : copie `formic_arch/lab/lab_server.py` en `formic_inkling/lab/lab_server_ink.py` avec
  `MODEL_DIR=/root/models/inkling-2-mini`, la file `/workspace/work_ink/queue` et la classe Inkling. Le chargement passe par
  ton chargeur de professeur (§ 4.1).

---

## 3. Plan de campagne

| Étape | Contenu | Sortie obligatoire | Arrêt si |
|---|---|---|---|
| **C0** | Identité, inventaire, définition du professeur | `results/checkpoint_headers_inkling.json`, `results/env.json`, `results/teacher_definition.md` | le checkpoint ne correspond pas à la révision notée ; présence d'activations quantifiées non résolue (§ 4.1) |
| **C1** | Professeur exact et harness | `results/teacher_check.json` | le graphe ne reproduit pas UD-Q8_K_XL (§ 4.1) |
| **C2** | Protocole d'évaluation et mesure des concurrents | `results/protocol_ink.json`, `results/competitor_*.json` | — |
| **C3** | Calibration | `results/calibration_ink.json` | couverture des experts sous le seuil déclaré pour plus de 5 % des experts |
| **C4** | Passe L0 : F1 transposée, budget principal | `results/variant_L0.json`, paquet `L0` | — |
| **C5** | Variantes déclarées (§ 7) | `results/variant_*.json` | — |
| **C6** | Gel, confirmation, paquet complet, rapport | `results/selection.lock.json`, `results/final_selected.json`, rapport, recette | — |

Durées estimées (non mesurées, carte de 80 Go) : C0–C1 une demi-journée ; C2 une demi-journée (réponses du professeur ~15–30 min,
3 GGUF concurrents ~45 min–1 h) ; C3 2 à 4 heures ; chaque passe LDLQ complète 1 à 3 heures (§ 4.3) ; C6 une demi-journée.
**Évaluation d'un candidat sur P-I complet : ~3–10 min** ; pour itérer, utilise la moitié dev (~2–5 min) et ne lis P-I complet
qu'à la fin. Remplace les estimations par les durées mesurées dans le rapport.

---

## 4. Adaptations obligatoires du pipeline

### 4.1 Le professeur (C0, C1)

Il n'existe pas de BF16 publié. **Le professeur est le checkpoint publié, déquantifié exactement.**

1. **Inventaire.** Lis tous les en-têtes safetensors : noms, formes, types. Écris-les dans `checkpoint_headers_inkling.json`
   avec le SHA-256 de chaque fichier et la révision Git du dépôt. Compare le décompte de paramètres à l'étude.
2. **Activations quantifiées ?** Cherche les `input_scale` (convention ModelOpt : `weight_scale` FP8 par bloc, `weight_scale_2`
   FP32 par tenseur, `input_scale` pour les activations). S'ils existent, la référence officielle peut quantifier les activations
   (W4A4). **Définition retenue d'avance** : professeur = **W4A16**, poids NVFP4/MXFP8 exacts et activations en BF16/FP32, comme
   llama.cpp, MLX et la référence UD-Q8_K_XL d'Unsloth. Écris cette définition dans `teacher_definition.md` avant toute passe.
3. **Déquantification.** Poids NVFP4 = `e2m1(code) × fp8(weight_scale du bloc) × weight_scale_2`. Avant d'écrire du code,
   vérifie dans les en-têtes l'ordre des quartets et la disposition des blocs ; poids MXFP8 selon le même principe.
   - Le produit E2M1 × E4M3 tient en BF16. **Le facteur FP32 par tenseur, lui, peut sortir de BF16.**
   - Règle déclarée d'avance : déquantifie en FP32, puis mesure `KL(professeur_FP32 ‖ professeur_BF16)` sur la moitié dev du
     protocole. **Si elle est ≤ 1·10⁻⁴**, le professeur BF16 est accepté et documenté.
   - Sinon, garde les experts dans un module qui déquantifie en FP32 à la volée, depuis les codes NVFP4 en mémoire (~16 Go), et
     calcule les Gram en FP32.
4. **Contrôle croisé.** Déquantifie `UD-Q8_K_XL` avec le `gguf-py` du fork. Ses experts NVFP4 doivent être **bit à bit
   identiques** à ta déquantification. Ses autres tenseurs (BF16, et le MXFP8 passé en BF16) doivent être identiques ou à
   l'arrondi près.
   - Une différence signale une erreur d'ordre ou de disposition, comme la permutation des têtes GDN trouvée en V7 : trouve-la et
     corrige-la avant d'aller plus loin.
   - Puis exécute UD-Q8_K_XL dans ton graphe. La KL par rapport à ton professeur doit être ≈ 0 (< 1·10⁻⁴).
5. **Logits.**
   - La tête applique `softcap(z) = 30·tanh(z/30)`.
   - La KL se calcule sur les **200 058 entrées utiles** du vocabulaire. Vérifie comment le modèle masque les entrées de
     remplissage ; si elles ne sont pas masquées, exclus-les explicitement et documente-le.
   - Le cache du professeur (`S["teach"]`) stocke les **logits log-softmax tronqués** aux positions scorées (à mettre en cache
     sur disque si nécessaire), ou l'état final avec la tête et le softcap rejoués. Ne reprends pas tel quel le
     `F.linear(h, T_head)` de Qwen, qui n'a pas de softcap.

### 4.2 La passe séquentielle et le cœur bouclé (C4)

La classe `Seq` de Qwen propage des états cachés en appelant `layer(h, **kwargs)` couche par couche. **Ce mécanisme ne
s'applique pas à Inkling** :
- les résidus d'attention lisent les sorties des blocs précédents ;
- les passes 2 et 3 lisent un message de la passe précédente ;
- le cœur est appelé trois fois.

Ne réimplémente pas ces états. **Stratégie imposée** : à chaque étape, fais un forward complet du modèle élève courant sur la
calibration, avec des crochets qui capturent les entrées des linéaires de la couche i à **toutes ses applications**, puis une
exception d'arrêt dès que la couche i a été appliquée pour la dernière fois :
- couche 0–5 : arrêt après le prélude ;
- couche du cœur : arrêt après la passe 3 de cette couche ;
- couche 18–23 : arrêt après elle.

Le coût est plus élevé que `Seq` (au plus ~24 forwards de la calibration), mais acceptable : le modèle n'a que 5,4 G
« activations » par token.

- **Ordre dans une couche**, comme F1 :
  1. projections d'entrée de l'attention : wq, wk, wv et `wr_du` s'ils partagent la même entrée → même Gram, lignes empilées ;
  2. projection de sortie wo ;
  3. pour chaque expert : `w13` (gate+up, mêmes entrées) ;
  4. puis `w2` de chaque expert, avec le Gram recalculé après la quantification des `w13`.
- **Gram d'une matrice du cœur** = somme sur les 3 passes, ce qui découle naturellement des crochets. Les passes 2 et 3 traversent
  des couches du cœur pas encore quantifiées (indices > i) : c'est l'élève courant, comme dans F1.
- **Second balayage** du cœur : variante V2 (§ 7), pas la passe L0.
- **Ne pas quantifier** (copie FP16 ou FP32 selon le type d'origine, comptée dans le paquet) :
  - routeurs (`mlp.gate.weight`, `mlp.gate.bias`) ;
  - contrôleurs de boucle et requêtes inter-passes ;
  - requêtes `attn_res` ;
  - normes, puits, `rel_logits_proj`, `gate_attn`, `sconv`, `q_l2_clip`/`rel_l2_clip`.

  Si un de ces tenseurs dépasse 10 M poids, signale-le et décide par une variante mesurée.
- **`embed`** : E8 au plus proche sans rétroaction, même multiplicateur.
- **`unembed`** : LDLQ sur le Gram des états finaux normalisés de l'élève (avant la tête, donc avant le softcap).
- **Points de reprise** tous les 4 couches, comme `_ckpt_save` / `_ckpt_load` : la passe Qwen a saturé la mémoire sur la tête.

### 4.3 Le MoE : Gram par expert (C3, C4)

- **Capture.** Crochet sur l'entrée du bloc MoE et sur la sortie du routeur (indices top-8 et poids normalisés × `route_scale`).
  Pour chaque expert e : `H_e = Σ x xᵀ` sur les tokens routés vers e, en FP32 sur GPU puis FP64 pour la factorisation.
  - Comptes : tokens × applications (×3 dans le cœur).
  - Mémoire : 128 × 2304² × 4 o ≈ 2,7 Go pour `w13`, 0,7 Go pour `w2`.
- **Passe L0 = transposition stricte** : Gram **non pondéré**. C'est l'erreur locale de sortie du linéaire, comme F1. Le Gram
  pondéré par le poids de routage au carré est la variante V3.
- **Seuil de couverture, déclaré d'avance** : un expert dont le Gram compte moins de `n_min = 4 × 2304 = 9 216` lignes, ou moins
  de `4 × 1152` pour `w2`, est quantifié au plus proche sans rétroaction, au même multiplicateur. Liste-les dans le résultat.
  L'amortissement reste à 1 % de la diagonale moyenne.
- **Accord de routage** (nouvelle mesure, sur la moitié dev) : pour chaque couche et chaque application, la part des 8 experts
  identiques entre l'élève et le professeur, et la part des tokens dont le top-1 expert change. À rapporter avec chaque KL.
- **Pas par ligne** : `w13` a 2304 lignes de 2304 colonnes (0,007 bpw de pas), `w2` 2304 lignes de 1152 colonnes (0,014 bpw).
  Matrices traitées une par expert : 3 072 `w13` et 3 072 `w2`. Groupe les experts d'une couche en lots sur le GPU pour l'efficacité.

### 4.4 Codec et paquet

- Copie `formic_v7/codec/` en `formic_inkling/codec/`. Un fichier par matrice d'expert est accepté : l'en-tête (~656 o) coûte
  ~4 Mo au total. Mesure-le.
- Nomme les matrices `llm.layers.{i}.mlp.experts.w13_weight.{e}` / `.w2_weight.{e}`. Adapte `load_package` pour réécrire la tranche
  `[e]` du tenseur fusionné.
- Le décodé canonique reste `bf16_rne(f32(U) · (f32(step)/2))`. S'il faut un facteur FP32 par tenseur (§ 4.1), tu ne le stockes
  pas : tu quantifies le poids déquantifié complet, et le pas par ligne l'absorbe.
- **Vision** : E8 au plus proche à 5 bpw. **Drafter DSpark** : E8 au plus proche à 4 bpw (non évalué en KL). **Annexes** :
  tokenizer, `tiktoken/`, gabarit, configurations, copiés tels quels.
- `package.py verify` doit passer. Le paquet complet doit tenir dans 8 000 000 000 à 9 000 000 000 octets.

---

## 5. Évaluation et comparaison à Unsloth

### 5.1 Ce que l'on sait du corpus d'Unsloth (recherche du 22 novembre 2026)

- La fiche `unsloth/Inkling-2-Mini-GGUF` et la page `unsloth.ai/docs/models/inkling-2-mini` disent seulement : « We compare each
  GGUF with `UD-Q8_K_XL` across seven domains using identical llama.cpp settings. »
- **Les sept domaines, le corpus, le nombre de fenêtres, la longueur de contexte, le gabarit et la commande ne sont publiés nulle
  part** : fiche, page, discussions du dépôt, page de méthode Dynamic 3.0. Le dépôt ne contient ni fichier imatrix ni journal de KL.
- La page de méthode Dynamic 3.0, qui porte sur Qwen3.8-27B et pas sur Inkling, décrit deux autres mesures :
  - **Divergence-300 @32** : 300 exemples hors calibration tirés de Terminal-Bench 2.1, DeepSWE, Harbor, MathArena 2025-26 et
    d'invites non latines ou longues ; décodage glouton de 32 tokens, BF16 contre quantifié ;
  - des mesures KL sur Wikitext et du code, avec les jeux `Calibration_v3` et `Calibration_v5` (gists de bartowski1182 et
    tristandruyen).

  Rien ne dit que le tableau Inkling utilise l'une de ces mesures.
- Autres repères publiés, **non comparables** entre eux :
  - ggml-org : Wikitext-2, 40 × 512 tokens, contre BF16 ;
  - AtomicChat : 30 langues, 87 × 4096 tokens, `llama-perplexity --kl-divergence`.

**Conséquence : on ne peut pas comparer nos KL aux 0,0726 / 0,0595 publiés.** La seule comparaison défendable est celle-ci :
**mesurer nous-mêmes les GGUF d'Unsloth sur notre protocole**, contre la même référence, dans les deux moteurs. L'opérateur peut
demander le corpus à Unsloth (discussion sur le dépôt HF) ; **toi, tu ne contactes personne**.

### 5.2 Protocole principal `P-I` (C2)

À construire avant toute passe, puis gelé (hash dans `protocol_ink.json`). Conçu pour être lisible **à l'identique par llama.cpp**.

- **Fenêtres de 4096 tokens** lues depuis un contexte vide. **Positions scorées = la seconde moitié (2048–4095)**, ce qui
  correspond exactement à ce que score `llama-perplexity -c 4096`.
- **Neuf domaines, 24 fenêtres chacun, 216 fenêtres au total** (~440 000 positions scorées) :
  1. **Agentique et outils** : conversations avec appels d'outils au gabarit d'Inkling.
  2. **Code** : Python, C/C++, JS, Rust.
  3. **Mathématiques** : énoncés et solutions. **Pas MetaMathQA**, qui est dans la calibration.
  4. **Raisonnement** : invites dont la réponse, raisonnement compris, est **générée par le professeur** au niveau d'effort
     par défaut (`high`).
  5. **Encyclopédique et rappel factuel** : la faiblesse publiée du modèle.
  6. **Multilingue** : ≥ 8 langues dont le français, au moins deux écritures non latines.
  7. **Contexte long** : documents dont le préfixe dépasse 2048 tokens utiles.
  8. **Instructions** : conversations générales.
  9. **Web.**
- Pour les domaines à invite (1, 4, 8), la réponse scorée est celle du professeur, produite une fois puis figée. Le gabarit de
  conversation est celui d'Inkling (`chat_template.jinja`).
- **Moitiés `dev` et `conf`** : alternance par domaine, comme `arki_teacher`. **`conf` est scellée** : résultats écrits dans
  `/workspace/work_ink/sealed/`, jamais imprimés avant le gel.
- **Disjonction** par identifiant de document avec la calibration (§ 6), plus le filtre de 13 mots communs de `build_cal2.py`.
- **Tokenisation llama.cpp** : écris chaque fenêtre en texte. Vérifie que `llama-tokenize` de chaque build redonne exactement les
  mêmes 4096 identifiants. Les fenêtres qui diffèrent sont exclues **des deux moteurs** ; indique leur nombre.
- **Métriques**, toutes à rapporter globalement, par domaine et par moitié :
  - KL(professeur ‖ candidat) sur tout le vocabulaire, en nats, moyenne par token ;
  - 99ᵉ centile de la KL ;
  - accord top-1, que llama.cpp appelle « same top p » ;
  - accord de routage ;
  - IC 95 % appariés par fenêtre, avec la méthode de V7.

**Mesures secondaires, à titre de recoupement, jamais pour sélectionner :**
- **P-W** : Wikitext-2 test, `-c 512`, comme ggml-org. Sert à situer nos chiffres face aux publications qui suivent cette pratique.
- **P-D32** : 300 invites agentiques, mathématiques et non latines, hors calibration, sur le modèle de Divergence-300.
  Décodage glouton de 32 tokens : part des séquences identiques au professeur et position de la première divergence.

### 5.3 Concurrents à mesurer (C2)

**Trois concurrents seulement**, choisis par l'opérateur parce que la cible fait moins de 9 Go. Télécharge du dépôt
`unsloth/Inkling-2-Mini-GGUF` :

| Rôle | Fichier | Taille publiée | KL publiée (protocole inconnu) |
|---|---|---:|---:|
| Concurrent à octets égaux (**budget de référence**) | `UD-IQ2_XXS` | 8,49 Go | 0,0726 |
| Concurrent légèrement plus gros (9,14 Go : au-dessus de notre plafond de 9 Go) | `UD-Q2_K_XL` | 9,14 Go | 0,0595 |
| Plafond de comparaison, ~3 Go plus gros que nous | `UD-IQ3_XXS` | 11,66 Go | 0,0269 |

Ajoute `UD-Q8_K_XL` (22,87 Go) **uniquement comme référence de mesure** : c'est elle contre laquelle Unsloth calcule sa KL, et
pour le contrôle croisé du § 4.1. Ce n'est pas un concurrent. N'en télécharge aucun autre (ni `UD-IQ1_M`, ni `IQ4_NL`). Note la
révision de chaque fichier.

**Lecture des comparaisons.** Seule `UD-IQ2_XXS` est à octets égaux : c'est elle qui fixe le critère de réussite du § 7. Contre
`UD-Q2_K_XL`, qui est plus grosse que nous, égaler ou battre sa KL avec moins d'octets est un résultat fort, à rapporter sans le
présenter comme une comparaison à octets égaux. `UD-IQ3_XXS` situe l'écart avec un modèle ~3 Go plus gros.

1. **Dans notre graphe.** Déquantifie avec le `gguf-py` du fork. Vérifie la disposition tenseur par tenseur (cosinus ≥ 0,99 avec le
   professeur pour les 2 bits, ≥ 0,9999 pour UD-Q8_K_XL). Charge dans le modèle Transformers et évalue sur P-I.
2. **Dans llama.cpp** (fork) :
   - créer la base de KL avec le professeur : `llama-perplexity -m UD-Q8_K_XL.gguf -f pi.txt -c 4096 -b 4096 --kl-divergence-base pi_ref.kld` ;
   - puis, pour chaque GGUF : `llama-perplexity -m <quant>.gguf -c 4096 -b 4096 --kl-divergence-base pi_ref.kld --kl-divergence` ;
   - archive les sorties complètes ;
   - **attention au disque** : le fichier `--kl-divergence-base` stocke ~2 octets × 201 024 par position scorée, soit ~400 Ko.
     Sur P-I complet (442 000 positions) cela donne ~178 Go. **Pour le volet llama.cpp, utilise un sous-ensemble de 72 fenêtres
     (8 par domaine, mêmes pour tous les modèles, ~59 Go)** et rapporte aussi la KL de notre graphe sur ce même sous-ensemble.
     Le graphe Transformers garde P-I complet, car il met en cache les états cachés (~2 Go) et non les logits.
3. **Contrôle de cohérence des deux moteurs.** Pour chaque GGUF, KL mesurée dans notre graphe et dans llama.cpp, et leur rapport.
   On attend le même ordre. Un écart de plus de 15 % est à expliquer avant de continuer (tokenisation, positions, masque du vocabulaire).
4. **Octets texte de chaque GGUF** : somme des tailles des tenseurs du chemin texte lue par `gguf-py`, métadonnées exclues. Ces
   octets fixent nos budgets.

**Option recommandée.** Exporter notre modèle décodé en GGUF BF16 non quantifié (~52 Go, avec le convertisseur du fork) et le
passer dans la même commande llama.cpp. On obtient ainsi un chiffre Formic directement dans le moteur et le protocole des
concurrents.

---

## 6. Calibration (C3)

- **Reprends les sources** de `formic_arch/data/build_corpus.py` et `formic_v7/data/build_cal2.py` :
  - Wikipedia en 8 langues, FineWeb-Edu, MetaMathQA, UltraChat 200k, du code ;
  - ToolACE, glaive-function-calling-v2, OpenThoughts-114k, glaive reasoning-v1-20m.

  Écris `formic_inkling/data/build_cal_ink.py`.
- **Rendu** : tokenizer d'Inkling ; conversations, outils et raisonnement rendus avec **le gabarit d'Inkling**, y compris son
  format d'appel d'outils et de bloc de raisonnement. Lis le gabarit, n'invente pas de balises. Un document brut sur deux commence
  par le jeton de début de document d'Inkling, s'il en existe un ; vérifie dans `tokenizer_config.json`.
- **Volume** : **2 097 152 tokens** (1024 documents × 2048). Composition :
  - 35 % conversations, outils et raisonnement ;
  - 20 % code ;
  - 15 % prose et encyclopédique ;
  - 15 % multilingue ;
  - 10 % mathématiques et sciences ;
  - 5 % web.

  Si la couverture des experts (§ 4.3) reste sous le seuil pour plus de 5 % des experts, passe à 4 194 304 tokens avant C4 et
  note-le. Ne choisis pas le volume d'après une KL.
- **Disjonction** stricte avec P-I, P-W et P-D32, avec le contrôle de V7.
- Sortie `calibration_ink.json`, qui contient :
  - la composition et les identifiants des sources ;
  - le nombre de tokens routés par expert, couche et passe, calculé sur le professeur ;
  - les experts sous le seuil.

---

## 7. Variantes, sélection et gel

**L0 (référence obligatoire).** La transposition stricte décrite au § 4 :
- un seul multiplicateur global pour toutes les matrices E8 du texte ;
- réglé par `calibrate_mult` pour que les **octets réels** du chemin texte (E8 + petits tenseurs) égalent ceux de
  `UD-IQ2_XXS` à ±0,1 %.

**Variantes autorisées.** Toutes à octets texte égaux, chacune comparée à L0 sur `P-I dev` :

| Id | Changement | Raison |
|---|---|---|
| V1 | Cœur bouclé +Δ, prélude et coda −Δ à octets égaux ; Δ ∈ {0,2 ; 0,4} bit, par deux multiplicateurs | erreur injectée 3 fois dans le cœur |
| V2 | Second balayage du cœur : Gram recalculés sur l'élève complet, LDLQ refait depuis les poids d'origine | ordre circulaire de la boucle |
| V3 | Gram des experts pondéré par (poids de routage)² | erreur réellement transmise par le MoE |
| V4 | Attention et `unembed` à un multiplicateur plus fin (≈ +1 bit), experts compensés | 3,5 % des poids, chemins sensibles |
| V5 | Calibration doublée (4,2 M tokens) | couverture des experts rares |

Pas plus de **trois passes complètes en plus de L0** sans accord de l'opérateur. Combine les variantes gagnantes, au plus une
combinaison.

**Aucun second budget.** Une seule taille est produite (celle de `UD-IQ2_XXS`). Si l'opérateur veut plus tard un point plus bas
ou plus haut, il le demandera.

**Règle de sélection.** Elle est écrite et commitée dans `results/selection_rule.lock.json` **avant** de lancer la première
variante. Modèle :
- KL la plus basse sur `P-I dev` parmi les candidats qui respectent trois garde-fous :
  - aucun domaine au-dessus de 1,10 × UD-IQ2_XXS ;
  - accord de routage moyen ≥ celui de L0 − 1 point ;
  - paquet complet ≤ 9 000 000 000 octets ;
- départage par le plus grand nombre de domaines gagnés si l'écart est inférieur à son IC 95 %.

**Gel.** Choix figé dans `results/selection.lock.json` (commit), puis lecture de `P-I conf` et de P-W et P-D32 sur le candidat
retenu et sur les concurrents.

**Critères de réussite**, déclarés ici et à recopier dans le rapport sans les modifier :

| Niveau | Critère, sur `P-I conf`, à octets texte égaux à UD-IQ2_XXS, dans notre graphe |
|---|---|
| Échec | KL Formic ≥ KL UD-IQ2_XXS, ou IC 95 % du gain qui contient 0 |
| Atteint | gain ≥ 10 %, IC 95 % entièrement sous 0 |
| Renforcé | gain ≥ 25 % (hypothèse de l'étude : 25–45 %, **estimée**) |
| Contrôle | même conclusion dans llama.cpp si l'export GGUF BF16 du § 5.3 a été fait |

---

## 8. Paquet final et rapport (C6)

1. **Reconstruire** le modèle **depuis les fichiers** du paquet par un processus séparé (`load_package`). Réévaluer sur `P-I dev` :
   les KL doivent être identiques à 6 décimales près aux chiffres en mémoire.
2. **Vision** à 5 bpw. Évaluer sur ≥ 96 invites image + texte (domaines : documents, graphiques, captures d'écran, photos), avec
   la réponse du professeur sur 64 tokens. Comparer la tour en BF16 et la tour du paquet.
3. **Drafter** à 4 bpw. Tenter une mesure du taux d'acceptation des brouillons dans Transformers ou SGLang. Si ce n'est pas
   exécutable, l'écrire (« non évalué »), comme le MTP en V7.
4. `package.py verify` ; taille réelle sur disque ; manifeste archivé dans `results/package_manifest_ink.json`.
5. **Rapport `RAPPORT_INKLING_F1.md`.** Commencer par une réponse en une page avec :
   - le tableau concurrents × KL × top-1 × routage, dans les deux moteurs ;
   - les octets par bloc et le débit réel par famille : prélude, cœur, coda, attention, vocabulaire, Vision, drafter ;
   - les coûts mesurés : durée des passes, pic VRAM.

   Ajouter les limites : protocole maison ; corpus d'Unsloth inconnu ; aucune tâche exécutée ; drafter et Vision partiels ; une
   seule graine.
6. **Recette `RECETTE_LUMEN_INKLING.md`** : tableau « élément / valeur retenue / code », « ce qui change par rapport à F1 »,
   commandes de reproduction, limites.

---

## 9. Pièges connus

- **Experts fusionnés.** `w13_weight` est `[E, 2·1152, 2304]` (à vérifier) : gate et up sont empilés en lignes et partagent le
  même Gram. L'ordre gate/up dans la fusion se vérifie par un forward d'un expert isolé contre l'implémentation de référence.
- **Softcaps du MLP** (4 et 25). Ils agissent après `w13` : ils ne changent pas le Gram de `w13`, mais changent les entrées de
  `w2`, que la capture après quantification de `w13` prend en compte.
- **Couches globales 5, 11, 17, 23** : ont des `q/k/v_input_norm` et des `sconv` à `static_w` / `weight_proj`. Ce sont de
  petits tenseurs, laissés intacts.
- **Mémoire de l'évaluation.** Le vocabulaire de 201 024 entrées rend un log-softmax sur 4096 positions coûteux : traite par
  morceaux de 1024, comme `arki_eval`.
- **Gabarit.** L'effort de raisonnement se règle dans le gabarit (`reasoning effort` de 0 à 0,99, défaut 0,9). Fixe-le pour la
  génération des réponses du professeur et note-le.
- **Ne pas confondre** les 26,2 G (fiche) et 26,4 G (annonce) : utilise ton inventaire C0.
- **Le MXFP8 des couches 20–23** est requantifié en E8 comme le reste. Il n'y a pas de raison de le protéger : c'est une
  précision plus fine que la cible.

---

## 10. Ce que tu rends à la fin, et ce que tu ne fais pas

Rends : les fichiers du § 0, un résumé de 10 lignes dans le journal du pod, la liste de ce qui n'a pas été exécuté.

Ne fais pas :
- de runtime d'inférence : c'est un projet séparé ;
- de benchmark de tâches : il demande l'accord de l'opérateur ;
- d'optimisation de la composition de la calibration au-delà de V5 ;
- de comparaison chiffrée avec les KL publiées par Unsloth autrement que comme repère étiqueté « publié, protocole inconnu ».

## Sources

- https://huggingface.co/thinkingmachines/Inkling-2-Mini (fiche, `config.json`, `hf_quant_config.json`)
- https://thinkingmachines.ai/news/inkling-2-mini/
- https://huggingface.co/blog/inkling2mini
- https://huggingface.co/unsloth/Inkling-2-Mini-GGUF et https://unsloth.ai/docs/models/inkling-2-mini (tableau KL, référence UD-Q8_K_XL)
- https://unsloth.ai/docs/basics/dynamic-3.0-ggufs (Divergence-300 @32, Calibration_v3/v5)
- https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF (Wikitext-2, 40 × 512) ; https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF (30 langues, 87 × 4096)
