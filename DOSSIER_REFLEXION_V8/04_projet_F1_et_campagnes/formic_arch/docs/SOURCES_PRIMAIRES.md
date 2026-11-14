# Sources primaires — compression bas-débit d'un LLM hybride (LUMEN Q1 V3.1)

Date : 2026-11-13. Chaque entrée est étiquetée **VÉRIFIÉ** (page récupérée dans cette session) ou **NON VÉRIFIÉ**.
Note : les extraits chiffrés proviennent de pages web converties/résumées par un outil ; les PDF arXiv bruts étaient illisibles, donc seuls les chiffres lus sur pages HTML/abs sont marqués VÉRIFIÉ.

## 1. QuIP# — VÉRIFIÉ (abs + ar5iv)
Tseng, Chee, Sun, Kuleshov, De Sa. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*. ICML 2024. arXiv:2402.04396. https://arxiv.org/abs/2402.04396
- Abstract : transformée de Hadamard aléatoire (incohérence), quantification vectorielle par codebooks basés sur le réseau E8 (« optimal 8-dimension unit ball packing »), fine-tuning.
- Table 4 (wikitext2, ctx 4096), FP16 / 4b / 3b / 2b : Llama-2 7B 5.12 / 5.19 / 5.41 / 6.19 ; 13B 4.57 / 4.63 / 4.78 / 5.35 ; 70B 3.12 / 3.18 / 3.35 / 3.91. (Table 2, ctx 2048, 7B : 5.56 / 5.79 / 6.66 à 4/3/2 b.)
- Citation : « On Llama 2, QuIP# 3 bit outperforms a theoretical lossless 4 bit model (FP16 at 4 bits) » ; QuIP# serait la première PTQ où les modèles 3 bits « scale » mieux que 4 bits, ce qui « directly refutes » la thèse « 4 bits optimal » de Dettmers & Zettlemoyer. À 2 et 3 bits : surpasse nettement AQLM de taille similaire ou « achieves similar performance with a smaller model ».
- Block LDLQ : décomposition LDL par blocs du Hessien proxy ; arrondi bloc par bloc avec rétroaction linéaire de l'erreur des blocs précédents ; borne sur la perte proxy attendue. (Dernier ~16k caractères de la page non lus.)

## 2. QuIP — VÉRIFIÉ (abs seulement, partiel)
Chee, Cai, Kuleshov, De Sa. *QuIP: 2-Bit Quantization of Large Language Models With Guarantees*. NeurIPS 2023 (venue NON VÉRIFIÉ sur la page ; la page n'indique que arXiv). arXiv:2307.13304. https://arxiv.org/abs/2307.13304
- Abstract : arrondi adaptatif minimisant un objectif quadratique proxy + pré/post-traitement par matrices orthogonales aléatoires ; « first LLM quantization methods that give viable results at two bits per weight » ; la théorie « also applies to an existing method, OPTQ ».
- Le nom LDLQ n'apparaît pas dans l'abstract ; l'énoncé explicite « LDLQ est équivalent à GPTQ » est cité par YAQA (§8, VÉRIFIÉ via cette dernière), pas lu dans QuIP même : NON VÉRIFIÉ à la source.

## 3. GPTQ — VÉRIFIÉ (abs)
Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023. arXiv:2210.17323. https://arxiv.org/abs/2210.17323
- Quantification one-shot par information du second ordre approchée ; modèles 175B en ~4 h GPU ; 3 ou 4 bits avec perte minimale ; précision « raisonnable » à 2 bits et ternaire ; accélérations 3.25x (A100), 4.5x (A6000).

## 4. Grand modèle peu de bits vs petit modèle plus de bits
- **VÉRIFIÉ (abs)** Dettmers & Zettlemoyer, *The case for 4-bit precision: k-bit Inference Scaling Laws*, arXiv:2212.09720 (ICML 2023 : NON VÉRIFIÉ ; la page n'indique aucun venue). https://arxiv.org/abs/2212.09720
  - > 35 000 expériences zero-shot, 3 à 8 bits, 19M–176B paramètres (BLOOM, OPT, NeoX/Pythia, GPT-2). Conclusion : 4 bits « presque universellement optimal » à nombre total de bits fixé. Exemple : un 30B 8 bits vs un 60B 4 bits aux mêmes bits totaux (le 4 bits gagne).
  - Gains seulement via petits blocs de quantification et choix du type de données.
- **VÉRIFIÉ** QuIP# (§1) contredit cette conclusion pour les quantiseurs avancés : les modèles 3 bits scalent mieux que 4 bits. Par dessus : le tableau QuIP# montre qu'en absolu, à modèle donné, 4b < 3b < 2b en ppl ; le gain vient de la taille.
- AQLM, « Compressing LLMs: The Truth is Rarely Pure and Never Simple » : NON VÉRIFIÉ (non consultés).
- Optimum rapporté : 4 bits (scalaire, 2022) ; ~3 bits avec VQ/réseau (QuIP#). Pas de source lue qui fixe un optimum à ~2.8 b.

## 5. Suppression de couches
- **VÉRIFIÉ (abs)** Gromov, Tirumala, Shapourian, Glorioso, Roberts. *The Unreasonable Ineffectiveness of the Deeper Layers*. ICLR 2025 (camera-ready v2). arXiv:2403.17887. https://arxiv.org/abs/2403.17887
  - Les performances QA se dégradent peu jusqu'à « up to half » des couches retirées pour certains modèles ouverts ; bloc choisi par similarité entre couches ; « healing » par QLoRA (une A100 40 Go).
  - Effet sur la loss vs QA : NON VÉRIFIÉ (PDF illisible ; l'abs ne mentionne que les benchmarks QA). Mon souvenir (loss monte en douceur alors que QA chute en transition abrupte) n'est pas confirmé ici.
- **VÉRIFIÉ (abs)** Men et al. *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect*. arXiv:2403.03853 (v3 oct. 2024). https://arxiv.org/abs/2403.03853
  - Métrique Block Influence (BI), suppression des couches à faible BI, « compatible with quantization ». Pas de chiffres dans l'abs : chiffres NON VÉRIFIÉS.

## 6. Élagage structuré en largeur sans ré-entraînement
- **VÉRIFIÉ (HTML)** Ashkboos, Croci, Gennari do Nascimento, Hoefler, Hensman. *SliceGPT: Compress Large Language Models by Deleting Rows and Columns*. ICLR 2024. arXiv:2401.15024. https://arxiv.org/abs/2401.15024
  - Table 1, wikitext2 ppl, dense / 20 % / 25 % / 30 % de tranchage (sans recovery fine-tuning) : Llama-2 7B 5.47 / 6.64 / 7.24 / 8.12 ; 13B 4.88 / 5.81 / 6.30 / 6.99 ; 70B 3.32 / 4.25 / 4.60 / 5.05.
  - Abstract : jusqu'à 25 % des paramètres (embeddings inclus) retirés ; LLAMA2-70B, OPT 66B, Phi-2 gardent 99 %, 99 %, 90 % de la perf zero-shot.
- FLAP, LLM-Pruner, chiffres Llama-3 : NON VÉRIFIÉ.

## 7. Activations massives et quantification
- **VÉRIFIÉ (HTML)** Sun, Chen, Kolter, Liu. *Massive Activations in Large Language Models*. COLM 2024. arXiv:2402.17762. https://arxiv.org/abs/2402.17762
  - LLaMA2-7B : « at most four » activations massives par état caché (~40 000), dimensions fixes 1415 et 2533. Table 1 : top-4 = 2622.0, 1547.0, 802.0, 477.3 vs médiane 0.2.
  - Table 2 (100 séquences) : top-1 moyen 2556.8 ± 141.0, top-2 −1507.0 ± 83.0 ; « their values are constants and input agnostic ».
  - Table 3 : mises à zéro -> ppl « inf », accuracy zero-shot moyenne 68.95 % -> 36.75 % ; remplacées par leur moyenne -> ppl 5.47 (wikitext), accuracy 68.94 % (inchangée). Elles agissent comme termes de biais.
- **VÉRIFIÉ (abs)** Ashkboos et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. arXiv:2404.00456. https://arxiv.org/abs/2404.00456 : rotation qui supprime les outliers de l'état résiduel, sortie inchangée ; W/A/KV en 4 bits ; Llama-2-70B perd au plus 0.47 ppl wikitext2, 99 % du zero-shot.
- SpinQuant, OSTQuant, PrefixQuant : NON VÉRIFIÉ.
- Remarque : la rotation Hadamard de QuIP# joue un rôle analogue côté poids (incohérence) ; l'implication pour des dimensions massives du flux résiduel à 5120 est une inférence, non un fait cité.

## 8. Pondération Fisher / second ordre
- **VÉRIFIÉ (abs)** Kim, Hooper, Gholami, Dong, Li, Shen, Mahoney, Keutzer. *SqueezeLLM: Dense-and-Sparse Quantization*. ICML 2024. arXiv:2306.07629. https://arxiv.org/abs/2306.07629 : quantification non uniforme guidée par la sensibilité + décomposition dense/sparse ; 3 bits quasi sans perte ; écart de ppl à FP16 réduit jusqu'à 2.1x à mémoire égale. (Usage de la Fisher diagonale : NON VÉRIFIÉ dans cette page.)
- **VÉRIFIÉ (abs + HTML)** Tseng, Sun, De Sa. *Model-Preserving Adaptive Rounding* (YAQA). arXiv:2505.22988 (page indique ICML 2026). https://arxiv.org/abs/2505.22988
  - Erreur mesurée en sortie du réseau ; « provably better than GPTQ/LDLQ and empirically reduces the KL by ≈30% » ; Hessien approché H ≈ H_O ⊗ H_I (facteurs de Kronecker, itération de puissance) ; « LDLQ is equivalent to GPTQ ».
  - Table 1 (QTIP + incohérence, sans finetuning), Llama 3.1 8B Instruct (BF16 W2 6.50), W2 ppl LDLQ / YAQA-A / YAQA-B : 2 b 9.39 / 8.79 / 8.39 ; 3 b 7.05 / 6.89 / 6.85 ; 4 b 6.63 / 6.62 / 6.61. KL 2 b : 0.356 / 0.284 / 0.241. 70B Instruct (BF16 3.52) : 2 b 6.02 / 5.56 / 5.30 ; 3 b 4.26 / 4.10 / 4.01 ; 4 b 3.74 / 3.73 / 3.69. (Alignement des blocs de table « garbled », meilleure lecture.)
  - Coût : ~20-30 h GPU pour un modèle 10B (esquisses A/B).
- **VÉRIFIÉ (recherche web + HTML partiel)** Kim et al. *GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance*. ICML 2025. arXiv:2505.07004. https://arxiv.org/abs/2505.07004 : gradients de la loss finale injectés dans l'objectif par couche (Hessien bloc-diagonal approché par Fisher empirique, contre Fisher diagonale pour les travaux précédents). Wikitext2, ctx 4096 : LNQ -> +GuidedQuant, Llama-2-7B 2.01 b 23.31 -> 8.83 ; 3.03 b 5.89 -> 5.57 ; 70B 2.01 b 5.23 -> 5.04 ; 3.01 b 3.50 -> 3.47 (Table 3). QTIP -> +GuidedQuant (Table 4) : 7B 2.00 b 6.82 -> 6.11 ; 3.00 b 5.38 -> 5.28 ; 70B 2.00 b 3.87 -> 3.80 ; 3.00 b 3.27 -> 3.25. Gains décroissants avec la taille et le débit.
- Lecture : le gain d'une pondération par sensibilité est grand à 2 b sur petits modèles et faible (<= 0.1 ppl) à ~3 b sur gros modèles. À notre débit (~2.8 b) et grande taille : gain attendu modeste (inférence, non mesurée).

## 9. Gated DeltaNet / Qwen3-Next
- **VÉRIFIÉ (abs)** Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. ICLR 2025. arXiv:2412.06464. https://arxiv.org/abs/2412.06464 : règle delta à porte (gating pour l'effacement adaptatif + règle delta pour les éditions ciblées), algorithme d'entraînement parallèle ; surpasse Mamba2 et DeltaNet ; hybrides avec attention à fenêtre glissante ou Mamba2. L'équation exacte de la récurrence n'a pas été lue : NON VÉRIFIÉ.
- Sensibilité à la quantification des états récurrents / linéaire-attention : aucune source trouvée ; NON VÉRIFIÉ.
- Qwen3-Next / Qwen3.5 : page officielle non récupérée (le blog qwen.ai ne rend que « Qwen »). Seulement des sources secondaires via recherche web (non fetchées) : ratio 3:1 Gated DeltaNet / Gated Attention, 80B-A3B à 48 couches, MTP intégré. Statut : NON VÉRIFIÉ pour source officielle.

## 10. Théorie débit-distorsion et gain granulaire
- Cover & Thomas, *Elements of Information Theory*, 2e éd., Wiley 2006, chap. 10 (« Rate Distortion Theory »), sous-section « Parallel Gaussian sources » (reverse water-filling : D_i = min(λ, σ_i²), R = Σ ½ log(σ_i²/D_i)⁺). NON VÉRIFIÉ (citation de mémoire, livre non consulté).
- Conway & Sloane, *Sphere Packings, Lattices and Groups*, Springer (chap. 2, 21). NON VÉRIFIÉ.
- Recherche web : gain granulaire = 10·log10(G(Z)/G(Λ)), G(Z) = 1/12 ; un tutoriel donne G(E8)=0.0717 (et G(D4)=0.1042) -> ≈ 0.65 dB (calcul fait ici : 10·log10(0.08333/0.0717) ≈ 0.65). Les pages n'ont pas été fetchées (résultats de recherche seulement) : NON VÉRIFIÉ au sens strict ; URL de la source tutoriel : https://platform.commit.tu-berlin.de/en/book/cm/chapter/ch16/section/s03

## Récapitulatif des statuts
VÉRIFIÉ : 1 (QuIP#), 2 (partiel), 3, 4 (Dettmers + QuIP#), 5 (abs seulement), 6 (SliceGPT), 7 (Massive Activations, QuaRot), 8 (SqueezeLLM abs, YAQA, GuidedQuant), 9 (GDN abs).
NON VÉRIFIÉ : LDLQ=OPTQ à la source QuIP ; chiffres Gromov/ShortGPT ; FLAP, LLM-Pruner ; SpinQuant/OSTQuant/PrefixQuant ; AQLM et « Truth is Rarely Pure » ; récurrence GDN et sensibilité de l'état ; page officielle Qwen3-Next/3.5 ; Cover & Thomas ; Conway & Sloane / 0.65 dB à la source.
