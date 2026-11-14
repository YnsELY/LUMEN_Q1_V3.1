# Journal de reprise — Mandat V8 « architecture théorique » FORMIC / Lumen F1

> **But de ce fichier** : permettre de reprendre le travail sans refaire les calculs ni les lectures déjà faits.
> **Source** : reconstitué à partir des transcripts complets des deux sessions interrompues (sorties de commandes comprises), pas seulement à partir des messages de progression.
> **Rien n'a été relancé** pour produire ce journal, sauf une relecture des JSON de résultats existants (`formic_arch/results/variant_*.json`). Aucun calcul GPU. Aucun résultat d'origine n'a été modifié.
> **Statut du livrable** : `PROPOSITIONS_ARCHITECTURALES_FORMIC_F1.md` **n'existe pas**. Il n'a pas été commencé. Tout le travail s'est arrêté avant la rédaction.

Dossier d'artefacts associé : `DOSSIER_REFLEXION_V8/05_artefacts_sessions/` (voir §12).

---

## 0. Corrections par rapport à la première version de ce journal

- Le modèle est **`Qwen/Qwen3.8-27B`**, révision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Le code de référence est le module `transformers` **`qwen3_5`** (transformers 5.22.2), qui l'implémente. Ce n'est pas « Qwen3.5 ».
- L'identité du checkpoint **a été vérifiée** : tous les fichiers concordent (voir §3). Le verdict n'était pas dans le résumé, il est dans `identity_check.txt`.
- Le test d'ajustement de l'exposant (`beta.py`) **n'a jamais abouti** : il a dépassé son délai (code 124). Il n'y a donc aucun résultat d'ajustement. Voir §8.

## 1. Chronologie des deux sessions

| Session | Heures (transcript) | Ce qui s'est passé | Fin |
|---|---|---|---|
| `78f73527-ae7c-4365-8749-6095616e0753` | 20:52 → 21:36 | Lecture du mandat et des documents, vérification d'identité, diagnostics, statistiques, bibliographie, calculs statiques, rétrodiction du modèle de bruit | Limite de 5 h atteinte à 21:36, pendant `beta.py` |
| `9445cdc8-3705-421b-a9bd-17c6f096d840` | 23:22 → 23:37 | Relecture de la session précédente, deux calculs statiques, lecture du code du modèle, relecture de résultats bruts | Limite de 5 h atteinte à 23:37 |

Deux fois, le mode auto a refusé d'exécuter une commande (« auto mode's check had no usable answer »). Dans le premier cas, l'avis portait sur un `cd` manquant dans une commande (calcul des facteurs AM/GM des pivots LDL, 21:17). L'agent l'a relancée aussitôt et le résultat figure au §5.5. Aucune commande utile ne semble avoir été perdue, mais le second cas n'est pas documenté.

## 2. Le mandat V8 (résumé fidèle)

Fichier : `PROMPT_AGENT_ARCHITECTURE_THEORIQUE_FORMIC_F1_V8.md`.

- **Mission** : concevoir et justifier **théoriquement** des modifications d'architecture de Formic (Qwen3.8-27B compressé) qui, traitées par la recette **Lumen F1 inchangée**, donnent :
  - **Objectif A** : `B(A1) ≤ B(A0)` et `K(A1) < K(A0)` (meilleure fidélité sans grossir), ou
  - **Objectif B** : `B(A1) < B(A0)` et `K(A1) ≤ K(A0)` (plus petit, même fidélité).
- Taille de référence F1 : **9 974 484 649 octets** ; plafond absolu 10 000 000 000.
- **Interdit** : modifier F1 (E8, LDLQ, règles de pas et de codage, calibration, modules protégés) ; tout forward, quantification, évaluation, entraînement, GPU, téléchargement massif.
- **Autorisé** : lecture, bibliographie, dérivations, calculs légers, exploitation des statistiques déjà enregistrées (provenance et limites à préciser).
- **Pistes déjà essayées et sans gain robuste** : rotations globales, changements d'échelle, biais, ordres de colonnes, réductions de largeur, factorisations (diagnostics spectraux défavorables). Une reprise exige un mécanisme différent et une explication de ce que l'essai précédent ne couvrait pas.
- **Erreur de transcription connue** : le candidat à biais vaut **0,11325**, pas 0,133 (confirmé au §6).
- **Livrable** : `PROPOSITIONS_ARCHITECTURALES_FORMIC_F1.md`, autonome. Contenu exigé : résumé et référence F1 ; carte des directions retenues et écartées ; 2 ou 3 propositions solides (avant/après, équations, preuves conditionnelles, bilan taille/fidélité, compatibilité F1, risques) ; tableau démontré / mesuré / à valider ; plus petit test ultérieur par proposition (non exécuté) ; sources primaires et artefacts. Dérivations longues en annexe. **S'arrêter après ce dossier.**
- Niveaux de preuve à distinguer : publié / mesuré dans le projet / démontré sous hypothèses / à tester. Ne sommer ni soustraire aucune KL. Une baisse de MSE locale ne prouve pas une baisse de KL.

## 3. Identité du checkpoint et référence F1

- Racine locale : `/root/models/qwen3.8-27b`. Arbre enregistré : `.cache/huggingface/trees/37561816e66b5b9153b1b64c10968c18c1e6f4a0.json`.
- **Résultat du hachage (`identity_check.txt`) : `ALL_MATCH`, 53,4 s.** Les 18 shards `model-000NN-of-00018.safetensors` et `tokenizer.json` sont comparés en SHA-256. Les autres fichiers (config, vocab, merges, index, chat_template, etc.) le sont en identifiant d'objet git. Tous concordent.
- **Réserve** : cette comparaison est faite contre l'arbre enregistré dans le cache local. Elle ne prouve pas la provenance de cet arbre. `formic_arch/results/checkpoint_headers.json` indique d'ailleurs `identity_evidence: "absent"`, `trusted_manifest_hashes_checked: false`, `publisher_provenance_verified: false`. À signaler dans le dossier : les fichiers locaux sont cohérents avec la révision attendue, la provenance indépendante n'est pas établie.
- Empreinte du checkpoint (`checkpoint_headers.json`) : 1199 tenseurs, 27 781 427 952 paramètres, 55 562 855 904 octets, fingerprint `85b7bb26…718e`.
- **Architecture réelle** (en-têtes de tenseurs) :
  - 64 couches texte : **48 GDN** (`linear_attn`) et **16 attention complète** (couches 3, 7, 15, 23, 31, 39, 47, 55, 63 observées dans les stats ; la liste exacte est à relire dans la config).
  - Hidden 5120. MLP 17408 (SwiGLU : `gate`, `up`, `down`). Vocabulaire 248 320. `embed_tokens` et `lm_head` **séparés** (1,27 G paramètres chacun).
  - GDN : `in_proj_qkv` (10240×5120), `in_proj_z` (6144×5120), `in_proj_a` et `in_proj_b` (48×5120), `out_proj` (5120×6144), `conv1d` (10240×1×4), `A_log`, `dt_bias`, `norm` (128).
  - Attention : `q_proj` (12288×5120, contient la porte : `q` et `gate` sont séparés par `chunk` sur la dernière dimension), `k_proj` et `v_proj` (1024×5120), `o_proj` (5120×6144), `q_norm`, `k_norm` (256).
  - Bloc **MTP** (1 couche) et **Vision** (27 blocs, hidden 1152).
- **RMSNorm de ce modèle** : `y = x · rsqrt(mean(x²)+ε) · (1 + w)`. Le poids stocké est `w`, le gain effectif est `1 + w`. Sortie en `(x·w).to(dtype)`.
- **Manifeste F1** (`formic_v7/results/package_manifest_F1.json`) : `Counter` = `('e8','prod')` 581, `('raw',None)` 581, `('e8','shell')` 37. Soit 618 tenseurs E8 et 581 bruts. Aucun `kind` autre. La nature précise de `shell` et `prod` reste à lire dans `formic_v7/codec/README.md` et `fcodec.py` (non retranscrite dans les sorties).

### Débit réel par famille dans le paquet F1 (`package_manifest_F1.json`, calculé : 8·octets / poids)

| Famille | n | Poids (G) | Octets (MB) | bpw moyen | min – max |
|---|---:|---:|---:|---:|---|
| `in_proj_a` | 48 | 0,0118 | 4,13 | 2,7990 | 2,7542 – 2,8196 |
| `in_proj_b` | 48 | 0,0118 | 4,11 | 2,7872 | 2,7494 – 2,8201 |
| `in_proj_qkv` | 48 | 2,5166 | 884,44 | 2,8116 | 2,8076 – 2,8178 |
| `in_proj_z` | 48 | 1,5099 | 530,83 | 2,8124 | 2,8091 – 2,8176 |
| `out_proj` (GDN) | 48 | 1,5099 | 529,68 | 2,8064 | 2,8009 – 2,8092 |
| `mlp.down_proj` | 64 | 5,7043 | 2002,41 | 2,8083 | 2,8042 – 2,8200 |
| `mlp.gate_proj` | 64 | 5,7043 | 2005,33 | 2,8124 | 2,8095 – 2,8193 |
| `mlp.up_proj` | 64 | 5,7043 | 2005,52 | 2,8127 | 2,8112 – 2,8196 |
| `k_proj` | 16 | 0,0839 | 29,36 | 2,7999 | 2,7632 – 2,8124 |
| `o_proj` | 16 | 0,5033 | 176,67 | 2,8080 | 2,7922 – 2,8126 |
| `q_proj` | 16 | 1,0066 | 353,82 | 2,8119 | 2,8087 – 2,8137 |
| `v_proj` | 16 | 0,0839 | 29,38 | 2,8023 | 2,7819 – 2,8128 |
| `embed_tokens` | 1 | 1,2714 | 446,43 | 2,8090 | — |
| `lm_head` | 1 | 1,2714 | 446,67 | 2,8106 | — |
| MTP (`fc`, mlp, attn) | 8 | 0,42 | 212 | **3,976 – 3,999** | — |
| Vision (blocs, merger, patch_embed, pos_embed) | 112 | 0,46 | 284 | **4,59 – 4,998** | — |

Les 498 matrices du texte sont à ≈2,81 bpw. Le gain d'un débit bpw plus bas sur MTP et Vision n'a pas été examiné.

## 4. Documents et fichiers lus (à ne pas relire sans besoin)

Mandat V8 ; `formic_v7/RECETTE_LUMEN_V7_EXP.md`, `RAPPORT_V7.md`, `results/A_LIRE_ETAT_V7.md`, `package_breakdown_F1.json`, `package_F1_cal12_uniform.json`, `v7_selection…` ; `formic_v7/codec/README.md`, `fcodec.py` ; `formic_v7/docs/AUDIT_PIPELINE.md` ; `formic_v7/lab/v7.py` ; `formic_arch/lab/{formic_q.py, seq.py, harness.py}` ; `formic_arch/RAPPORT_ARCHITECTURE_X_LUMEN.md` ; `formic_arch/docs/{GRAPHE_AVANT_APRES.md, REPONSE_DOSSIER_PISTES.md, SOURCES_PRIMAIRES.md}` ; `DOSSIER_PISTES_ARCHITECTURALES_LUMEN.md` ; `RECETTE_MAX_PROPOSEE_V3_1.md` ; `DOSSIER_CONCURRENTS_KL_ET_LEVIERS_LUMEN_V2.md` ; `FORMAT_REFERENCE_E8_V3_1.md` ; `formic_arch/jobs/{004,005,009,038,043,045,048}_*.py` ; `modeling_qwen3_5.py` (lignes 200–910, 935–1300 puis 560–650, 699–709, 763–846). Dépôt : `git log` consulté. Données de travail : `/workspace/work/{stats_fisher.pt, dstat_L1.pt, logs/, packages/F1_cal12_uniform, sealed/}`.

**Contenu des fichiers de statistiques** (`/workspace/work`) :
- `stats_fisher.pt` : clés `X`, `G`, `TS`, `MU`, `ntok` (73 728), `T` (512), `n_docs` (144). `X[nom]` = E[x_j²] de l'entrée de chaque matrice (497 entrées). `MU[nom]` = E[x_j]. `G[nom]` = sensibilité (Fisher diagonal) par ligne de sortie (`lm_head` : 248 320 valeurs). `TS[layers.i.mlp.down_proj]` = saillance de Taylor des 17 408 canaux SwiGLU.
- `dstat_L1.pt` : 401 entrées par matrice avec `dsum`, `dlogmean` (pivots LDL amortis) et `r` (pas par ligne).

## 5. Résultats propres à ces deux sessions (calculés, sorties intégrales dans `DOSSIER_REFLEXION_V8/05_artefacts_sessions/outputs/`)

### 5.1 Activité des canaux SwiGLU (`X` de `down_proj`) — **aucune redondance exacte**

- Aucun canal mort : sur 64×17 408 = 1 114 112 canaux, **130** seulement sont sous 0,1 × la médiane de leur couche, **0** sous 0,01 ×.
- Ces 130 sont dans les couches 60–63. Couche 63 : 46,5 % sous 0,1 × médiane, max/médiane 4474,7, AM/GM 5,71.
- Médiane des X par couche (croît de 3,5·10⁻⁵ en couche 0 à 0,353 en couche 63). max/médiane : 857 en couche 0, 24,7 en couche 32, 504 en couche 52. min/médiane ≥ 0,13 hors couches 60–63.
- **Conséquence** : aucune élagage de canaux SwiGLU « gratuit » n'est soutenu par ces statistiques.

### 5.2 Gains de RMSNorm — **aucune égalisation exploitable, une exception : la dimension 3994**

- AM/GM de g² (g = 1+w) par couche : **1,003 à 1,078** (0,002 à 0,054 bit). Norme finale : 1,011. MTP `pre_fc_norm_embedding` : 1,142 (0,096 bit). → il n'y a presque rien à gagner à repartager des gains de norme.
- **Dimension 3994 du résiduel** :
  - gain `post_attention_layernorm` : **0,0039 (≈ 1/256)** aux couches 0–5 et 8–10, **0,0 exactement à la couche 7**, 0,121 (couche 6), 0,074 (couche 11) ;
  - gain `input_layernorm` : 0,30 – 0,44 aux couches 1, 2, 4, 5, 6, 8, 9, 10, 0,328 à la couche 1.
  - Les MLP des couches 0–10 sont donc **masqués** à cette dimension, mais pas les mixeurs.
- Autres dimensions à petits gains : 253, 4406, 3986, 4939, 2610, 4349, 3842.
- Conditionnement : diviser par un gain 0,0039 amplifie par 256. Ceci est un **obstacle** aux transformations de jauge qui touchent la dimension 3994, noté pour la section « obstacles du graphe réel » du mandat.

### 5.3 Dimensions massives du résiduel — **la dimension 3994 domine**

`X` = E[x_j²] aux entrées des lecteurs du résiduel (`mixer_in` = `in_proj_qkv` ou `q_proj` ; `mlp_in` = `gate/up`) :

| Couche | médiane X (mixer_in) | X[3994] | autres dimensions (X) | part de la moyenne² dans ΣX | `mlp_in` : X[3994] |
|---:|---:|---:|---|---:|---:|
| 0 | 0,522 | — (non dominante) | 1119 : 463 ; 4772 : 265 | 0,086 | — |
| 1 | 0,014 | **39 286** | 3842 : 362 ; 310 : 119 | 0,808 | — (3842 : 135) |
| 2 | 0,022 | 28 433 | 3842 : 378 ; 310 : 218 | 0,775 | — |
| 3 | 0,082 | **95 854** | 310 : 1983 ; 3842 : 572 | 0,929 | — |
| 7 | 0,070 | 90 588 | 310 : 3106 | 0,907 | — |
| 15 | 0,093 | 34 578 | 310 : 2161 | 0,818 | 299 |
| 16 | 0,099 | 13 114 | 310 : 518 | 0,673 | 382 |
| 31 | 0,200 | 10 295 | 310 : 716 | 0,587 | 2864 |
| 32 | 0,217 | 8 976 | 310 : 220 | 0,586 | 3037 |
| 47 | 0,333 | 6 190 | 310 : 447 | 0,437 | 1939 |
| 62 | 0,777 | 973 | 4042 : 210 | 0,187 | 654 |
| 63 | 0,754 | 853 | 4042 : 93 | 0,155 | 884 |

- Rapport max/médiane jusqu'à **3,9·10⁴** (couche 1, `in_proj_qkv`) et `X_top8_share` 0,81 (`diag_structure.json`).
- **Pas par ligne aux écritures dans le résiduel** (`dstat_L1.r`, rapport ligne 3994 / médiane des lignes) : `out_proj` GDN couche 0 : **20,7×**, couche 1 : 15,4×, couche 2 : 8,7×, couche 16 : 6,2×, couche 62 : 3,3×. `o_proj` couche 3 : 17,8×, couche 7 : 11,7×, couche 31 : 3,4×. `down_proj` couche 0 : 11,5×, couche 1 : 3,6×, couche 7 : 2,2×, couche 62 : 1,8×.
- **Lecture** : la dimension 3994 est écrite avec un gros pas par les premières couches, lue avec un second moment énorme par les mixeurs, et masquée aux MLP. C'est le mécanisme « bias terms » de Sun et al. (voir §9) : il s'agit d'une **mesure ici**, l'interprétation causale reste une hypothèse.

### 5.4 Mode commun de `embed_tokens` et `lm_head` (mesuré, 27,4 s CPU)

| | embed_tokens | lm_head |
|---|---:|---:|
| part d'énergie du mode commun (moyenne des lignes) | 0,973 % | **5,313 %** |
| |moyenne| / médiane des normes de ligne | 0,0968 | 0,2328 |
| coefficient de variation du RMS de ligne | 0,165 | 0,129 |
| lignes à énergie nulle | 0 | 0 |
| rapport d'énergie par ligne après centrage (quantiles 1/25/50/75/99 %) | 0,979 / 0,987 / 0,990 / 0,993 / 1,002 | 0,833 / 0,918 / 0,943 / 0,969 / 1,066 |

Test de pondération par la sensibilité (`G` de `lm_head`, 23:34) :
- 1 % des lignes porte **72,3 %** de G. Médiane de G 6,3·10⁻¹³, max 0,0107.
- Facteur d'énergie de bruit **sans pondération**, centrage par la moyenne simple : **0,9469**.
- **Avec pondération par G** : centrage par la moyenne simple : **1,1227 (pire)** ; centrage par la moyenne pondérée par G : 0,9656.
- cos(µ, µ_G) = **−0,6826** (|µ| = 0,2267, |µ_G| = 0,1710, norme médiane des lignes 0,9737).
- 9,52 % des lignes voient leur énergie **augmenter** après centrage simple, et elles portent **83,6 %** de la masse G.
- µ·MU = −2,0452 ; |MU| = 44,642 ; cos = −0,2021 ; borne basse (µ·MU)² = 4,18.
- **Conclusion mesurée** : un centrage naïf de `lm_head` (extraire le mode commun) ne réduit pas l'énergie de bruit pondérée par la sensibilité (×1,12). Une version pondérée par G donne ×0,966, un gain faible. Le gain éventuel dépend de la métrique, pas seulement de l'énergie.

### 5.5 Marges pour LDLQ et égalisation de colonnes

**AM/GM des pivots LDL amortis** (`dstat_L1.pt`, par famille ; les familles qui partagent la même entrée ont des valeurs identiques) :

| Famille | n | médiane | min – max | cas marquants |
|---|---:|---:|---|---|
| `in_proj_qkv` = `in_proj_z` | 48 | 1,215 | 1,086 – 1,626 | L1 1,63 ; L2 1,54 ; L60–62 ≈ 1,10 |
| GDN `out_proj` | 48 | 1,431 | 1,200 – 2,821 | **L60 2,82 ; L61 2,69 ; L62 2,31** |
| `gate_proj` = `up_proj` | 64 | 1,063 | 1,026 – 1,205 | L0 1,20 |
| `down_proj` | 64 | 1,119 | 1,064 – 2,653 | L0 1,77 ; L61 1,64 ; L62 1,79 ; **L63 2,65** |
| `q_proj` = `k_proj` = `v_proj` | 16 | 1,302 | 1,134 – 2,045 | L3 2,04 ; L7 1,91 |
| `o_proj` | 16 | 1,324 | 1,214 – 2,357 | L3 2,36 ; L7 1,72 |
| `lm_head` | 1 | 1,086 | — | — |

Valeurs plus anciennes (`diag_ldl_headroom.json`, 9 couches), avec rotation aléatoire (`/r`) : lecteurs du résiduel ramenés à 1,02–1,08 ; `mlp_in` 1,04–1,15 (r 1,02–1,07) ; `mixer_in` médiane 1,29 (1,10–2,04), part des 8 premiers pivots jusqu'à 0,47 (L3) ; `mixer_out` médiane 1,46 (1,23–2,36) ; `mlp_down` médiane 1,25 (1,07–2,63).

**Facteur de Cauchy-Schwarz** `CS = (Σ√X/n)² / (ΣX/n)` : réduction maximale de l'erreur proxy d'un arrondi sans rétroaction, par colonnes uniformisées (jauge α = 1/2, modèle diagonal). Médiane par famille, et **sans la dimension dominante** (`outputs/equalization_headroom_CS_factors.txt`) :

| Famille | AM/GM(X) médian | CS médian | CS sans dimension dominante | prédiction log-normale |
|---|---:|---:|---:|---:|
| `gate_proj` | 1,31 | 0,808 | 0,926 | 0,872 |
| `down_proj` | 1,19 | 0,900 | 0,908 | 0,918 |
| `in_proj_qkv` | 2,57 | **0,437** | 0,804 | 0,624 |
| GDN `out_proj` | 1,57 | 0,777 | 0,798 | 0,798 |
| `q_proj` | 3,15 | **0,370** | 0,722 | 0,564 |
| `o_proj` | 1,59 | 0,778 | 0,787 | 0,793 |
| `lm_head` | 1,41 | 0,775 | 0,785 | 0,841 |

Cas extrêmes (CS / CS sans dimension dominante) : `in_proj_qkv` L1 0,16 / 0,73, L2 0,20 / 0,76 ; `q_proj` L3 0,07 / 0,65, L7 0,08 / 0,59 ; `out_proj` L0 0,06 / 0,09 ; `down_proj` L63 0,33 / 0,34.

**Lecture** : l'essentiel de la marge diagonale des mixeurs vient de la **seule dimension 3994**. Hors cette dimension, la marge descend vers 0,72 – 0,93. Attention : c'est un modèle d'arrondi **sans rétroaction**. LDLQ capture déjà une partie de cette marge (pivots AM/GM 1,06–1,4 seulement), donc la marge réelle sous F1 est plus petite.

**Amortissement** `λ = 0,01 × moyenne(diag X)` (damp 0,01 de F1) comparé aux X typiques (`9445cdc8`, 23:30) :

| Famille | médiane λ/médiane(X) | max | médiane λ/AM(d) | max | médiane λ/GM(d) | max |
|---|---:|---:|---:|---:|---:|---:|
| `in_proj_qkv` | 0,030 | 0,097 | 0,033 | 0,091 | 0,040 | 0,149 |
| GDN `out_proj` | 0,016 | 0,241 | 0,016 | 0,184 | 0,023 | 0,271 |
| `gate_proj` | 0,015 | 0,022 | 0,020 | 0,028 | 0,021 | 0,034 |
| `down_proj` | 0,013 | 0,081 | 0,017 | 0,038 | 0,019 | 0,100 |
| `q_proj` | 0,037 | 0,208 | 0,037 | 0,106 | 0,047 | 0,217 |
| `o_proj` | 0,016 | 0,041 | 0,026 | 0,038 | 0,035 | 0,078 |
| `lm_head` | 0,015 | — | 0,026 | — | 0,029 | — |

- Aux couches où une dimension massive existe, `λ` est dominé par elle. Exemple : `in_proj_qkv` L1, `λ(sans top-1) / λ = 0,207`, `λ(centré) / λ = 0,193`. L'amortissement y est ≈ 5 fois plus grand que sans la dimension massive. À partir de la couche 60, ce rapport est > 0,85.
- `out_proj` L0 : `λ(centré) / λ = 0,071`, le second moment est presque entièrement de la moyenne² (part 0,94).
- Cet amortissement est **de la recette F1 figée** : l'observation est un diagnostic, pas une recette modifiable.

### 5.6 Poids MTP vs couches principales — **aucun partage possible**

Cosinus entre poids du bloc MTP et couches principales : `k_proj` −0,0008 (L63), −0,0004 (L59), −0,0002 (L3) ; `v_proj` ≈ 0 ; `o_proj` 0,0002 ; `down_proj` −0,0000 ; `gate_proj` 0,0036. Rapports de normes 0,93 – 1,79. Aucun tenseur identique. → pas de mutualisation MTP / couche principale.

### 5.7 Vocabulaire et lignes inutilisées de l'embedding

- Tokenizer BPE : 248 044 entrées + 33 jetons ajoutés = **248 077** identifiants distincts (max 248 076). `embed_tokens` et `lm_head` ont **248 320** lignes → **243 lignes** sans jeton (0,098 %).
- Non chiffré dans la session : l'économie en octets. (Estimation de ma part, non vérifiée : à ≈ 2,81 bpw, une ligne de 5120 poids pèse ≈ 1,8 ko, soit < 1 Mo pour les 243 lignes des deux matrices. Les lignes ne sont pas nulles : « rows with zero energy : 0 ».)

### 5.8 Spectres et structure (diagnostics antérieurs relus)

`diag_structure.json` : `lm_head` ≈ `embed·M` avec **R² = 0,1724** (pas de lien linéaire fort) ; cosinus moyen entre lignes embed/head 0,0093. Rang effectif (erank) de la matrice de poids seule 1857–4342 sur 5120 ; **pondérée par les activations** : `lm_head` 151, `in_proj_qkv` L1 94, `in_proj_z` L1 33, `q_proj` L15 **4,03**, `gate_proj` L31 539 (contre ≈ 3000–3800 sans pondération pour les MLP). Kurtosis par ligne 3,1–4,8 ; RMS de ligne CV 0,04–0,27.

`diag_error_structure.json` (après le vrai LDLQ, 6 couches, 144 documents) : erreur relative pondérée **0,17–1,35 %**, erreur de Frobenius relative 0,18–0,21 à ≈ 2,82 bpw. La part d'erreur dans les 64 premières directions est de 3–17 % ; les mêmes octets dépensés en précision E8 retireraient 30–40 %. Le rang effectif joint de deux couches voisines est ≈ 3930–4260 sur 5120 : pas de base commune à partager. Les 8 colonnes les plus fautives portent jusqu'à 53 % de l'erreur pondérée (`L1.mixer_in`), 39 % (`L16.mixer_in`), < 1 % pour `mlp_down`.

## 6. Résultats bruts des essais antérieurs relus (KL sur 72 documents de sélection)

Recalculés par lecture des JSON (`formic_arch/results/variant_*.json`) ; `ppl_t` (professeur) = 6,46351 partout.

| Variante | KL | agree | ppl élève | Remarque |
|---|---:|---:|---:|---|
| `A0_Q0ldlq` (**témoin LDLQ**) | **0,11232** | 0,89107 | 6,886 | bpw 2,82124 |
| `Ab_Q0ldlq` (biais ancré) | **0,11325** | 0,89339 | 6,821 | **confirme 0,11325, pas 0,133** |
| `A0_Qldlcol_ldlq` (échelles de colonne par pivots) | 0,11864 | 0,89137 | 6,896 | +5,6 % |
| `Arot_Q0ldlq` (rotation, première version) | 0,12131 | 0,89033 | 7,360 | |
| `ArotN_Q0ldlq` (rotation, version native) | **0,10581** | 0,89787 | 6,834 | −5,8 % sur 72 documents |
| `ArotN_Q0ldlq_big` | 0,10581 | 0,89787 | 6,834 | identique à `ArotN_Q0ldlq` |
| `A0_Q0ldlq_big` | 0,11232 | 0,89107 | 6,886 | identique au témoin |
| `A0_Qperm_ldlq` (ordre de colonnes) | 0,12396 | 0,88708 | 6,935 | |
| `Ab2_Q0ldlq` | **3,00842** | 0,41508 | 60,4 | divergence en profondeur, cause non identifiée |
| `A0_Q0rtn` (RTN) | 0,19008 | 0,85305 | 6,759 | |
| `A0_Qmix_rtn` | 0,17984 | 0,86098 | 6,829 | |
| `Ab_Qmix_rtn` | 0,16169 | 0,87051 | 6,946 | |
| `A0_Qcol_rtn` | 0,21301 | 0,84851 | 6,761 | |

**Attention** : sur 72 documents la résolution statistique est de ±15 % du témoin (±5 % sur 600). Le rapport antérieur conclut sur 600 documents (`*_big`, `long_*`, `big_*`) : rotation **égalité** (ΔKL +0,0001, IC95 % [−0,0055 ; +0,0059], 306/600 documents améliorés) ; ordre des colonnes **égalité** (ΔKL +0,0035 [−0,0022 ; +0,0091]). Le gain apparent de `ArotN` sur 72 documents (−5,8 %) n'est donc **pas** confirmé sur 600. Les fichiers `big_*.json` et `long_*.json` n'ont **pas** été relus dans ces sessions.

### Sensibilité par famille (`diag_bykind_rules.json`, une seule famille quantifiée à ≈ 2,825 bpw, E8 sans rétroaction, 24 documents)

| Famille | Poids (G) | KL « lumen » | KL « col » | KL « fisher_local » | KL par G de poids |
|---|---:|---:|---:|---:|---:|
| `mlp_in` (gate, up) | 11,41 | 0,0377 | 0,0354 | 0,0626 | 0,0033 |
| `mlp_down` | 5,70 | 0,0259 | 0,0232 | 0,0232 | 0,0045 |
| `gdn_in` (qkv, z, a, b) | 4,05 | 0,0316 | 0,0271 | **1,874** | 0,0078 |
| `gdn_out` | 1,51 | 0,0180 | 0,0278 | 0,0161 | 0,0119 |
| `attn_in` (q, k, v) | 1,17 | 0,0240 | 0,0244 | 0,0385 | 0,0205 |
| `attn_out` | 0,50 | 0,0084 | 0,0090 | 0,0105 | 0,0169 |
| `embed` | 1,27 | 0,0058 | — | — | 0,0046 |
| `lm_head` | 1,27 | 0,0134 | 0,0182 | — | 0,0105 |

Somme des KL par famille 0,165 contre dommage conjoint 0,190 : les dommages s'additionnent à peu près. **Les poids d'attention complète sont ≈ 6 fois plus sensibles par poids que l'entrée des MLP.**

Autres résultats antérieurs relus : retrait d'une seule couche (`diag_layerdrop_single.json`, 24 documents) : couche 0 → KL 8,70 (vitale), couche 63 → 0,69, couche 13 → 0,027 (minimum), couche 62 → 0,17, médiane 0,047. **Aucune couche n'est gratuite.** Bilan des hypothèses H7–H12 du rapport antérieur : H7 biais ancré rejeté ; H8 pas Fisher rejeté ; H9 répartition des bits non concluante (prédit −16 %, mesuré +0,8 %) ; H10 échelles de pivots rejeté (+5,6 %) ; H11 rotation égalité ; H12 ordre de colonnes égalité.

## 7. Rétrodiction du modèle de bruit contre les mesures enregistrées

Scripts : `scripts/retro.py`, `scripts/retro2.py` (entrées : `outputs/colenergy.pt`, `stats_fisher.pt`, `diag_bykind_rules.json`). **Modèle** : erreur proxy d'un arrondi RTN avec échelles de colonne `c` : `(G·mult²/n)·(Σ_j C_j c_j²)·(Σ_j X_j / c_j²)`, `C_j` énergie de la colonne j, `X_j` second moment d'activation.

**Test 1** (`retro.py`) — rapport des KL col / lumen, mesuré contre prédit :

| Famille | KL col/lumen **mesuré** | modèle α=1 (pondéré, médiane) | gain demi-puissance γ (pondéré, médiane) |
|---|---:|---|---|
| `mlp_in` | 0,939 | 0,875 ; 0,879 | 0,774 ; 0,795 |
| `mlp_down` | 0,897 | 0,762 ; 0,936 | 0,545 ; 0,903 |
| `gdn_in` | 0,858 | 0,941 ; 1,451 | 0,542 ; 0,464 |
| `gdn_out` | **1,544** | 0,803 ; 0,901 | 0,536 ; 0,764 |
| `attn_in` | 1,015 | 1,127 ; 0,910 | 0,378 ; 0,380 |
| `attn_out` | 1,068 | 0,708 ; 0,914 | 0,595 ; 0,752 |
| `lm_head` | 1,362 | 1,519 ; 1,519 | 0,806 ; 0,806 |

**Test 2** (`retro2.py`) — poids de proxy `w_j = Var_j + λ·Mean_j²` (λ = 1 : second moment brut, comme dans LDLQ/F1 ; λ = 0 : variance centrée). Part de la moyenne² dans ΣX par famille : `mlp_in` 0,186 ; `mlp_down` 0,067 ; `gdn_in` **0,404** ; `gdn_out` 0,116 ; `attn_in` **0,538** ; `attn_out` 0,206 ; `lm_head` 0,113.

| λ | modèle/mesuré par famille (`mlp_in`, `mlp_down`, `gdn_in`, `gdn_out`, `attn_in`, `attn_out`, `lm_head`) | erreur log RMS |
|---|---|---:|
| **1,00** | 0,87/0,94 · 0,76/0,90 · 0,94/0,86 · 0,80/1,54 · 1,12/1,02 · 0,71/1,07 · 1,52/1,36 | **0,307** |
| 0,50 | 0,95/0,94 · 0,77/0,90 · 1,16/0,86 · 0,84/1,54 · 1,54/1,02 · 0,76/1,07 · 1,57/1,36 | 0,337 |
| 0,25 | 1,00/0,94 · 0,78/0,90 · 1,31/0,86 · 0,86/1,54 · 1,89/1,02 · 0,79/1,07 · 1,60/1,36 | 0,388 |
| 0,10 | 1,02/0,94 · 0,78/0,90 · 1,43/0,86 · 0,87/1,54 · 2,20/1,02 · 0,81/1,07 · 1,62/1,36 | 0,434 |
| 0,00 | 1,04/0,94 · 0,78/0,90 · 1,52/0,86 · 0,88/1,54 · 2,46/1,02 · 0,82/1,07 · 1,63/1,36 | 0,473 |

**Lecture** :
- Le second moment brut (λ = 1) est le meilleur poids de proxy (erreur log 0,307 contre 0,473 pour la variance centrée). Cela soutient l'idée que la moyenne² compte pour l'erreur effective.
- Le modèle simple **manque** `gdn_out` (prédit 0,80, mesuré 1,54), `attn_out` (0,71 contre 1,07), `mlp_down` (0,76 contre 0,90). Il surestime `lm_head` (1,52 contre 1,36). La prédiction n'est donc correcte qu'à ≈ ±35 % (erreur log 0,31). Ce n'est pas une validation.
- **Le test décisif — l'exposant effectif entre l'énergie d'activation d'une colonne et son poids réel en KL — n'a pas été achevé** (§8).
- Fait confirmé dans le code par la session 78f73527 : les anciens essais d'échelles utilisaient l'**exposant plein** (α = 1), cas dégénéré du modèle. Leur échec ne réfute pas un exposant intermédiaire. Le rejet de `col` (+12 % partout) et le gain limité aux mixeurs (`Qmix_rtn` −5 %, `Ab_Qmix_rtn` −15 %, **72 documents, avec incertitude de ±15 %**) vont dans le sens d'un effet concentré sur les entrées des mixeurs, donc sur la dimension 3994.

## 8. Ce qui a été interrompu : `beta.py`

`scripts/beta.py` doit ajuster, par famille, l'exposant β du poids de KL par colonne `ω_j ∝ X_j^β` à partir du test RTN α = 1 enregistré. Il prédit ensuite l'exposant optimal α* et le gain correspondant, selon `K(α) ∝ Σ_matrices (G·mult²/n)·(Σ_j C_j X_j^α)·(Σ_j X_j^(β−α))`, avec normalisation par la moyenne géométrique.

- Exécuté en premier plan avec un délai : **code de sortie 124 (dépassement)**. Aucun résultat.
- À relancer **en arrière-plan** (pas de limite de 120 s), ou à alléger (sous-échantillonnage des matrices, `torch.set_num_threads` élevé : la machine a 252 CPU, 944 Go de RAM).

## 9. Bibliographie vérifiée (détail dans `biblio/rapports_sources_primaires.md`)

Lecture par 4 agents web-fetch. **Les pages ont été tronquées vers 40 000 caractères** : les passages marqués « summary » dans les rapports ne sont pas vérifiés mot à mot.

| Source | Ce qui a été vérifié (verbatim) | Réserves |
|---|---|---|
| **Sun, Chen, Kolter, Liu**, *Massive Activations in LLMs*, arXiv 2402.17762v2 (14 août 2024), COLM 2024 | Définition : magnitude > 100 et ≈ 1000× la médiane (règle empirique). LLaMA2-7B : top-1 2622, médiane 0,2. **Table 3** : mise à la moyenne → perplexité inchangée (5,47 → 5,47 ; C4 7,85 → 7,86) ; mise à zéro → `inf`, accuracy 68,95 → 36,75 %. LLaMA2-13B : 4,88 → 5729 avec zéro. « Fixed but important biases ». Dimensions 1415 et 2533, couche 2, constantes jusqu'à la couche 30. Jetons concernés : jeton initial et premier délimiteur. | Le rôle de la normalisation (Sec. 4.2) est une hypothèse du papier. Rien vérifié sur la norme du résiduel. v1 non lue. |
| **He, Noci, Paliotta, Schlag, Hofmann**, arXiv 2405.19279 (NeurIPS 2024) | Définition des outlier features, kurtosis sur les normes d'activations par neurone, combinaison « Pre-Norm + Adam » prédisposée, bloc « Outlier Protected ». | Corps de l'article non lu. |
| **Elhage, Lasenby, Olah**, *Privileged Bases in the Transformer Residual Stream* (16 mars 2023) | « Les normaliseurs par dimension d'Adam sont à blâmer » ; LayerNorm et précision flottante écartés. Outlier = scalaire de valeur absolue > 6. | Chiffres de kurtosis non vérifiés. |
| **Chee, Cai, Kuleshov, De Sa**, *QuIP*, arXiv 2307.13304 (v2 du 15 janv. 2024) | Éq. (1) proxy `tr((Ŵ−W)H(Ŵ−W)ᵀ)` ; LDLQ, éq. (2)–(4) ; **Théorème 1** (optimalité de LDLQ parmi les arrondis à rétroaction linéaire fonction de H : `(m/4)tr(D)` pire cas, `(m/c)tr(D)` cas moyen, c = 12 arrondi au plus proche, 6 stochastique) ; **Lemme 2** `tr(D) ≤ (μ²/n)·tr(H^{1/2})²` pour H μ-incohérente ; Théorème 4 (sans incohérence, aucune borne spectrale ne sépare LDLQ de la base). | Venue NeurIPS 2023 d'après la bibliographie de QuIP#. Section 5.2 et annexes non vérifiées. |
| **Tseng, Chee, Sun, Kuleshov, De Sa**, *QuIP#*, arXiv 2402.04396v2 (4 juin 2024) | BlockLDLQ, **Théorème 4.1** : `E[tr((Ŵ−W)H(Ŵ−W)ᵀ)] ≤ (g·m·μ²·σ²/n)·tr(H^{1/2})²`. Codebook E8P : 2¹⁶ entrées, table de 2⁸ entrées (1 Kio), 8 bits d'index + 7 signes + 1 bit de décalage ±1/4 ; surcoût des signes < 0,01 bit/poids. Poids après RHT ≈ boule gaussienne. | Lemme B.1 (annexe B) non lu mot à mot. Venue « ICML » déduite des mots-clés. |
| **Nagel et al.**, *Data-Free Quantization Through Weight Equalization and Bias Correction*, arXiv 1906.04721 | Éq. (2) `f(sx) = s·f(x)` (équivariance à l'échelle positive) ; éq. (5)–(7) `Ŵ² = W²·S`, `Ŵ¹ = S⁻¹·W¹`, `b̂¹ = S⁻¹·b¹` ; **éq. (11)** `s_i = (1/r_i²)·√(r_i¹·r_i²)` ; éq. (9) objectif de précision par canal. | Version non indiquée par ar5iv. L'annexe (éq. 20–28) non vérifiée. Le texte ne dit pas que (11) « maximise » directement l'objectif (9). |
| **Xiao et al.**, *SmoothQuant*, arXiv 2211.10438 (ICML) | **Éq. (4)** `s_j = max|X_j|^α / max|W_j|^(1−α)` ; α = 0,5 pour OPT/BLOOM, 0,75 pour GLM-130B ; Table 6 (LLaMA) α = 0,8 ; Table 7 : Llama-2-7B/13B 0,85, 70B 0,9, Falcon-7B 0,6, Falcon-40B 0,7, Mistral-7B 0,8, Mixtral-8x7B 0,8. | Les cas α=0 et α=1 sont ma lecture de l'éq. (4), non étiquetés par le papier. La zone 0,4–0,6 est un résumé non vérifié. |
| **Lin et al.**, *AWQ*, arXiv 2306.00978 (MLSys 2024) | Éq. (5) `s = s_X^α`, α par recherche en grille sur [0,1] (grille de 20) ; analyse Δ'/Δ · 1/s ; Table 2 : Δ'/Δ = 1,005 / 1,013 / 1,038 / 1,213 pour s = 1,25 / 1,5 / 2 / 4, optimum à s = 2. | |
| **Agrell et Allen**, *On the best lattice quantizers*, arXiv 2202.09605 (IEEE TIT 69(12), déc. 2023) | NSM de E8 = **0,071682099** (≈ 929/12960) ; Z^n : 1/12 ; définition éq. (9). | **1/(2πe) n'est PAS dans ce papier** : ne pas le lui attribuer. Le chiffre « 0,0717 » vient de Zamir et Feder, non lu directement. |
| **Tseng, Sun, De Sa**, *Model-Preserving Adaptive Rounding* (YAQA), arXiv 2505.22988v3 | Éq. (1) objectif KL ; éq. (2) approximation quadratique par la hessienne de KL ; éq. (3) LDLQ ; éq. (5) `H̃ = H_O ⊗ H_I` ; Théorème 3.4. Réduction de KL ≈ 30 % sur LDLQ (version HTML) ; la page abstract dit « error » et non « KL ». | Version v3 révisée le 3 juin 2026, « ICML 2026 » (non vérifié ailleurs). Table 1 et sec. 4.3 non lues. |
| **Yang, Kautz, Hatamizadeh**, *Gated Delta Networks*, arXiv 2412.06464v3 (ICLR 2025) | **Éq. (10)** `S_t = S_{t−1}·(α_t(I − β_t·k_t·k_tᵀ)) + β_t·v_t·k_tᵀ`, α_t, β_t ∈ (0,1) ; sortie implicite `o_t = S_t·q_t` (sec. 2.1). | Pas d'équation de sortie numérotée. |

**Non lus** : SliceGPT (arXiv 2401.15024 : la recherche a été lancée, mais aucune page n'a été lue en entier ; la réponse de recherche n'a pas été retranscrite), Zamir et Feder, QERA, SRR, SpQR, OCS (cités dans les commentaires des jobs). **À lire avant de revendiquer une antériorité.**

## 10. Graphe réel : vérifications du code (jauges exactes)

Lues dans `modeling_qwen3_5.py` (transformers 5.22.2). **Les conclusions sur l'exactitude des jauges n'ont pas été tirées** : c'était l'objet du dernier point en cours. Les faits relevés :

- **Attention** (l. 763–810) : `q_proj` produit `[q, gate]` (chunk sur la dernière dimension, taille 2·head_dim) ; `q_norm` et `k_norm` (RMSNorm par tête, dimension 256) sont appliqués **après** `q_proj` et `k_proj`, avant RoPE ; `attn_output *= sigmoid(gate)` avant `o_proj`. `repeat_kv` (l. 699) : `expand` + `reshape`, sans calcul.
- **RMSNorm** (l. 828–846) : `(1 + w)` en facteur. Replier une échelle de colonne d'entrée dans `1 + w` suppose un facteur > 0 et bien conditionné (voir dimension 3994 : gain 0,0039).
- **MLP** (l. 822–826) : `down(act(gate(x)) · up(x))`. Une échelle sur les lignes de `up_proj` compensée par l'inverse sur les colonnes de `down_proj` est exacte en arithmétique réelle (produit élément par élément). Une échelle sur `gate_proj` ne l'est pas (SiLU non homogène).
- **GDN** (l. 560–650) : `b = in_proj_b(x)`, `a = in_proj_a(x)` ; convolution causale `conv1d` sur la sortie de `in_proj_qkv` ; `beta = sigmoid(b)` ; `g = −exp(A_log)·softplus(a + dt_bias)` ; `q` et `k` L2-normalisés dans le noyau (`use_qk_l2norm_in_kernel=True`) ; sortie `norm(core_out, z)` (RMSNorm gaté par `silu(z)`, `z` issue de `in_proj_z`) puis `out_proj`.
- **Hypothèses de jauge à vérifier, NON conclues** (lecture du code seulement ; à confirmer avant de les écrire dans le dossier) :
  - Les normes `q_norm`, `k_norm`, L2 de q/k dans le GDN et RMSNorm gaté normalisent **par tête** (sur 256, 128 canaux). Elles n'absorbent donc a priori qu'une échelle **uniforme par tête**. Une échelle **par canal** à l'intérieur d'une tête change le dénominateur de la norme : elle n'est a priori **pas** exacte.
  - `in_proj_a` et `in_proj_b` passent par `softplus` et `sigmoid` : a priori aucune jauge exacte.
  - L'effet de `conv1d` (depthwise, suivi d'une activation) sur une échelle par canal appliquée aux lignes de `in_proj_qkv` n'a pas été examiné.
- Les lignes 560–650 ont été filtrées par `grep` : le détail des lignes 536–560 et 636–650 n'est pas dans la trace. **À relire intégralement avant d'affirmer quoi que ce soit.**

## 11. Où reprendre — plan ordonné

**Déjà fait : ne pas refaire** — identité du checkpoint, lecture du mandat et des rapports, diagnostics §5, bibliographie §9, rétrodiction §7, lecture du code §10, relecture des variantes §6.

**À faire, dans l'ordre :**

1. **Relancer `beta.py`** en arrière-plan (§8) et conclure sur l'exposant effectif par famille.
2. **Terminer l'analyse du graphe** (§10) : tableau par famille de tenseurs, des jauges exactes (échelle par canal ou rotation) et de celles qui sont seulement approchées. Relire `modeling_qwen3_5.py` 536–650 en entier.
3. **Chiffrer les propositions candidates** à partir des statistiques disponibles (§5), sans nouveau forward. Les directions que les données orientent :
   - **Dimension 3994 comme biais explicite / jauge** : traiter la dimension massive à part (pas de 11–21× dans les écritures, X jusqu'à 10⁵, 80–93 % de moyenne² dans X aux mixeurs des premières couches). Discuter l'antériorité (Sun et al.) et ce que les essais `Ab_Qmix_rtn` / `Qmix_rtn` (−15 % sur 72 documents, ±15 %) ne couvraient pas. Vérifier que l'invariance à l'échelle n'est pas déjà obtenue par la symétrie E8 et le pas par ligne de F1.
   - **Égalisation d'échelle à exposant intermédiaire sur les entrées des mixeurs** (Nagel éq. 11, SmoothQuant α ≈ 0,5–0,85) : marges CS §5.5 (0,37–0,44 en médiane sur `q_proj` / `in_proj_qkv`, 0,72–0,80 sans la dimension dominante). Jauges à vérifier en §10 : les normes sont par tête, donc une échelle par canal n'est a priori pas exacte.
   - **Couches 60–63 (`out_proj` GDN AM/GM 2,3–2,8 ; `down_proj` L63 2,65)** et couches 0–3 des mixeurs : zones où les pivots LDL sont les plus dispersés.
   - **`lm_head` : mode commun 5,3 %**, mais le centrage simple est défavorable sous pondération G (×1,12). À ne présenter que sous réserve.
4. **Directions à écarter explicitement (avec raisons déjà mesurées)** : élagage de canaux SwiGLU (§5.1), repartage de gains de normalisation (§5.2), factorisation de rang bas et base commune entre couches (§5.8), partage MTP/couche (§5.6), rotations globales et ordre de colonnes (égalité sur 600 documents).
5. **Combler les inconnues** nécessaires au bilan taille : coût en octets des facteurs (FP16, selon `has_col_scale`, `16·n` bits par vecteur d'échelle de colonne et `16·m` pour le biais d'après `harness.py`), débit des tenseurs `shell`/`prod`, effet du codage entropique.
6. **Lire les sources non lues** (§9) avant toute affirmation d'antériorité.
7. **Rédiger `PROPOSITIONS_ARCHITECTURALES_FORMIC_F1.md`** selon la structure du §8 du mandat, puis **s'arrêter**.

**Pièges à ne pas oublier** :
- Ne **jamais** reprendre 0,133 pour le candidat à biais (0,11325).
- Ne pas sommer de KL ; ne pas conclure d'un gain de MSE local à un gain de KL.
- Le modèle de bruit n'est qu'approximativement validé (erreur log 0,307) ; il rate `gdn_out`.
- Les essais sur 72 documents ont une résolution de ±15 % : le gain apparent de `ArotN` (−5,8 %) ne tient pas sur 600 documents.
- Toute recommandation doit rester compatible avec **Lumen F1 inchangée** ; ce qui exige de la modifier est « hors mandat ».

## 12. Artefacts sauvegardés (`DOSSIER_REFLEXION_V8/05_artefacts_sessions/`)

| Chemin | Contenu |
|---|---|
| `outputs/colenergy.pt` (31 Mo) | Énergies de colonnes et de lignes de **498 matrices** du texte, de `lm_head` et `embed_tokens` (clés `col`, `row`), calcul CPU de 62,5 s |
| `outputs/colenergy.log`, `identity_check.txt` | Journal du calcul, résultat du hachage (`ALL_MATCH`) |
| `outputs/equalization_headroom_CS_factors.txt` | Facteurs CS par famille et par couche (§5.5) |
| `outputs/COMMANDES_ET_SORTIES_COMPLETES_session_78f73527.txt` | **Toutes** les commandes Bash de la première session (depuis 20:54) avec leur sortie intégrale |
| `outputs/COMMANDES_ET_SORTIES_COMPLETES_session_9445cdc8.txt` | Idem pour la seconde (depuis 23:28) |
| `scripts/colenergy.py`, `retro.py`, `retro2.py`, `beta.py` | Scripts des calculs de §5.5, §7, §8. Les `retro*.py` et `beta.py` lisent `colenergy.pt` **dans le répertoire courant** ; copier le fichier à côté ou adapter le chemin |
| `biblio/rapports_sources_primaires.md` | Rapports bruts des 4 agents de lecture (§9) |
| `transcripts/` | Extraction texte des deux sessions et **transcripts bruts `.jsonl`** (source de vérité) |

Transcripts d'origine : `/root/.claude/projects/-workspace/78f73527-….jsonl` et `…/9445cdc8-….jsonl`. Les scratchpads d'origine sont dans `/tmp/claude-0/-workspace/…` : `/tmp` peut être nettoyé, d'où la copie ci-dessus.
