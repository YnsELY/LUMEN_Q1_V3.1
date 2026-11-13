# FORMIC / LUMEN — Architecture × compression sur le vrai Qwen3.8-27B

**Sprint du 13 novembre 2026 — pod RunPod A100 80 Go, 2,11 $/h (tarif et solde renseignés par l'opérateur).**
Base de compression : LUMEN Q1-v3.1. Tout ce qui suit a été exécuté sur le BF16 local complet
(`/root/models/qwen3.8-27b`, 18 shards, 27 781 427 952 paramètres), jamais sur un modèle de substitution.

> Vocabulaire de preuve : **publié** (source primaire relue, `docs/SOURCES_PRIMAIRES.md`), **mesuré** (ce sprint, fichiers bruts
> dans `results/`), **estimé** (calcul à partir de mesures), **non testé**.

## Réponse en une page

**Qu'ai-je modifié ?** Douze hypothèses ont été testées sur le vrai modèle, dont cinq modifications du graphe exécutées de bout en bout :
suppression de couches, rétrécissement des MLP avec réajustement, ajout d'un biais par linéaire (deux variantes), rotation du flux résiduel
avec gains de normalisation repliés (deux mises en œuvre). **Aucune n'est retenue.**

**Pourquoi cela devait s'accorder avec Lumen ?** Les candidats visaient ce que le codec E8 à pas par ligne ne voit pas : la composante
constante « massive » des activations (biais), l'anisotropie des pivots de LDLQ (changements de base), et la redondance supposée de la
structure (profondeur, largeur, rang, tête). Les diagnostics montrent que le codec n'a presque rien à y gagner : les poids sont déjà quasi
gaussiens à échelles de ligne uniformes, LDLQ annule de lui-même l'erreur sur la direction massive, et ce qui reste après LDLQ
(0,2 à 1,4 % d'erreur pondérée par matrice) est sans structure exploitable.

**Quel gain ou quelle perte face au contrôle équitable ?** Le contrôle est l'architecture d'origine quantifiée par mon portage GPU de
Lumen v3.1 (E8 + LDLQ) à 2,821 bpw idéal sur le chemin texte : **KL = 0,112 nat, accord top-1 89,1 %** (0,106 et 89,7 % sur 600 documents),
stable jusqu'à 8192 tokens de contexte.

| Candidat | Face au contrôle, à octets égaux | Statut |
|---|---|---|
| 62 couches au lieu de 64 | +33 % de KL (sans rétroaction) ; dommage BF16 seul 0,108 | perte, confirmée |
| MLP −12,5 % / −25 % | +33 % / +87 % (sans rétroaction) ; dommage BF16 seul 0,136 / 0,278 | perte, confirmée |
| Tête dérivée de l'embedding, factorisation, bases partagées, correction de faible rang | écartées sur diagnostic chiffré | non construites |
| Biais FP16 par linéaire | −15 à −28 % sans rétroaction ; **égalité avec LDLQ** | neutre |
| Échelles de colonne (pivots LDL) | +5,6 % / +0,1 % avec LDLQ | neutre à négatif |
| Rotation du résiduel | +0,1 % sur 600 documents (IC [-5,2 ; +5,5] %) | égalité |
| Ordre des colonnes de LDLQ | +3,3 % sur 600 documents (IC [-2,1 ; +8,5] %) | égalité |

**Conclusion.** Sur ce modèle et à ce taux, je n'ai pas trouvé d'architecture ni de représentation qui préserve mieux le BF16 que la
structure d'origine quantifiée par Lumen avec LDLQ. C'est une conclusion négative, limitée à ce qui a été mesuré : fidélité des logits en
texte, modifications sans entraînement.

**Que tester ensuite ?** (détail en section 10) Les leviers restants ne sont plus dans la forme des matrices : (1) une récupération par
entraînement, appliquée à égalité au témoin et aux candidats ; (2) la répartition du budget entre texte et Vision, première mesure à l'appui ;
(3) l'évaluation sur tâches, MTP et génération, sans lesquelles aucune décision produit n'est fondée.

## 1. Ce qui a été réellement exécuté

| Élément | Statut |
|---|---|
| Chargement HF du BF16 complet (`transformers 5.22.2`, `Qwen3_5ForConditionalGeneration`), 52,7 Go VRAM, premier forward | **mesuré** : 13 s de chargement, 3,5 s pour 8×1024 tokens |
| Identité du checkpoint | en-têtes des 18 shards cohérents avec l'index (1199 tenseurs) ; **la révision `37561816…` n'est attestée par aucun fichier local** (pas de dossier `snapshots/`, pas de `_commit_hash`) : provenance déclarée par l'opérateur, non certifiée |
| Portage GPU de `lumen_ref` (E8, candidats, LDLQ bloc-8, coût du codec mode 0) | **mesuré** : codes identiques à la référence CPU sur 16 configurations LDLQ et 24 000 vecteurs E8 (`tests/test_gpu_port.log`) |
| Octets | coût idéal du codec de référence ; encodage/décodage **physique** par le codec CPU sur 7 extraits de vrais poids : aller-retour exact, surcoût 0,005 bpw hors pas et en-tête (`results/codec_real_weights.json`) |
| Activations | hooks sur le vrai forward, entrées réelles de chaque linéaire ; passe séquentielle vérifiée **bit à bit** contre le forward du modèle (`jobs/030b_seq_parity.py`) |
| Données | 144 / 72 / 72 documents × 1024 tokens (calibration / sélection / confirmation), 6 domaines, sources disjointes ; 12 documents × 8192 tokens ; 24 invites image+texte |
| Métrique | KL(professeur BF16 ‖ élève) exacte sur les 248 320 logits, à toutes les positions ; accord top-1 ; entropie croisée sur le vrai token suivant |

**Budget commun à toutes les variantes quantifiées : 2,824 bpw idéal × 26,893 G poids du chemin texte** (pas FP16, échelles, biais et
en-têtes compris). Origine : budget du paquet complet calculé sur les en-têtes réels (`results/budget_package.json`) : plafond
10 000 000 000 octets − tenseurs protégés FP16 (6,2 Mo) − auxiliaires hérités de Lumen (84,1 Mo, hypothèse) − garde (18,5 Mo)
= 2,849 bpw physiques pour les 27,778 G poids quantifiables, **Vision et MTP inclus au même taux**, soit 2,840 bpw idéal après le
surcoût de codec retenu (0,009 bpw ; 0,005 mesuré). J'ai pris 2,824, sous la capacité déclarée par Lumen (2,8328).
Le « 2,821 » du témoin est donc un **coût idéal de codec, tout compris pour le chemin texte** (symboles + rang + échappements + pas FP16 + en-têtes) ;
ce n'est ni une taille de fichier, ni le coût du paquet complet. **Aucun fichier de 10 Go n'a été produit.**

## 2. Hypothèses examinées et verdicts

Témoins : `A0_Q0rtn` = architecture d'origine, E8 au plus proche, pas Lumen (multiplicateur × RMS de ligne), **sans** rétroaction ;
`A0_Q0ldlq` = idem **avec** la passe E8-LDLQ séquentielle complète (algèbre de la référence v3.1, `rate_lambda = 0`, amortissement 1 %).
Le second est le contrôle équitable ; le premier ne sert qu'au criblage rapide.

| # | Hypothèse | Type | Expérience | Verdict |
|---|---|---|---|---|
| H1 | Retirer des couches, réinvestir les octets | reconstruction | 1 couche (64 essais), 2/4/8 couches ; version quantifiée à octets égaux | **rejetée (mesuré)** |
| H2 | Rétrécir les MLP SwiGLU (61,6 % des poids) avec réajustement, réinvestir | reconstruction | −12,5 % et −25 % de canaux, saillance de Taylor + moindres carrés séquentiels ; versions quantifiées à octets égaux | **rejetée (mesuré)** |
| H3 | Dériver `lm_head` de `embed_tokens` | reconstruction | régression linéaire tête ← embedding | **rejetée (mesuré)** : R² = 0,17 |
| H4 | Factoriser puis quantifier ; bases partagées entre couches | reconstruction | perte de troncature au rang finançable à octets égaux, pondérée par les activations, comparée à l'erreur LDLQ réelle ; rang conjoint de couches voisines | **rejetée sur diagnostic (mesuré)** : la troncature seule vaut 2 à 9 fois l'erreur de LDLQ, avant même de quantifier les facteurs |
| H5 | Correction ou protection de faible rang ; exceptions précises ; scission de canaux | ajout | spectre et concentration de l'erreur pondérée **après** le vrai LDLQ | **rejetée sur diagnostic (mesuré)** : erreur quasi plate, les mêmes octets en précision E8 retirent 3 à 10 fois plus |
| H6 | Biais FP16 par linéaire portant l'erreur sur la moyenne des activations (dimensions « massives ») | ajout exact | A_b, sans rétroaction puis avec LDLQ ; robustesse hors domaine | **gain sans rétroaction (−15 %), nul avec LDLQ (mesuré)** |
| H7 | Biais ancré sur la sortie moyenne du professeur | ajout | A_b2, passe LDLQ | **rejetée (mesuré)** : KL 3,0 ; cause de la divergence en profondeur non identifiée |
| H8 | Pas par ligne / échelles par colonne tirés d'un Fisher diagonal | recette | 3 règles sur le modèle entier + diagnostic par famille | **rejetée (mesuré)** : nulle à catastrophique |
| H9 | Répartition des bits entre familles selon la sensibilité mesurée | recette | décalages par famille à octets égaux | **non concluante (mesuré)** : prédiction −16 %, mesure +0,8 % |
| H10 | Changement de base diagonal des entrées qui aplatit les pivots de LDLQ | transformation exacte | passe LDLQ complète, chaque matrice au débit du témoin ; contrôle local | **rejetée (mesuré)** : +5,6 % de KL ; l'erreur locale ne baisse pas non plus |
| H11 | Rotation cohérente du flux résiduel, gains de normalisation repliés | transformation exacte | passe LDLQ complète dans la base tournée (émulée, puis native), vérification d'équivalence, comparaison sur 600 documents | **égalité (mesuré, 600 documents)** : ΔKL +0,0001, IC 95 % [-0,0055 ; +0,0059] (+0,1 %), 306 documents améliorés sur 600 ; résidu d'équivalence avant quantification 0,003–0,009 nat |
| H12 | Ordre de traitement des colonnes dans LDLQ (second moment décroissant) | transformation exacte (permutation) | diagnostic local sur 5 couches ; passe complète, comparaison sur 600 documents | **égalité (mesuré, 600 documents)** : ΔKL +0,0035, IC 95 % [-0,0022 ; +0,0091] (+3,3 %), 239 documents améliorés sur 600 ; gain local 2–5 % |

## 3. Diagnostics du vrai graphe (BF16, mesurés)

Graphe détaillé : `docs/GRAPHE_AVANT_APRES.md`.

**Aucune couche n'est gratuite.** Retirer une seule couche coûte entre 0,027 nat (couche 13) et 0,17 nat (couche 62), médiane 0,047 ;
la couche 0 est vitale (8,7 nats), la 63 aussi (0,69). Les couches d'attention complète ne sont pas moins chères que les couches GDN.
(`results/diag_layerdrop_single.json`, 24 documents de sélection.)

**Sensibilité par famille** (une seule famille quantifiée à 2,825 bpw, E8 sans rétroaction, le reste en BF16 ; `results/diag_bykind_rules.json`) :

| Famille | Poids (G) | KL seule | KL par G de poids |
|---|---:|---:|---:|
| MLP entrée (`gate`, `up`) | 11,41 | 0,0377 | 0,0033 |
| MLP sortie (`down`) | 5,70 | 0,0259 | 0,0045 |
| GDN entrée (`qkv`, `z`, `a`, `b`) | 4,05 | 0,0316 | 0,0078 |
| GDN sortie (`out_proj`) | 1,51 | 0,0180 | 0,0119 |
| Attention entrée (`q`, `k`, `v`) | 1,17 | 0,0240 | 0,0205 |
| Attention sortie (`o_proj`) | 0,50 | 0,0084 | 0,0169 |
| `embed_tokens` | 1,27 | 0,0058 | 0,0046 |
| `lm_head` | 1,27 | 0,0134 | 0,0105 |

Les poids d'attention complète sont 6 fois plus sensibles par poids que l'entrée des MLP. La somme (0,165) est proche du dommage
conjoint (0,190) : à ce taux les dommages s'additionnent à peu près.

**Les poids sont déjà presque idéaux pour E8 à pas par ligne** : kurtosis par ligne 3,2–3,6, coefficient de variation des RMS de ligne
0,08–0,21 ; erreur relative de Frobenius uniforme de 15,85 % à 2,82 bpw sur toutes les familles (arrondi sans rétroaction).

**Composante constante « massive » des activations.** 36 à 60 % du second moment des entrées des lecteurs du flux résiduel tient dans
le carré de la moyenne, concentré sur ~8 dimensions (rapport max/médiane jusqu'à 3,9·10⁴ ; phénomène **publié** par Sun et al. 2024,
ici **mesuré**). Aucune position ne domine (part de la position 0 < 1 %). Fait nouveau relevé en préparant la rotation :
**une dimension du résiduel a un gain de RMSNorm de 1/256 à l'entrée des MLP des couches 0 à 10, et exactement 0 à la couche 7** —
le modèle masque lui-même cette dimension à ses MLP.

**Ce qui reste après le vrai LDLQ est petit et sans structure** (`results/diag_error_structure.json`, 6 couches, entrées réelles) :

| Famille | Erreur relative pondérée par les activations après LDLQ | Part de l'erreur dans ses 64 premières directions | Rang 64 en FP16 : coût, et part d'erreur que les mêmes bits retireraient en précision E8 | Factorisation à 4 bits à octets égaux : rang, perte de signal |
|---|---:|---:|---:|---:|
| Entrée des mélangeurs | 0,2–0,6 % | 5–17 % | 0,28–0,30 bpw → 33–34 % | 2408–2549 / 5120 : 0,5–2,3 % |
| Sortie des mélangeurs | 0,2–1,4 % | 6–12 % | 0,37 bpw → 40 % | 1966–1970 / 5120 : 0,8–5,9 % |
| MLP `gate` | 0,2–0,9 % | 3–5 % | 0,26 bpw → 30 % | 2790–2796 / 5120 : 1,3–3,9 % |
| MLP `down` | 0,6–1,4 % | 3–6 % | 0,26 bpw → 30 % | 2787–2795 / 5120 : 5,2–9,7 % |

Lecture : une correction de rang 64 retirerait au mieux 3 à 17 % de l'erreur, quand les mêmes octets dépensés en précision E8 en
retirent 30 à 40 % ; la troncature seule d'une factorisation finançable perd 2 à 9 fois l'erreur totale de LDLQ sur la même matrice. Le rang effectif conjoint de
deux couches voisines (≈ 4000–4260 sur 5120) est proche de l'espace entier : pas de base commune à partager. Les 0,1 % d'entrées les
plus fautives ne portent que 1,2 à 5,7 % de l'erreur quadratique : pas de valeurs extrêmes à protéger.

**Marge d'un changement de base des entrées pour LDLQ** (moyenne arithmétique / géométrique des pivots LDL ; `results/diag_ldl_headroom.json`) :
0,03–0,10 bit sur les entrées de MLP, 0,05–0,29 bit sur leurs sorties (0,42–0,70 aux couches 62–63), 0,07–0,51 bit sur les entrées des
mélangeurs, 0,15–0,65 bit sur leurs sorties. Une rotation orthogonale aléatoire ramène ce rapport à 1,02–1,08 sur les lecteurs du résiduel.
C'est une marge sur le critère local, pas un gain de KL (voir H10, H11).

**Ordre des colonnes** (`results/diag_actorder.json`) : traiter les colonnes par second moment décroissant ramène le rapport des pivots à
1,00–1,23 et réduit l'erreur pondérée réelle de 2 à 5 % à bits voisins (19 % sur l'entrée du mélangeur de la couche 3) ; l'ordre inverse
la multiplie par 1,1 à 3,5. L'ordre natif est donc déjà proche du bon.

## 4. Résultats sur la sélection (72 documents, 73 728 positions)

ΔKL : différence appariée par document, IC 95 % par bootstrap sur les documents. « octets/cible » : écart du coût idéal total au budget commun.

| Variante | Identifiant | Octets / cible | KL | Accord top-1 | ΔKL vs témoin [IC 95 %] | chat | code | prose | math | multil. | science |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **témoin LDLQ** : architecture d'origine, E8-LDLQ | `A0_Q0ldlq` | -0,10 % | 0,1123 | 89,1 % |  | 0,244 | 0,202 | 0,078 | 0,042 | 0,067 | 0,041 |
| rotation du résiduel, native, E8-LDLQ | `ArotN_Q0ldlq` | -0,14 % | 0,1058 | 89,8 % | -0,0065 [-0,0216 ; +0,0059] (-5,8 %) | 0,195 | 0,209 | 0,078 | 0,044 | 0,068 | 0,040 |
| rotation du résiduel, émulée, E8-LDLQ | `Arot_Q0ldlq` | -0,14 % | 0,1213 | 89,0 % | +0,0090 [-0,0067 ; +0,0280] (+8,0 %) | 0,294 | 0,206 | 0,079 | 0,038 | 0,069 | 0,042 |
| ordre des colonnes (second moment décroissant), E8-LDLQ | `A0_Qperm_ldlq` | -0,29 % | 0,1240 | 88,7 % | +0,0116 [-0,0040 ; +0,0338] (+10,4 %) | 0,291 | 0,212 | 0,084 | 0,043 | 0,071 | 0,042 |
| biais FP16 + Gram centré, E8-LDLQ | `Ab_Q0ldlq` | -0,10 % | 0,1132 | 89,3 % | +0,0009 [-0,0157 ; +0,0181] (+0,8 %) | 0,242 | 0,205 | 0,079 | 0,044 | 0,069 | 0,041 |
| échelles de colonne (pivots LDL), E8-LDLQ | `A0_Qldlcol_ldlq` | -0,29 % | 0,1186 | 89,1 % | +0,0063 [-0,0083 ; +0,0181] (+5,6 %) | 0,229 | 0,228 | 0,088 | 0,047 | 0,077 | 0,043 |
| biais ancré professeur, E8-LDLQ | `Ab2_Q0ldlq` | +0,07 % | 3,0084 | 41,5 % | +2,8961 [+2,6618 ; +3,1466] (+2578,4 %) | 4,505 | 3,052 | 2,529 | 2,169 | 3,688 | 2,108 |
| **témoin sans rétroaction** : E8 au plus proche | `A0_Q0rtn` | +0,03 % | 0,1901 | 85,3 % |  | 0,408 | 0,288 | 0,122 | 0,101 | 0,115 | 0,106 |
| biais FP16, sans rétroaction | `Ab_Q0rtn` | +0,01 % | 0,1624 | 86,9 % | -0,0277 [-0,0453 ; -0,0015] (-14,6 %) | 0,385 | 0,244 | 0,096 | 0,081 | 0,094 | 0,074 |
| biais FP16, moyennes sans code ni chat | `Ab_Q0rtn_mu4dom` | +0,01 % | 0,1647 | 87,0 % | -0,0254 [-0,0438 ; +0,0028] (-13,3 %) | 0,396 | 0,247 | 0,096 | 0,082 | 0,094 | 0,073 |
| échelles de colonne sur entrées des mélangeurs | `A0_Qmix_rtn` | +0,10 % | 0,1798 | 86,1 % | -0,0102 [-0,0271 ; +0,0103] (-5,4 %) | 0,397 | 0,279 | 0,117 | 0,085 | 0,105 | 0,095 |
| biais + échelles de colonne (mélangeurs) | `Ab_Qmix_rtn` | +0,08 % | 0,1617 | 87,1 % | -0,0284 [-0,0450 ; -0,0080] (-14,9 %) | 0,346 | 0,263 | 0,104 | 0,076 | 0,100 | 0,081 |
| bits répartis par famille | `A0_Qalloc_rtn` | +0,02 % | 0,1916 | 85,8 % | +0,0015 [-0,0155 ; +0,0268] (+0,8 %) | 0,469 | 0,283 | 0,121 | 0,080 | 0,110 | 0,087 |
| échelles de colonne partout | `A0_Qcol_rtn` | +0,05 % | 0,2130 | 84,9 % | +0,0229 [-0,0036 ; +0,0646] (+12,1 %) | 0,488 | 0,312 | 0,143 | 0,090 | 0,145 | 0,100 |
| pas Fisher global | `A0_Qfg_rtn` | +0,34 % | 1,4266 | 60,2 % | +1,2365 [+1,0264 ; +1,4626] (+650,5 %) | 1,928 | 1,176 | 1,162 | 0,428 | 3,123 | 0,743 |
| pas Fisher local | `A0_Qfl_rtn` | +0,10 % | 7,6217 | 8,9 % | +7,4317 [+7,0868 ; +7,7890] (+3909,8 %) | 6,308 | 8,665 | 7,108 | 7,043 | 9,648 | 6,958 |
| 62 couches, octets réinvestis (2,907 bpw) | `A2_drop2_Q0rtn` | -0,01 % | 0,2527 | 84,1 % | +0,0626 [+0,0309 ; +0,1041] (+32,9 %) | 0,669 | 0,336 | 0,143 | 0,115 | 0,129 | 0,124 |
| MLP −12,5 %, octets réinvestis (3,068 bpw) | `A1_w0.125_Q0rtn` | -0,01 % | 0,2525 | 83,0 % | +0,0624 [+0,0461 ; +0,0815] (+32,8 %) | 0,508 | 0,348 | 0,223 | 0,107 | 0,193 | 0,137 |
| MLP −25 %, octets réinvestis (3,358 bpw) | `A1_w0.25_Q0rtn` | +0,00 % | 0,3553 | 80,1 % | +0,1652 [+0,1348 ; +0,2000] (+86,9 %) | 0,676 | 0,434 | 0,352 | 0,129 | 0,346 | 0,196 |
| rotation seule, BF16 (résidu d'équivalence) | `Arot_bf16` | — | 0,0030 | 98,8 % |  | 0,008 | 0,001 | 0,001 | 0,006 | 0,001 | 0,001 |
| 62 couches, BF16 | `A2_drop2_bf16` | — | 0,1082 | 90,8 % |  | 0,323 | 0,160 | 0,036 | 0,065 | 0,028 | 0,036 |
| 60 couches, BF16 | `A2_drop4_bf16` | — | 0,2378 | 85,9 % |  | 0,810 | 0,262 | 0,085 | 0,113 | 0,072 | 0,085 |
| 56 couches, BF16 | `A2_drop8_bf16` | — | 0,3657 | 81,6 % |  | 1,047 | 0,440 | 0,181 | 0,196 | 0,160 | 0,170 |
| MLP −12,5 % + moindres carrés, BF16 | `A1_w0.125_bf16` | — | 0,1359 | 88,5 % |  | 0,234 | 0,213 | 0,140 | 0,049 | 0,113 | 0,065 |
| MLP −12,5 % sans réajustement, BF16 | `A1_w0.125_bf16_nocomp` | — | 0,1586 | 87,7 % |  | 0,319 | 0,213 | 0,139 | 0,085 | 0,118 | 0,077 |
| MLP −25 % + moindres carrés, BF16 | `A1_w0.25_bf16` | — | 0,2784 | 83,1 % |  | 0,477 | 0,370 | 0,297 | 0,093 | 0,284 | 0,150 |

Lecture :

1. **Le témoin équitable est fort.** La passe E8-LDLQ ramène la KL de 0,190 à 0,112 (−41 %) à octets égaux. Tout candidat doit battre ce
   témoin-là, pas l'arrondi simple.
2. **Les reconstructions perdent nettement.** En BF16, sans rien quantifier, retirer 2 couches coûte déjà 0,108 et retirer 12,5 % des canaux
   de MLP 0,136 : autant ou plus que tout le dommage de quantification du témoin (0,112). À octets égaux, les bits réinvestis ne rattrapent
   pas : +33 % pour 62 couches à 2,907 bpw, +33 % pour −12,5 % de largeur à 3,068 bpw, +87 % pour −25 % à 3,358 bpw. Le réajustement
   par moindres carrés aide (0,159 → 0,136) sans changer le verdict. La saillance de Taylor attribuait ~1 % d'importance aux 12,5 % de canaux
   retirés : elle sous-estime massivement l'effet réel.
3. **Le biais (A_b) corrige un vrai défaut de l'arrondi simple, que LDLQ corrige déjà.** Sans rétroaction : −15 % en sélection, −28 % en
   confirmation, et le gain subsiste quand les moyennes sont calculées sans les domaines code et chat. Avec LDLQ : égalité
   (+0,8 % en sélection, −6,7 % en confirmation, aucun intervalle n'excluant zéro). La rétroaction de LDLQ est pilotée par un Gram dont la
   composante dominante est précisément la direction moyenne : elle annule cette erreur sans biais.
4. **Les règles tirées du critère local échouent en KL.** Pas guidés par Fisher : ×7,5 à ×40, par destruction des projections d'entrée GDN
   (KL 1,87 avec cette seule famille modifiée) — une sensibilité locale ne prédit pas l'effet d'une grande perturbation sur des unités
   saturées ou normalisées. Échelles de colonne par second moment : +12 % partout, −5 % si on les limite aux entrées des mélangeurs.
   Répartition des bits par famille : prédite à −16 %, mesurée à +0,8 %. Échelles par pivots LDL sous LDLQ : +5,6 % ; même l'erreur
   locale ne baisse pas à bits égaux, la rétroaction réinjectant dans les colonnes grossières une erreur amplifiée.
5. **La rotation du résiduel ne se distingue pas du témoin.** Sur 72 documents elle affichait −5,8 % (native) ou +8,0 % (émulée) selon
   l'implémentation, l'écart venant de quelques documents « chat » : c'est la mesure du bruit de ce jeu, pas un effet. Voir la section 5 bis.
6. **Le biais ancré sur le professeur diverge** (KL 3,0 sur 64 couches, alors qu'il se comporte normalement sur les 4 premières : 0,0075
   contre 0,0059 pour A_b). Un bug de statistique sur la tête a été écarté ; la cause de la divergence en profondeur n'est pas identifiée.
   Ce résultat réfute cette mise en œuvre, pas toute calibration asymétrique.

## 5. Confirmation (72 documents réservés, lus après gel)

| Variante | Identifiant | Octets / cible | KL | Accord top-1 | ΔKL vs témoin [IC 95 %] | chat | code | prose | math | multil. | science |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **témoin LDLQ** : architecture d'origine, E8-LDLQ | `A0_Q0ldlq` | -0,10 % | 0,1185 | 89,4 % |  | 0,251 | 0,127 | 0,078 | 0,129 | 0,082 | 0,044 |
| rotation du résiduel, native, E8-LDLQ | `ArotN_Q0ldlq` | -0,14 % | 0,1070 | 89,9 % | -0,0115 [-0,0261 ; +0,0026] (-9,7 %) | 0,194 | 0,126 | 0,076 | 0,125 | 0,078 | 0,044 |
| rotation du résiduel, émulée, E8-LDLQ | `Arot_Q0ldlq` | -0,14 % | 0,1241 | 89,1 % | +0,0056 [-0,0142 ; +0,0274] (+4,7 %) | 0,295 | 0,124 | 0,076 | 0,123 | 0,080 | 0,047 |
| biais FP16 + Gram centré, E8-LDLQ | `Ab_Q0ldlq` | -0,10 % | 0,1106 | 89,9 % | -0,0079 [-0,0189 ; +0,0027] (-6,7 %) | 0,195 | 0,124 | 0,077 | 0,142 | 0,080 | 0,045 |
| échelles de colonne (pivots LDL), E8-LDLQ | `A0_Qldlcol_ldlq` | -0,29 % | 0,1186 | 89,5 % | +0,0001 [-0,0130 ; +0,0122] (+0,1 %) | 0,221 | 0,141 | 0,084 | 0,126 | 0,088 | 0,050 |
| **témoin sans rétroaction** : E8 au plus proche | `A0_Q0rtn` | +0,03 % | 0,2123 | 84,9 % |  | 0,530 | 0,183 | 0,120 | 0,205 | 0,129 | 0,108 |
| biais FP16, sans rétroaction | `Ab_Q0rtn` | +0,01 % | 0,1531 | 87,5 % | -0,0592 [-0,0925 ; -0,0330] (-27,9 %) | 0,335 | 0,156 | 0,091 | 0,155 | 0,106 | 0,076 |
| biais FP16, moyennes sans code ni chat | `Ab_Q0rtn_mu4dom` | +0,01 % | 0,1566 | 87,5 % | -0,0557 [-0,0885 ; -0,0294] (-26,2 %) | 0,350 | 0,158 | 0,090 | 0,159 | 0,106 | 0,075 |
| échelles de colonne sur entrées des mélangeurs | `A0_Qmix_rtn` | +0,10 % | 0,1802 | 86,1 % | -0,0321 [-0,0628 ; -0,0080] (-15,1 %) | 0,389 | 0,178 | 0,107 | 0,197 | 0,116 | 0,094 |
| 62 couches, octets réinvestis (2,907 bpw) | `A2_drop2_Q0rtn` | -0,01 % | 0,2715 | 84,0 % | +0,0592 [+0,0185 ; +0,1138] (+27,9 %) | 0,759 | 0,221 | 0,134 | 0,244 | 0,147 | 0,124 |
| MLP −12,5 %, octets réinvestis (3,068 bpw) | `A1_w0.125_Q0rtn` | -0,01 % | 0,2622 | 83,0 % | +0,0499 [+0,0209 ; +0,0737] (+23,5 %) | 0,529 | 0,228 | 0,233 | 0,211 | 0,233 | 0,139 |
| rotation seule, BF16 (résidu d'équivalence) | `Arot_bf16` | — | 0,0086 | 98,9 % |  | 0,005 | 0,001 | 0,001 | 0,044 | 0,001 | 0,001 |
| 62 couches, BF16 | `A2_drop2_bf16` | — | 0,1237 | 91,0 % |  | 0,382 | 0,091 | 0,036 | 0,167 | 0,032 | 0,034 |
| MLP −12,5 % + moindres carrés, BF16 | `A1_w0.125_bf16` | — | 0,1442 | 88,4 % |  | 0,223 | 0,132 | 0,155 | 0,137 | 0,149 | 0,068 |

Les finalistes ont été figés par le commit `e533e72` (`results/selection.lock.json`) avant toute lecture des fichiers scellés.
**Les conclusions de la sélection tiennent** : les deux reconstructions à octets égaux restent significativement moins bonnes que leur témoin
(+23,5 % et +27,9 %) ; le biais sans rétroaction reste significativement meilleur (−27,9 %, et −26,2 % avec des moyennes calculées hors
code et chat) ; avec LDLQ, aucun candidat ne se distingue du témoin de façon significative. Le résidu d'équivalence de la rotation
seule monte à 0,0086 en confirmation, dont 0,044 sur le domaine math : la transformation n'est pas exacte en stockage BF16.

## 5 bis. Comparaison déclarée à l'avance sur 600 documents

Les jeux de 72 documents ne distinguent pas des écarts inférieurs à ~15 % du témoin. Pour trancher les deux transformations exactes encore
ouvertes, trois variantes ont été déclarées **avant** exécution (`results/big_comparison.lock.json`) puis recréées par une passe E8-LDLQ complète
et évaluées une seule fois sur 600 nouveaux documents (100 par domaine, sources disjointes de tous les autres jeux).
La ré-exécution du témoin reproduit le premier passage à l'identique (KL de sélection 0,11232 dans les deux cas).

| Variante | Octets / cible | KL | Accord top-1 | ΔKL vs témoin [IC 95 %] | docs améliorés | chat | code | en_pro | math_r | multil | scienc |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `A0_Q0ldlq_big` | -0,10 % | 0,1064 | 89,7 % |  |  | 0,224 | 0,166 | 0,060 | 0,067 | 0,076 | 0,045 |
| `ArotN_Q0ldlq_big` | -0,14 % | 0,1064 | 89,5 % | +0,0001 [-0,0055 ; +0,0059] (+0,1 %) | 306 / 600 | 0,229 | 0,161 | 0,060 | 0,068 | 0,076 | 0,045 |
| `A0_Qperm_ldlq` | -0,29 % | 0,1098 | 89,3 % | +0,0035 [-0,0022 ; +0,0091] (+3,3 %) | 239 / 600 | 0,235 | 0,168 | 0,063 | 0,068 | 0,079 | 0,047 |

* **Rotation du résiduel** : égalité (ΔKL +0,0001, IC 95 % [-0,0055 ; +0,0059] (+0,1 %), 306 documents améliorés sur 600).
* **Ordre des colonnes** : égalité (ΔKL +0,0035, IC 95 % [-0,0022 ; +0,0091] (+3,3 %), 239 documents améliorés sur 600). La variante a reçu 0,19 % d'octets de moins que le témoin (−0,0053 bpw, soit
  environ +0,7 % de KL attendu) : même corrigé de cet écart, le gain local de 2–5 % mesuré sur l'erreur pondérée ne se retrouve pas en KL.

Avec 600 documents, l'intervalle de confiance d'une différence est de ±5 % du témoin. Les deux transformations sont donc **sans gain
supérieur à ~5 %** ; un effet plus petit n'est pas exclu, mais ne justifierait pas leur intégration.


## 6. Contexte long, récurrence, Vision

| Variante | KL moyenne | par tranche de 1024 tokens (0–8192) | Accord top-1 |
|---|---:|---|---:|
| `A0_Q0ldlq` | 0,1070 | 0,106 · 0,106 · 0,090 · 0,114 · 0,110 · 0,101 · 0,100 · 0,130 | 90,6 % |
| `ArotN_Q0ldlq` | 0,1078 | 0,106 · 0,103 · 0,091 · 0,113 · 0,109 · 0,102 · 0,105 · 0,132 | 90,5 % |
| `Arot_Q0ldlq` | 0,1086 | 0,108 · 0,111 · 0,089 · 0,118 · 0,104 · 0,098 · 0,100 · 0,141 | 90,6 % |
| `Ab_Q0ldlq` | 0,1127 | 0,109 · 0,109 · 0,092 · 0,117 · 0,117 · 0,103 · 0,104 · 0,150 | 90,4 % |
| `A0_Qldlcol_ldlq` | 0,1225 | 0,115 · 0,121 · 0,103 · 0,124 · 0,135 · 0,116 · 0,119 · 0,149 | 89,9 % |
| `A0_Q0ldlq_big` | 0,1070 | 0,106 · 0,106 · 0,090 · 0,114 · 0,110 · 0,101 · 0,100 · 0,130 | 90,6 % |
| `ArotN_Q0ldlq_big` | 0,1078 | 0,106 · 0,103 · 0,091 · 0,113 · 0,109 · 0,102 · 0,105 · 0,132 | 90,5 % |
| `A0_Qperm_ldlq` | 0,1128 | 0,110 · 0,115 · 0,095 · 0,114 · 0,116 · 0,107 · 0,108 · 0,138 | 90,3 % |

Le dommage du témoin est **plat** sur 8192 tokens (0,09–0,13 par tranche de 1024, sans tendance) et sur les positions 0–1024 de la sélection
(0,10–0,13 par tranche de 128) : pas de signe d'accumulation d'erreur dans l'état récurrent des Gated DeltaNet à cette échelle.
Non testé au-delà de 8192 tokens, ni en génération avec états persistants.

**Vision — première mesure (diagnostic, hors sélection).** 24 invites image+texte (12 photos COCO sous licence CC BY / CC BY-SA, 12 images
rendues : pages de texte, tableaux, graphiques) ; le professeur génère 64 tokens par invite et la KL est mesurée sur ces positions, à préfixe imposé.
E8 sans rétroaction au multiplicateur du témoin (`results/diag_vision_check.json`) :

| Quantifié | KL | Accord top-1 | Photos | Texte | Tableaux | Graphiques |
|---|---:|---:|---:|---:|---:|---:|
| Tour Vision seule (2,830 bpw, 460 M poids) | 0,024 | 96,4 % | 0,043 | 0,005 | 0,004 | 0,007 |
| Chemin texte seul (2,825 bpw) | 0,102 | 93,4 % | 0,094 | 0,146 | 0,124 | 0,063 |
| Les deux | 0,126 | 92,3 % | 0,138 | 0,153 | 0,119 | 0,069 |

Au même taux que le texte, la tour Vision ajoute près de la moitié du dommage du chemin texte sur les photos. Elle ne pèse que 1,7 % des
poids : lui donner 1 bit de plus coûterait 57,5 Mo, soit 0,017 bpw repris au reste. **Estimé**, non testé : ce transfert diviserait par ~4
le dommage Vision pour ~2 % de KL en plus sur le texte. **MTP : non chargé par ce runtime, toujours non évalué.**

## 7. Coûts

| Poste | Valeur | Statut |
|---|---|---|
| Passe E8-LDLQ séquentielle (texte entier, 144×1024 tokens de calibration) | 17 min sur A100 (15,4 s/couche), pic 64,4 Gio alloués ; points de reprise toutes les 4 couches | mesuré |
| Arrondi E8 sans rétroaction (texte entier) + calage du budget | 37 s | mesuré |
| Évaluation sélection + confirmation | 70 s | mesuré |
| Statistiques Fisher diagonales (144×512 tokens, rétropropagation) | 171 s, pic 75 Gio | mesuré |
| Élagage de largeur + moindres carrés (64 couches) | 4,4 min | mesuré |
| Stockage de A_b | +4,43 M valeurs FP16 = 8,9 Mo (0,0026 bpw), payés sur le budget de symboles | mesuré (compté) |
| Mémoire et calcul de A_b à l'exécution | +8,9 Mo ; une addition de vecteur par linéaire | estimé |
| Rotation du résiduel | aucun octet ni calcul supplémentaire une fois fusionnée ; conversion hors ligne | estimé |
| Coût d'inférence des structures rejetées | sans objet | non testé |
| Sprint complet | ≈ 6 h 55 de pod (08:00–14:55 UTC) × 2,11 $/h ≈ 14,6 $ ; 9 passes E8-LDLQ complètes réussies, 4 perdues sur incidents | estimé (tarif × durée ; le solde RunPod fait foi) |

## 8. Limites — à lire avant toute décision produit

* **KL, accord top-1 et perplexité ne sont pas des pourcentages de capacités.** Aucun benchmark de tâche n'a été exécuté.
* **Vision** : une seule mesure diagnostique sur 24 invites, sans rétroaction LDLQ. **MTP** : non chargé, non évalué. Leur budget reste supposé au même taux que le texte.
* **Génération** (échantillonnage, cache, états persistants) : non évaluée ; seul le forward à préfixe imposé l'est, jusqu'à 8192 tokens.
* Contexte ≤ 1024 tokens pour la sélection et la confirmation ; 12 documents seulement à 8192 tokens, sur 3 domaines. Rien au-delà.
* Les poids évalués sont les poids décodés convertis en **BF16** dans le modèle ; le dtype d'exécution du produit reste à fixer.
* Octets : coût **idéal** du codec, validé physiquement sur 7 extraits (≤ 330 k poids chacun), pas sur des matrices entières ; aucun paquet de 10 Go écrit, aucun décodage à l'échelle dans un processus indépendant.
* Une seule graine de calibration et un seul corpus (6 domaines, 7 langues). Plancher de bruit numérique : 1,7·10⁻⁴ nat pour un simple changement de composition des lots, ~5·10⁻⁴ nat pour toute réécriture d'une couche en BF16.
* Résolution statistique : ±15 % du témoin sur 72 documents, ±5 % sur 600. Seules la rotation et l'ordre des colonnes ont été évalués sur 600 documents ; le biais et les échelles de colonne ne l'ont été que sur 72 + 72.
* Le domaine « chat » pèse lourd dans la KL moyenne (dommage 2 à 6 fois supérieur aux autres pour toutes les variantes).
* La référence historique L0 n'est pas disponible : **le témoin est mon portage GPU de la référence v3.1, pas L0**. `rate_lambda > 0`, BGCC, cible asymétrique et polissages n'ont été activés ni pour le témoin ni pour les candidats.
* Les reconstructions (profondeur, largeur) n'ont reçu qu'une récupération locale en forme close. Une récupération par entraînement pourrait réduire leur dommage ; elle améliorerait aussi le témoin et n'a pas été tentée.
* Le runtime a signalé l'absence des noyaux optimisés `flash-linear-attention` / `causal_conv1d` : exécution de référence PyTorch, correcte mais lente.

## 9. Incidents du sprint (transparence)

* Deux passes LDLQ ont été perdues : ma passe séquentielle réutilisait un cache d'états entre lots (fuite mémoire **et** activations faussées).
  Corrigé, puis vérifié bit à bit contre le forward ; **aucun résultat rapporté ici n'a été produit avec le code fautif**.
* Une passe A_b a échoué sur une covariance centrée non définie positive (annulation en simple précision) ; corrigé par centrage avant accumulation.
* Une première passe de rotation a échoué : elle divisait par des gains de normalisation dont un vaut exactement 0 ; corrigé en repliant les gains sans division.
* Un diagnostic a saturé la VRAM (copies float64) ; les passes longues écrivent désormais un point de reprise toutes les 4 couches.
* Un diagnostic local (colonnes, H10) a consommé 25 min de GPU au lieu des 3 prévues.
* La clé `RUNPOD_API_KEY` du pod a été affichée dans le journal de session par un filtre de masquage défaillant (jamais écrite dans le dépôt).
  Une requête en lecture à l'API RunPod a été refusée par le garde-fou de permissions et n'a pas été contournée.
* Le dossier `DOSSIER_PISTES_ARCHITECTURALES_LUMEN.md` rapporte pour le biais « 0,133 contre 0,1123 » : la valeur enregistrée est **0,11325**.

## 10. Décision d'intégration et suite

**Décision.** Intégrer à Lumen : **rien de structurel**. Conserver l'architecture d'origine et la passe E8-LDLQ comme base (`A0Q0`).

| Élément | Décision | Motif |
|---|---|---|
| Suppression de couches, réduction de largeur, tête liée, factorisation, bases partagées, correction de faible rang | ne pas intégrer | pertes mesurées ou diagnostics chiffrés défavorables |
| Biais FP16 par linéaire (A_b) | ne pas intégrer avec LDLQ ; **à garder en réserve** pour tout chemin sans rétroaction (matrices trop grandes pour un Gram, variante rapide) | −15 à −28 % sans rétroaction pour 8,9 Mo ; neutre avec LDLQ |
| Rotation du résiduel | ne pas intégrer | égalité sur 600 documents, coût d'intégration réel (toutes les interfaces, Vision, MTP), équivalence non exacte en BF16 |
| Ordre des colonnes de LDLQ | ne pas intégrer | égalité sur 600 documents (tendance défavorable : 239 documents améliorés sur 600) |
| Pas guidés par Fisher, biais ancré professeur | proscrire dans ces formes | catastrophiques |
| Protocole | **adopter** : témoin LDLQ obligatoire, ≥ 600 documents pour tout effet attendu < 15 %, vérification d'équivalence en BF16 de toute transformation « exacte », points de reprise | les trois ont changé des conclusions pendant ce sprint |

**Ce qu'il faut tester ensuite, par ordre de valeur attendue.**

1. **Récupération par entraînement, à effort égal pour le témoin et les candidats.** L'erreur restante après LDLQ est petite par matrice et
   sans structure : aucune reparamétrisation ne la réduit, seul un ajustement global des paramètres continus (pas, gains de normalisation,
   biais) contre la KL du professeur peut le faire. C'est aussi la seule voie qui pourrait réhabiliter une reconstruction (largeur,
   profondeur), dont le dommage BF16 n'a reçu ici qu'une récupération locale en forme close. Non testé.
2. **Budget Vision.** Au même taux que le texte, la tour Vision ajoute une KL de 0,043 sur photos. Elle pèse 1,7 % des poids : mesurer la
   frontière texte / Vision avec une vraie passe LDLQ sur la tour et un jeu d'images plus large.
3. **Évaluation de capacités.** Tâches indépendantes, génération avec cache et états persistants, contextes > 8192 tokens, MTP réellement
   chargé. Aucune revendication produit n'est fondée sans cela.
4. **Paquet réel.** Encoder les matrices entières avec le codec, fermer les 10 Go par des fichiers, décoder dans un processus indépendant.
   Le calcul des poids tournés ou repliés, s'il y en a, doit se faire en précision supérieure au BF16.
5. **Recette Lumen non activée ici** : `rate_lambda > 0`, cible asymétrique complète, BGCC. Hors du périmètre architectural ; à juger
   avec le même protocole (600 documents).
6. **Pistes à entraînement lourd** (fusion de têtes, MLA, réécriture de la règle delta, partage en profondeur) : deuxième campagne,
   seulement si un besoin mesuré le justifie. Rien dans les mesures actuelles ne désigne l'attention ou l'état récurrent comme le maillon faible.
