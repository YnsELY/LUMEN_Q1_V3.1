# FORMIC / LUMEN — Architecture × compression sur le vrai Qwen3.8-27B

**Sprint du 13 novembre 2026 — pod RunPod A100 80 Go, 2,11 $/h (tarif et solde renseignés par l'opérateur).**
Base de compression : LUMEN Q1-v3.1. Tout ce qui suit a été exécuté sur le BF16 local complet
(`/root/models/qwen3.8-27b`, 18 shards, 27 781 427 952 paramètres), jamais sur un modèle de substitution.

> Vocabulaire de preuve : **publié** (source primaire relue, `docs/SOURCES_PRIMAIRES.md`), **mesuré** (ce sprint, fichiers bruts
> dans `results/`), **estimé** (calcul à partir de mesures), **non testé**.

@@RESUME@@

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
| H11 | Rotation cohérente du flux résiduel, gains de normalisation repliés | transformation exacte | passe LDLQ complète dans la base tournée, émulée dans le graphe d'origine | @@H11@@ |

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

## 4. Résultats sur la sélection (72 documents, 73 728 positions)

ΔKL : différence appariée par document, IC 95 % par bootstrap sur les documents. « octets/cible » : écart du coût idéal total au budget commun.

@@TABLE_SELECT@@

Lecture :

@@LECTURE@@

## 5. Confirmation (72 documents réservés, lus après gel)

@@CONFIRM@@

## 6. Contexte long, récurrence, Vision

@@LONG@@

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

## 8. Limites — à lire avant toute décision produit

* **KL, accord top-1 et perplexité ne sont pas des pourcentages de capacités.** Aucun benchmark de tâche n'a été exécuté.
* **Vision** : une seule mesure diagnostique sur 24 invites, sans rétroaction LDLQ. **MTP** : non chargé, non évalué. Leur budget reste supposé au même taux que le texte.
* **Génération** (échantillonnage, cache, états persistants) : non évaluée ; seul le forward à préfixe imposé l'est, jusqu'à 8192 tokens.
* Contexte ≤ 1024 tokens pour la sélection et la confirmation ; 12 documents seulement à 8192 tokens, sur 3 domaines. Rien au-delà.
* Les poids évalués sont les poids décodés convertis en **BF16** dans le modèle ; le dtype d'exécution du produit reste à fixer.
* Octets : coût **idéal** du codec, validé physiquement sur 7 extraits (≤ 330 k poids chacun), pas sur des matrices entières ; aucun paquet de 10 Go écrit, aucun décodage à l'échelle dans un processus indépendant.
* Une seule graine de calibration et un seul corpus (6 domaines, 7 langues). Le bruit numérique d'évaluation lié à la composition des lots est de 1,7·10⁻⁴ nat.
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

@@DECISION@@
