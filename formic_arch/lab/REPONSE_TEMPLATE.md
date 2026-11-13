# Réponse mesurée au dossier « Attention et pistes architecturales complémentaires »

**13 novembre 2026.** Le dossier (`DOSSIER_PISTES_ARCHITECTURALES_LUMEN.md`) est une étude documentaire ; ce document confronte chacune
de ses pistes aux mesures faites sur le vrai Qwen3.8-27B pendant le sprint. Détails et tableaux : `RAPPORT_ARCHITECTURE_X_LUMEN.md`.

## Une correction factuelle

Le dossier rapporte pour le candidat à biais « 0,133 contre 0,1123 » et qualifie la lecture « 0,1133 » d'incorrecte. La valeur enregistrée
est **0,11325** (`results/variant_Ab_Q0ldlq.json`). Le biais constant n'est pas battu par le témoin : il est **à égalité** avec lui sous LDLQ
(écart apparié +0,0009, IC 95 % [−0,016 ; +0,018] en sélection ; −6,7 % en confirmation, non significatif) et gagne 15 à 28 % sans rétroaction.

## Piste par piste

| Piste du dossier | Priorité du dossier | Ce qui a été fait | Verdict mesuré |
|---|---|---|---|
| §4.1 Changement de base cohérent du résiduel (rotation, gains repliés) | A | Diagnostic des pivots LDL ; passe E8-LDLQ complète dans la base tournée, en version émulée puis native ; vérification d'équivalence avant quantification | @@ROT@@ |
| §6.5 Réorganisation exacte SwiGLU / frontières E8 — version diagonale (`S`) | A/B | Échelles de colonne tirées des pivots LDL sur toutes les matrices, passe complète à débit égal par matrice ; contrôle local | **Rejetée** : +5,6 % (sélection), +0,1 % (confirmation) ; l'erreur locale ne baisse pas non plus à bits égaux |
| §6.5 — version permutation (`P`), ordre de traitement de LDLQ | A/B | Diagnostic local sur 5 couches ; passe complète | @@PERM@@ |
| §6.3 Préserver puis quantifier, correction de faible rang | A | Spectre de l'erreur pondérée après le vrai LDLQ ; le cas de rang 1 (moyenne) testé de bout en bout par le biais | **Rejetée sur diagnostic** : un rang 64 retire au mieux 3–17 % de l'erreur pour 0,26–0,37 bpw, quand les mêmes bits en précision E8 en retirent 30–40 % |
| §6.1–6.2 Factoriser puis quantifier | A | Perte de troncature au rang finançable à octets égaux, pondérée par les activations | **Rejetée sur diagnostic** : la troncature seule vaut 2 à 9 fois l'erreur totale de LDLQ sur la même matrice |
| §6.4 Bases partagées entre couches | B | Rang effectif conjoint de couches voisines | **Rejetée sur diagnostic** : ≈ 4000–4260 sur 5120, soit presque l'espace entier |
| §6.6 Exceptions précises, scission de canaux | « seulement si diagnostic » | Concentration de l'erreur après LDLQ | **Non motivée** : les 0,1 % d'entrées les plus fautives portent 1,2–5,7 % de l'erreur |
| §7.1 Blocs hétérogènes | B | Sensibilités par couche et par famille ; répartition des bits par famille | **Non concluante** : aucune couche ni famille « bon marché » ; la répartition prédite à −16 % mesure +0,8 % |
| §7.2 Réduction du résiduel après rotation (SliceGPT) | — | Non testé | La largeur des MLP, elle, perd +23 à +33 % à octets égaux ; résiduel non essayé |
| §4.2–4.3 Transformations compatibles avec les portes de l'attention | B | Jauge diagonale V/O couverte par les échelles de colonne de `o_proj` / `out_proj` | **Rejetée** avec la ligne « version diagonale » |
| §4.4 Moins de têtes KV, MLA | C | Chiffrage | **Écartée pour les 10 Go** : `k_proj` + `v_proj` = 0,6 % des poids, et ce sont les poids les plus sensibles (×6 par rapport aux entrées de MLP) |
| §5.1 Jauge projection / convolution GDN | B | Non testée isolément | Les pas par ligne de Lumen absorbent déjà une échelle de ligne ; gain attendu nul (argument du dossier lui-même) |
| §5.2 Base de l'état récurrent | secondaire | Non testé | KL plate jusqu'à 8192 tokens : pas de dérive de l'état à réparer à cette échelle |
| §5.3, §7.3, §7.4 Réécriture de la récurrence, partage en profondeur, remplacement d'attentions | C | Non testé | Demandent de l'entraînement ; deuxième campagne |
| §8 Tête, Vision, MTP | — | Tête ← embedding : R² = 0,17 ; première mesure Vision | Vision au même taux n'est pas gratuite (KL 0,043 sur photos) ; MTP toujours non chargé |

## Ce que les mesures ajoutent au dossier

1. **Le critère local n'est pas un guide fiable à ce taux.** Quatre idées optimales sur le papier ont échoué en KL : pas guidés par Fisher
   (catastrophique), échelles de colonne par second moment (+12 %), répartition des bits par famille (0 %), échelles par pivots LDL (+6 %).
   Toute piste de représentation doit être jugée sur une passe complète et une évaluation assez grande, pas sur une marge théorique.
2. **Le seuil de détection compte.** Sur 72 documents, l'intervalle de confiance d'une différence de KL vaut ±15 % du témoin, à cause de
   quelques documents « chat ». Les effets attendus des transformations exactes sont de l'ordre de 2 à 8 % : il faut ~600 documents pour les voir.
3. **Une transformation « exacte » ne l'est pas en BF16.** Réécrire une seule couche (même un aller-retour de rotation) déplace la KL de
   ~5·10⁻⁴ nat ; réécrire tout le chemin texte dans une base tournée coûte 0,003 nat en sélection et 0,009 en confirmation (0,044 sur le
   domaine math). Le produit devra calculer les poids tournés en précision supérieure avant quantification.
4. **Le modèle masque déjà sa dimension massive aux MLP** (gain de RMSNorm 1/256, voire 0, couches 0–10). La base native n'est pas
   quelconque : sur les entrées de MLP, 42 % des poids, il n'y a presque rien à gagner (pivots déjà plats à 1,04–1,15).
5. **Attention et Gated DeltaNet** : mauvaises cibles de réduction (6 % et 20 % des poids, sensibilité par poids 2,4 à 6 fois celle des MLP,
   aucune couche redondante), seules cibles où un changement de représentation a de la marge (0,07–0,65 bit sur le critère local).

## Organisation proposée pour la suite

@@SUITE@@
