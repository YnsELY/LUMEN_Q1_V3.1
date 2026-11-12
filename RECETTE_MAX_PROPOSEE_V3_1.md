# Recette candidate LUMEN Q1-v3.1 — Contrats corrigés, promotion progressive

**Statut : outils CPU testés + proposition d’intégration.** Cible inchangée : Formic-Max complet sous **10 000 000 000 octets**, sans abandon délibéré de domaine. Les détails du checkpoint proviennent du projet ; aucune substitution de la révision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` n’est autorisée.

## 1. Ce qui change et ce qui ne change pas

| Composant | Décision |
|---|---|
| Q0, références historiques L0/S0 | Conservées, immuables, distinctes du prototype CPU |
| LDLQ / onde frontale | Algèbre conservée, symétrisation et gardes corrigées ; candidat à raccorder |
| Calibration asymétrique ridge | Premier pilote ; aucune activation globale avant contrôle réel |
| BGCC conditionné à la sortie complète | Formulation et solveur testés ; option de recherche, non activée par défaut |
| Polissage libre / iso-coquille | Non activé : résultats de sélection historiquement négatifs |
| Cible quadratique fraîche | Primitive mathématique, non validée comme stratégie de récupération |
| E8/rANS | Oracle CPU décodable ; format produit GPU non remplacé implicitement |
| Métriques et acceptation | Contrats stricts, contrôle après décodage, biais de cache explicite |
| Allocation | Modèles conditionnels ou DP additive ; pas d’allocation optimale Qwen déclarée |
| MTP, Vision, contextes longs | Branches et évaluations spécifiques à mettre en œuvre ; aucune préservation présumée |

## 2. Passe 0 — Identité et contrats

Conserver hashes des poids, tokenizer, processeur, template, runtime et kernels. Identifier les paramètres effectivement chargés, y compris MTP. Travailler sur copie et ne pas modifier Q0.

Pour une transformation `W'=D_o^-1 W D_i U^T`, les entrées deviennent `x'=U D_i^-1 x`. Utiliser les statistiques dans ces mêmes coordonnées : `H'=U D_i^-1 H D_i^-1 U^T`, `G'=D_o^T G D_o`. La version CPU de matrice n’embarque pas automatiquement ces transformations : tout facteur nécessaire à l’inférence doit être sérialisé et compté.

Un tenseur conservé en BF16 peut être accumulé en FP32 ; ce sont des objets différents du fichier. Toute modification continue doit être exportée dans le dtype réellement retenu avant évaluation. Le scaling défavorable de l’Étape 1 reste désactivé dans son régime, sans prétendre condamner tous les scalings.

## 3. Passe 1 — Captures et statistiques

Le schéma 2 valide des formes, des hashes et des partitions. Il n’atteste pas que le collecteur a exécuté le bon forward. Auditer séparément : mêmes préfixes et positions, normes complètes avant découpage, vraies entrées de la sous-couche, états et resets, biais Vision, chemin MTP et code exécuté.

Ne pas utiliser les embeddings initiaux comme substituts aux entrées profondes. Pour un essai sur une tranche, conserver les autres contributions à la projection/non-linéarité ; sinon étiqueter le test comme système réduit.

Séparer calibration, sélection et confirmation par documents/épisodes. Des colonnes `X_fp == X_q` peuvent être correctes en début de réseau ou avec préfixe intact. La diversité effective, le conditionnement et la régularisation comptent ; **aucun seuil universel `Ncal >= 10n` n’est adopté**.

Les sondes Fisher ont une identité de covariance sous une distribution spécifiée. Les approximations bloc-diagonales/Kronecker ne deviennent pas le Hessien complet sans biais. Les statistiques de MTP et Vision exigent leurs propres entrées/sorties. Aucune nouvelle hiérarchie fixe des sensibilités n’est imposée.

## 4. Passe 2 — Première expérience de calibration

### Cible symétrique et cible asymétrique

Comparer, sur les mêmes couples d’entrées :

\[
J_{sym}(B)=\|B X_q-WX_q\|_F^2/N+\rho\|B-W\|_F^2,
\]

\[
J_{asym}(B)=\|B X_q-WX_{fp}\|_F^2/N+\rho\|B-W\|_F^2.
\]

Pour la seconde, les équations normales sont :

\[
B(X_qX_q^T/N+\rho I)=W X_{fp}X_q^T/N+\rho W.
\]

`asymmetric_target` résout cette cible continue. Ce n’est **pas** une reproduction de toutes les mises à jour séquentielles de GPTAQ. Elle doit être quantifiée, exportée, décodée et évaluée. Les biais et cibles observées sont pris en charge explicitement par l’appelant.

Les candidats reçoivent soit un plafond commun avec une vraie frontière taille/qualité, soit une contrainte plus stricte `bytes(candidat) <= bytes(contrôle retenu)`. Inclure les paramètres du contrôle de départ dans sa grille. Le budget temps doit compter la résolution, les candidats, le codage et l’évaluation. Rapporter les coûts non égalisés.

### Quantification et métriques

`quantize_ldlq` est une référence lente en Python/NumPy. Les facteurs symétriques définis positifs sont contrôlés ; une matrice invalide ne reçoit pas de pivots artificiellement tronqués. Le sélecteur E8 recherche un ensemble fini de candidats : aucune optimalité globale du réseau n’est revendiquée.

Les échelles sont arrondies avant la sélection. Les fichiers effectivement retournés sont décodés pour construire les poids évalués. Les coûts entropiques locaux servent à la recherche ; seul le fichier ferme le budget physique.

## 5. Variante BGCC : expérimentale

Avec porte quantifiée `g` et projection de sortie `D` fixées :

\[
J(W)=\|D[g\odot(WX_q)]-Y_{fp}\|_F^2/N+\rho\|W-W_0\|_F^2.
\]

L’opérateur normal appliqué à une variation `V` est :

\[
\mathcal A(V)=\{g\odot[D^T D(g\odot VX_q)]\}X_q^T/N+\rho V.
\]

La réalisation applique les produits sans construire tous les Grams par canal. Elle exige encore des activations et du calcul ; aucune durée H100 ni passage à l’échelle 27B n’est établi.

**Condition de validation numérique :** le solveur doit retourner `converged=True` avec un vrai résidu vérifié. Sinon la cible n’est pas promue. Une résolution exacte du problème de calibration ne prouve pas un gain hors échantillon. Les résultats synthétiques antérieurs de BGCC restent irréguliers ; la variante est désactivée par défaut.

Vision n’a pas nécessairement un SwiGLU. Un MLP `fc2(GELU(fc1(x)))` exige un autre problème de calibration ; on ne remplace pas SiLU par GELU dans une formule bilinéaire.

## 6. Passe 3 — Récupération avec refus d’une régression

Le polissage continu/discret peut être exploré seulement sous un protocole séparé. L’artefact accepté comprend codes, échelles, tables, groupements et métadonnées ; un rollback ne restaure pas un sous-ensemble de paramètres seulement.

Une amélioration du critère à table fixe n’implique ni amélioration de sélection ni conservation des octets après refit. Vérifier les trois. L’invariance iso-coquille exige notamment mêmes symboles position par position, parité en mode 1, groupement, table et échelles. Les résultats négatifs historiques demeurent visibles.

`compare_matrix_artifacts` exécute ce contrôle sur de vraies matrices CPU décodées. Le remplacement à l’échelle du modèle doit garder toutes les interfaces et sentinelles pertinentes. `accept_decoded` seul ne prouve pas la provenance d’un score fourni par son appelant.

## 7. Passe 4 — Export et confirmation

Contrôler taille complète, décodage dans un processus indépendant, hashes et fidélité après rechargement. La somme d’un payload estimé et d’auxiliaires hypothétiques n’est pas un fichier de 10 Go.

Le codec `LE8CPU3` / `LMATv3` fournit cet oracle pour de petites matrices. Il ne fournit pas le chargement multimodal complet, le kernel GPU, les 18 shards packagés ni une exécution en 10 Go de VRAM. Le prétraitement, l’indexation et les conversions nécessaires au produit doivent être explicités et comptés.

La confirmation réservée vient après le gel des choix. KL agrégée, MSE et perplexité ne sont pas des pourcentages de capacités. Toute revendication produit nécessite des tâches indépendantes et une comparaison complète à périmètre égal.

## 8. Intégration avec la recherche architecturale

Versionner `A0Q0`, `A0Q1`, `A1Q0`, `A1Q1` quand ces cases existent réellement. Recalculer les statistiques si les entrées, bases, dimensions ou états changent. Ne pas sommer les gains des deux agents ; ne pas consommer automatiquement le budget réservé au pilote structurel.
