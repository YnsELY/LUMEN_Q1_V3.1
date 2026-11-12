# Recette candidate LUMEN-Max Q1-v3-ref — Diff et contrats d’implémentation

**12 novembre 2026.** Remplace la proposition Q1-v2 pour ce chantier ; ne remplace pas automatiquement la recette de référence Q0. Le rapport associé est `RAPPORT_RECHERCHE_RECETTE_MAX_V3.md`.

## 1. Identités et invariant

- `A0` : architecture et checkpoint fournis `Qwen/Qwen3.8-27B@37561816e66b5b9153b1b64c10968c18c1e6f4a0`, à vérifier sur le pod.
- `Q0` : recette historique du projet ; fichiers et résultats inchangés.
- `L0_historique`, `S0_historique` : configurations réelles de l’Étape1, non réimplémentées par simple renommage.
- `Lref3` : référence CPU livrée, Block-8 LDLQ corrigé, candidats E8 finis, tables explicites, échelles FP16 et flux de référence `LE8CPU3`.
- `Q1-v3-ref` : propositions et composants ci-dessous. Les nombres de qualité synthétique ne la qualifient pas comme recette supérieure à Q0.

**Plafond Max : 10 000 000 000 octets complets.** Le budget comprend texte, Vision, MTP, normes, contrôles, tables, échelles, index, conversions stockées et métadonnées requises. Ne pas confondre package et VRAM totale. Tous les domaines restent visés ; les domaines non évalués doivent être identifiés.

## 2. Décisions par défaut

| Brique | Décision | Motif |
|---|---|---|
| Inverse LDLQ, Gram SPD, SiLU | Remplacer les réalisations incorrectes | Invariants et contre-tests passent |
| Scaling d’entrée historique défavorable | Ne pas réactiver par défaut | Ancien signal local rapporté ; aucune preuve générale d’interdiction des scalings |
| Hadamard | Garder comme option/contrôle historique | La bibliothèque ne l’insère pas silencieusement dans les nouveaux essais ; ses tables/graines sont à compter |
| Tête prédite depuis embedding | Toujours suspendue | Aucune nouvelle donnée justifiant la réouverture |
| Cible asymétrique régularisée | Premier pilote recommandé | Formulation cohérente, signal synthétique plutôt favorable, transfert à vérifier |
| BGCC par ligne ou sortie complète | Expérimental, désactivé | Code mathématique de référence, résultats réservés insuffisamment réguliers |
| Courbure Kronecker enrichie, cible gradient fraîche | Option de recherche | Collecte modèle non livrée ; pas une garantie YAQA héritée |
| Polissage E8 libre | Désactivé par défaut | Objectif peut baisser alors que fichier grossit |
| Polissage à coquille constante | Désactivé par défaut | Taille égale vérifiée, cinq essais de sélection défavorables |
| Codeur CPU coquille-uniforme | Référence de vérification | Vrai flux ; pas le format existant du produit |
| Codeur conditionné par co-ensemble | Option choisie par octets réels | Gain sur source synthétique biaisée, perte ailleurs |
| Allocation continue | Outil de modèle conditionnel | ω doit venir de mesures ou être marqué hypothèse |
| Allocation discrète additive | Outil sur tableaux mesurés | Ne résout pas les interactions ni toute la variante RCO |
| GKD, RL, grandes passes de récupération | Report conditionnel | Pas de campagne complète implémentée ou chronométrée |
| Correcteur QERA | Option distincte avec B | Coût corrigé ; pas un module ajouté par défaut |

## 3. Passe0 — État de référence et mathématiques

Conserver l’original en lecture seule. Épingler le tokenizer, le processeur, le runtime, les kernels et les paramètres de génération. Vérifier les normes (`1+w` vs `w`) et les portes dans le code réellement chargé. Ne pas corriger un désaccord entre config et code en remplaçant silencieusement une fonction.

Pour les vecteurs colonnes :

$$W'=D_o^{-1}WD_iU^\top,\quad x'=UD_i^{-1}x,$$
$$H'=UD_i^{-1}HD_i^{-1}U^\top,\quad G'=D_o^\top GD_o.$$

Ces expressions présupposent diagonales inversibles et `U` orthogonale. La régularisation doit être appliquée dans des coordonnées définies, puis transformée avec cohérence. La valeur de λ doit correspondre aux unités de l’objectif.

Dans la référence CPU, la matrice est quantifiée dans les coordonnées fournies. Pour insérer un Hadamard, transformer aussi les statistiques puis revenir à la bonne base au runtime ; sérialiser toute conversion nécessaire. Une matrice de largeur5120 n’est pas directement traitable par le petit Hadamard dense puissance-de-deux livré : l’intégration doit définir les blocs et leur layout.

Les petits tenseurs gardés en BF16 natif peuvent rester stockés ainsi s’ils ne changent pas. Une norme optimisée en FP32 demande soit son stockage FP32 budgété, soit un arrondi BF16 dont l’effet a été réévalué. « Protéger le type » n’est pas « préserver exactement la valeur optimisée ».

## 4. Passe1 — Captures réelles et statistiques

L’entrée d’un MLP doit être capturée au point exact exécuté, après les opérations précédentes. Le collecteur doit conserver les identifiants de documents et leur partition. `Xfp` et `Xq` correspondent aux mêmes positions, masques, préfixes et états initiaux ; seul le chemin de calcul diffère.

La capture doit contenir la pleine largeur pour la reconstruction globale. Si une projection est testée par tranches, la contribution des colonnes restantes doit être gardée avant SiLU ou GELU. Pour une tranche normalisée, exiger le dénominateur RMS complet.

Le schéma de capture livré refuse les entrées MLP A0 de largeur128. Il ne remplace pas le collecteur ni un audit des logs du pod. Les documents/calibration, sélection et confirmation sont disjoints. Les IDs de tokens peuvent naturellement se recouper entre textes.

Commencer par `H=XqXqᵀ/N+ρI`. Les blocs de sortie ou sondes ne doivent être ajoutés qu’après une première ablation, avec définition du Fisher ou Gauss-Newton voulu. La covariance des sondes n’est pas un théorème d’absence de biais du Hessien complet. Pour MTP, préciser quelle distribution est utilisée au point de développement.

Ne pas déterminer toutes les couches à partir de la kurtosis des seuls extraits couche0. Le tirage historique de256lignes est rapporté comme aléatoire `47Q/45K/164V`, et non comme « les256premières ». La kurtosis V élevée reste un fait rapporté non recalculé ici.

## 5. Passe2 — Quantifier et reconstruire avec des objectifs explicites

### 5.1 LDLQ de référence

`quantize_ldlq` décompose `H=U_I D_I U_Iᵀ` et éventuellement `G=U_O D_O U_Oᵀ`. L’inverse du sous-bloc triangulaire est calculé correctement. Les dépendances d’onde frontale sont testées sur une équivalence dense indépendante. Le sélecteur minimise seulement parmi ses candidats finis.

Le coût de débit utilisé est **bits par vecteur E8 de8poids**. Le λ ne doit pas changer implicitement d’unité lorsque l’on rapporte des bits/poids. Un Gram normalisé par sa trace doit entraîner la transformation cohérente du λ ; la référence évite cette normalisation cachée.

Les échelles FP16 sont arrondies avant le choix des codes. Les statistiques, λ et tables utilisées sont enregistrés. Les tables sont figées pendant un balayage ; leur réestimation à l’export impose un nouveau contrôle des octets.

### 5.2 Calibration asymétrique

Calculer la cible par le problème ridge documenté, puis quantifier. Comparer à un contrôle qui dispose du même plafond et d’un effort de recherche de réglages documenté. Un meilleur résultat après plus de calcul n’est pas automatiquement attribuable au mécanisme seul.

### 5.3 BGCC

La porte et le down utilisés par l’objectif sont figés et identifiés (BF16 ou quantifiés). Ne pas entraîner une cible contre `down_BF16` puis prétendre avoir mesuré la même fonction sous un `down_quantifié` non testé.

- Référence par ligne : Gram pondéré par la porte ET second membre cohérents, métrique retenue lors de l’arrondi.
- Référence sortie complète : opérateur SPD matrice-free, CG jusqu’au résidu déclaré. Un arrêt incomplet est rapporté, pas renommé solution exacte.
- Référence discrète : petits Grams8×8, résidu après down, comparaison du code courant et des candidats.

Aucun seuil universel `N/n` n’est imposé. Régularisation et arrêt sont choisis sur sélection, puis gelés. Ne pas généraliser les gains de calibration. Le transfert à Vision n’est pas « remplacer SiLU par GELU ».

### 5.4 Codec et échelles

Deux stratégies indépendantes : porter l’algorithme corrigé dans le format Lumen existant, ou étudier le nouveau flux de référence. La compatibilité binaire n’est pas supposée.

Le codec CPU garantit une reconstruction déterministe sur les tests réalisés. Il ne propose ni déquantification fusionnée CUDA, ni profil partagé14tables, ni FP8 SG512, ni accès aléatoire d’embeddings validé. Toute adoption au produit passe par une spécification et des mesures de ces points.

## 6. Allocation et budget

Collecter un tableau par groupe : configuration, provenance, octets réels, perte locale ou globale, incertitude, coût temporaire. Déduire les paramètres protégés et ressources partagées du budget une seule fois.

Une allocation additive peut préparer un candidat, mais sa qualité jointe doit être mesurée. Les contraintes Vision/MTP incluent des tests de comportement ; un plancher d’octets ne garantit pas leurs performances. Les versions des baselines ne changent pas pendant une comparaison.

Le script ne fournit plus un faux total « exact optimisé » à partir d’omegas manuels. Il sépare scénarios de capacité, modèle continu, DP à coûts entiers et comptage de fichiers physiques. Le packager final compte les fichiers effectivement requis, sans remplir artificiellement le plafond.

## 7. Passe3 — Acceptation, export et rollback

1. Partir d’un artefact accepté et décodable.
2. Produire une proposition continue ou discrète, dans le budget d’apprentissage autorisé.
3. Réestimer les profils, arrondir toutes les échelles stockées et encoder réellement.
4. Décoder dans un état indépendant ; vérifier la parité avec les poids évalués.
5. Mesurer la métrique primaire et les sentinelles sur sélection.
6. Accepter seulement si taille et règles préenregistrées passent ; sinon conserver l’artefact précédent entier.
7. Figer les finalistes avant la confirmation, jamais utilisée pour ajuster le prochain λ.

`accept_decoded` implémente la logique de garde ; le calcul des métriques reste à charge du moteur. Les sentinelles de cette API suivent toutes la convention « plus bas est meilleur » : un taux d’acceptation doit donc être converti en perte, par exemple 1 moins ce taux. Un appel avec de faux nombres ne devient pas une validation. Les sauvegardes de fichiers du produit doivent être transactionnelles ; la bibliothèque rend les octets retenus et n’écrase pas l’original.

Le cache Top-K porte une interprétation explicite de ses valeurs, sa normalisation, ses indices, la masse de queue, le masque et la provenance du préfixe. Tester son classement sur une vraie KL complète. Une corrélation observée n’est pas une garantie universelle.

## 8. Passe4 — Livraison complète

Exiger dimensions, types, codec, transformations, profils, offsets et états à recharger sans hypothèse cachée. Vérifier à froid tout fichier final et son SHA-256. Réévaluer après la dernière fusion/requantification. La taille et la VRAM sont deux colonnes distinctes.

Le texte, les chemins image/vidéo et le vrai chemin MTP restent à vérifier séparément. Les benchmarks publics appartiennent à la campagne gelée finale ; ils ne servent pas à régler les candidats pendant la recherche. Une marge de non-infériorité est fixée avant confirmation ; une absence de significativité n’est pas une preuve d’égalité.

## 9. Liaison avec l’agent architecture

Conserver la matrice `A0Q0/A0Q1/A1Q0/A1Q1`. Une modification de dimensions, de bases, de porte ou d’état invalide les statistiques qui en dépendent. Aucun gain des deux agents n’est additionné sans test conjoint. Les demandes V3 n’autorisent ni un budget GPU supplémentaire ni le remplacement du mandat N3 de B.

## 10. Conditions de promotion

Q1-v3-ref peut servir d’oracle de correction et de base de prototype. Elle devient une recette Lumen intégrable au produit seulement après : reproduction des tests, portage sur le codec/moteur retenu, capture vraie, ablation à coût comparable, confirmation hors calibration, export complet et vérification des domaines/interfaces. Une brique peut être promue sans que toutes les options le soient.
