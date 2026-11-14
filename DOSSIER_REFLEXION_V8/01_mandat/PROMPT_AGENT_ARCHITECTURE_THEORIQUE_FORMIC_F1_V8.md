# FORMIC / LUMEN F1 — Recherche théorique d’optimisation architecturale

**Mandat V8 — nouvelle conversation, recette de compression figée, sans campagne expérimentale.** Ce mandat remplace les instructions antérieures autorisant une co-optimisation de la recette ou le lancement de tests GPU.

## 1. Ta mission

Tu es responsable d’une recherche scientifique sur l’architecture de **Formic**, notre version compressée de `Qwen/Qwen3.8-27B`. Notre recette de compression **Lumen F1**, retenue à l’issue de la campagne V7, constitue désormais la référence à conserver.

Nous cherchons une organisation du modèle qui exploite mieux cette recette inchangée. Ton objectif est de concevoir des modifications architecturales permettant soit **une meilleure fidélité au BF16 à taille identique ou inférieure**, soit **la même fidélité dans un modèle plus petit**.

**Cette mission porte sur la conception et la justification théorique, pas sur l’exécution d’une nouvelle campagne.** Tu dois produire des propositions suffisamment précises pour être implémentées et testées ultérieurement. Tu choisis toi-même les hypothèses et les solutions ; aucun mécanisme particulier ne t’est imposé.

## 2. Comprendre la référence avant de proposer des modifications

Commence par lire `formic_v7/RECETTE_LUMEN_V7_EXP.md`, le rapport final V7, les résultats bruts, les manifestes du paquet F1 et le code réellement utilisé. Retrouve leurs chemins dans les fichiers disponibles ; n’invente pas de fichier manquant. Consulte ensuite l’infrastructure, Lumen Q1-v3.1 et les dossiers de recherche antérieurs comme contexte historique, sans les confondre avec F1 final.

Le checkpoint de référence du projet est `Qwen/Qwen3.8-27B`, révision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Vérifie l’identité à partir des artefacts locaux et n’effectue aucune substitution silencieuse.

### Résumé transmis de la campagne V7, à confronter aux fichiers

F1 conserve l’architecture de Qwen et utilise **E8 + LDLQ, avec une allocation uniforme annoncée et sans apprentissage**. Le changement retenu est une calibration combinée de **409 600 tokens**, enrichie en conversations, appels d’outils et traces de raisonnement. Les règles exactes de pas, de débit, de protection et de codage doivent être lues dans la recette et son implémentation, pas déduites de ce résumé.

Le paquet annoncé mesure **9 974 484 649 octets sur disque**, incluant texte, Vision, MTP, petits tenseurs, tokenizer et configurations. Sa reconstruction depuis les fichiers reproduit les KL rapportées. Les comparaisons textuelles portent sur les mêmes **498 matrices**, pour un budget d’environ **9,45 Go**.

| Protocole rapporté | KL F1, en nat/token |
|---|---:|
| Protocole public Arki05, 108 fenêtres sur 128 | 0,0596 |
| Moitié réservée à la confirmation de ce protocole | 0,0613 |
| 300 documents neufs issus de la distribution interne | 0,1041 |
| Jeu interne de 600 documents | Environ 0,1015 |

**Ces valeurs ne sont pas interchangeables : chaque protocole possède sa propre référence.** Elles mesurent la fidélité des prédictions, pas un pourcentage de compétences conservées. Aucun benchmark de tâche complet n’est annoncé dans ce bilan ; Vision n’a été examinée que sur 24 invites et MTP est inclus mais n’a pas été exécuté. Le paquet se décode vers des poids denses : son poids sur disque ne prouve pas une exécution native dans moins de 10 Go de VRAM.

Ce résumé est une information rapportée, non une certification indépendante. Signale toute divergence avec les fichiers et cite la version réellement retenue. La comparaison principale de ta recherche est **F1 contre une architecture modifiée traitée par la même F1**, pas contre un ancien témoin plus faible.

## 3. Deux objectifs, dont un seul suffit à justifier une proposition

Note `A0` l’architecture actuelle, `Q_F1` la procédure figée de compression, `B` le nombre total d’octets nécessaires au paquet et `K_D` la KL moyenne `D_KL(P_BF16_original || P_candidat)` sur un protocole `D` fixé.

Pour une architecture proposée `A1`, cherche à justifier l’une des situations suivantes :

**Objectif A — meilleure fidélité sans grossir :**

`B(A1, Q_F1) <= B(A0, Q_F1)` et `K_D(A1, Q_F1) < K_D(A0, Q_F1)`.

**Objectif B — modèle plus petit sans perdre en fidélité :**

`B(A1, Q_F1) < B(A0, Q_F1)` et `K_D(A1, Q_F1) <= K_D(A0, Q_F1)`.

La taille de F1, **9 974 484 649 octets**, est le comparateur principal ; **10 000 000 000 octets** reste le plafond produit absolu. N’utilise pas simplement la marge restante pour attribuer à l’architecture un gain financé par davantage d’octets.

À cette étape, ces inégalités sont des objectifs à étayer, non des résultats à déclarer atteints. Pour une future validation de « même fidélité », propose une marge de non-infériorité justifiée et fixée avant les mesures ; l’absence de différence statistiquement significative ne démontre pas l’équivalence.

Ne sacrifie délibérément aucun domaine, aucune modalité ni MTP. Une meilleure moyenne obtenue en dégradant nettement les outils, le code ou le multilingue ne suffit pas. Un gain de vitesse ou de cache seul n’atteint pas les objectifs ci-dessus.

## 4. Frontière stricte : architecture libre, recette inchangée

**Tu ne dois pas modifier Lumen F1.** Sont figés : quantifieur E8, LDLQ et ses règles, politique de précision, règles de pas et de codage, corpus et procédure de calibration, paramètres de référence et périmètre des modules protégés. Aucun nouveau codec, nouvelle allocation apprise, ajout de données, changement de perte, distillation, RL ou fine-tuning ne peut être présenté comme une réponse à ce mandat.

Tu peux concevoir des transformations des poids et du graphe : changements de représentation cohérents, réorganisation ou restructuration de modules, connexions, factorisations, mutualisations, dimensions internes ou mécanismes de calcul. Tu peux examiner l’attention, DeltaNet, SwiGLU, les normalisations, le résiduel et leurs interfaces ; cette liste ne limite pas ta recherche.

**L’architecture d’entrée de F1 peut changer ; la méthode F1 ne change pas.** Recalculer les activations et statistiques sur cette architecture avec les mêmes données et les mêmes règles est une adaptation nécessaire, pas une nouvelle recette. Une adaptation de formes ou de raccordement doit être explicitée sans modifier la logique du quantifieur.

Les nouveaux paramètres, facteurs, tables de transformation ou raccordements doivent suivre les règles de représentation existantes et être comptés. Ne suppose pas qu’une nouvelle branche peut rester gratuitement en haute précision. Si F1 ne peut pas traiter un nouveau tenseur sans une nouvelle règle de quantification, indique cette incompatibilité plutôt que modifier la recette en silence.

Une économie structurelle reste une économie : ne la réinvestis pas dans une nouvelle allocation de bits pour faire gagner la proposition. Si une idée exige de modifier F1, classe-la séparément **hors mandat actuel**. Une architecture qui nécessite un apprentissage de récupération peut être discutée comme proposition conditionnelle pour une campagne ultérieure, sans attribuer à l’architecture seule les gains espérés de cet apprentissage ni les annoncer acquis.

## 5. Exploiter ce qui a déjà été appris

Étudie les résultats et limites des essais antérieurs avant de recycler une piste. Les rotations globales, changements d’échelle, corrections par biais, ordres de colonnes et réductions de largeur n’ont pas fourni une amélioration architecturale robuste dans les expériences rapportées. Leur échec ne réfute pas toutes leurs variantes, mais une reprise exige un mécanisme différent et une explication précise de ce que le premier essai ne couvrait pas.

Les factorisations ont aussi reçu des diagnostics spectraux défavorables : confronte une nouvelle proposition à ces diagnostics, sans les transformer en impossibilité universelle. Pour le candidat à biais, le retour corrigé indique environ **0,11325**, pas **0,133** ; les anciens dossiers contiennent cette erreur de transcription. Vérifie les résultats bruts et ne reprends pas les anciennes projections illustratives de benchmarks.

**Ne te limite pas à un catalogue de techniques connues.** Cherche des mécanismes adaptés aux erreurs et contraintes documentées de F1. Les pistes des dossiers existants sont des points de départ à critiquer, pas un programme imposé. Une transformation qui change seulement l’exécution ou le rangement du fichier doit être distinguée d’une modification du graphe ou de sa paramétrisation.

## 6. Niveau de justification exigé

Pour les quelques propositions les plus solides, construis une argumentation complète couvrant les points suivants.

### Transformation et faisabilité

Décris le calcul avant/après, les tenseurs et dimensions concernés, les équations, les raccordements et le transfert des poids préentraînés. Précise si la transformation part du BF16 ou du paquet déjà quantifié : ce ne sont pas des procédures équivalentes. Explique ce qui est conservé exactement et ce qui est approximé. Une proposition qui suppose de « réentraîner pour réparer » doit expliciter cette dépendance ; elle n’a pas encore démontré son gain dans notre cadre sans apprentissage.

Vérifie les obstacles du graphe réel : portes, SiLU, normalisations et gains effectifs, positions, attention partagée, résiduel, convolutions, états récurrents et interfaces multimodales. Ne divise pas par un gain presque nul sans traiter le conditionnement. Distingue identité en arithmétique réelle et comportement avec arrondis BF16, stockage des facteurs et décodage.

### Mécanisme d’amélioration après la même quantification

Explique pourquoi `Q_F1` devrait mieux représenter la nouvelle architecture. Une équivalence avant quantification ne prouve aucun gain après quantification. Vérifie aussi qu’une symétrie du réseau E8 ou une invariance du quantifieur ne rend pas ta transformation inutile.

Analyse conjointement l’erreur structurelle, celle des facteurs quantifiés, la propagation entre blocs et le coût des nouvelles représentations. **Ne somme ni ne soustrais arbitrairement des KL.** Une baisse de MSE locale, un spectre favorable ou des pivots plus plats ne suffisent pas à prouver une baisse de KL globale.

Lorsque tu peux établir une propriété, une borne ou une condition suffisante, fournis la dérivation et ses hypothèses. Une borne supérieure plus petite n’est pas automatiquement la preuve que l’erreur effectivement obtenue sera plus faible. Indique ce que les artefacts existants permettent de vérifier et ce qui demeure inconnu sur Qwen.

### Taille et exécution

Compte les paramètres indépendants et tous les octets supplémentaires : facteurs, indices, permutations, échelles, alignement, métadonnées et informations nécessaires au décodeur. Une matrice partagée est stockée une fois, ses adaptations et raccordements aussi. Le budget reste complet, Vision et MTP inclus.

Avec un codage entropique, moins de paramètres ne garantit pas une économie proportionnelle. Donne un bilan conditionnel, les hypothèses de débit et le seuil à partir duquel la solution devient avantageuse. Distingue poids sur disque, mémoire d’exécution, états, calcul et latence ; signale une solution compacte qui rendrait l’exécution disproportionnellement coûteuse.

### Niveau de preuve et antériorité

Sépare explicitement **résultat publié**, **mesure déjà présente dans le projet**, **propriété démontrée sous hypothèses** et **hypothèse à tester**. Consulte des sources primaires, cite leurs liens, sections et versions, et explique leur transférabilité à F1. Une méthode connue ou sa simple combinaison ne doit pas être annoncée comme une invention sans examen d’antériorité.

**Ne fabrique pas de garantie de gain sur Qwen.** Ton travail doit démontrer les mécanismes accessibles théoriquement, identifier les conditions de succès et livrer les propositions les mieux justifiées. L’incertitude n’autorise pas à remplacer la recherche par un refus ou une liste vague ; elle doit être localisée dans l’argumentation.

## 7. Travail autorisé maintenant

Lecture du dépôt et des résultats, recherche bibliographique, dérivations mathématiques et calculs légers de dimensions, d’octets et de complexité. Tu peux exploiter les statistiques déjà enregistrées en précisant leur provenance et leurs limites de transfert.

**Ne lance aucun forward du modèle, nouvelle quantification, évaluation de corpus, entraînement, benchmark, travail GPU ou téléchargement massif. Ne loue aucune ressource.** Ne modifie ni le code F1, ni les poids, ni les résultats d’origine. N’ouvre pas un jeu encore réservé pour choisir les propositions.

Une expérience future peut être décrite, mais pas exécutée. Tu peux rédiger du pseudo-code de transformation et une spécification d’intégration sans construire le pipeline maintenant. Ton autonomie concerne la recherche et la conception dans ce périmètre, pas une extension implicite de la campagne.

## 8. Livrable et arrêt

Livre un fichier principal **`PROPOSITIONS_ARCHITECTURALES_FORMIC_F1.md`**, autonome, comprenant :

1. Un résumé accessible, la référence F1 identifiée et les deux objectifs.
2. Une carte des directions examinées et de celles écartées, avec les raisons.
3. Les **deux ou trois propositions les plus solides**, si la recherche en justifie autant : spécification avant/après, équations, preuves conditionnelles, bilan taille/fidélité, compatibilité avec F1 et risques. Ne remplis pas un quota avec des idées faibles.
4. Un tableau distinguant ce qui est démontré, ce que les mesures existantes soutiennent et ce qui exige encore une validation.
5. Pour chaque proposition retenue, le plus petit test ultérieur qui pourrait la confirmer ou la réfuter, avec contrôle, métriques et critères fixés à l’avance — sans l’exécuter.
6. Les sources primaires et références exactes aux artefacts consultés.

Conserve une partie principale lisible et place les dérivations longues en annexe dans le même fichier. Classe les propositions par solidité de l’argument, potentiel et coût d’intégration, sans probabilités de réussite inventées. Si aucune n’atteint le niveau requis, présente les meilleurs candidats conditionnels et leurs verrous précis ; n’annonce pas un vainqueur artificiel.

**Arrête-toi après ce dossier.** Nous déciderons séparément des prototypes et de leur budget. Le résultat attendu est une proposition architecturale techniquement défendable et compatible avec **Lumen F1 inchangée**, pas une nouvelle recette ni un gain expérimental fictif.
