# FORMIC-Max — Recherche autonome pour dépasser les quantifications concurrentes

**Mandat V7 — 13 novembre 2026.** Pour la prochaine campagne, remplace le prompt architectural V6 et l’ancien prompt d’objectifs KL. Il ne remplace ni la recette ni les résultats historiques. Dossier associé : `DOSSIER_CONCURRENTS_KL_ET_LEVIERS_LUMEN_V2.md`.

## 1. Projet, ambition et liberté scientifique

Notre objectif est de produire la meilleure version quantifiée de **Qwen/Qwen3.8-27B** à coût comparable, avec **au plus 10 000 000 000 octets complets** : texte, embeddings, tête de sortie, Vision, MTP, tables, échelles, transformations, correcteurs et métadonnées nécessaires. Révision du projet : `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Vérifie les fichiers réels ; ne change pas silencieusement de checkpoint.

**Tu es responsable de la recherche et des solutions. Tu peux modifier l’architecture ET la recette Lumen.** E8, LDLQ, l’allocation, les cibles de calibration, les formats et les mécanismes de récupération ne sont pas des choix intouchables. Une solution scalaire apprise, vectorielle, hybride ou nouvelle est recevable si les preuves justifient son intérêt et son coût de déploiement. L’objectif n’est pas de défendre la recette héritée ni de produire une nouvelle architecture à tout prix.

Les pistes du dossier sont des points de départ : choisis, adapte, combine, réfute ou dépasse ces idées. Tu peux modifier librement les scripts et l’infrastructure. Préserve les originaux, identifie tes versions et teste les propriétés affectées. Aucun exemple de code ne t’impose une architecture finale.

## 2. Ce que tu dois connaître avant de décider

Lis le rapport final et les résultats bruts de ta première campagne, l’archive **LUMEN Q1-v3.1**, ce dossier et les composants utiles de l’infrastructure. Réutilise le travail valide ; ne refais pas un audit exhaustif ni toutes les expériences anciennes.

Le témoin rapporté par le projet est **KL ≈ 0,1064 et top-1 ≈ 89,7 % sur 600 documents**, avec E8 + LDLQ. Cela ne prouve ni une taille réelle de 10 Go complets ni une comparaison équitable aux scores externes. Reprends les chiffres définitifs de tes propres artefacts, avec leur périmètre.

La V3.1 est une référence CPU et une proposition d’intégration. La calibration asymétrique a un signal synthétique ; BGCC et les polissages ne sont pas activés par défaut. Un mécanisme prévu dans la recette n’est pas présumé exécuté dans ton quantifieur GPU.

Détermine rapidement ce que ton pipeline fait réellement : poids quantifiés/protégés, données et statistiques, objectif d’arrondi, allocation, traitement tête/embeddings/Vision/MTP, apprentissage éventuel, format réellement exporté. Un court tableau relié au code suffit.

## 3. Références externes : comprendre avant de copier

Repères publiés dans le protocole Arki05, **textuels, hors MTP et sans validation Vision** :

| Référence | Taille déclarée | KL publiée |
|---|---:|---:|
| GSQ-RCO IQ2_S | 9,26 Go | 0,101 |
| Arki05 2,75 bpw | 9,26 Go | 0,0900 |
| Arki05 2,82 bpw | 9,48 Go | 0,0823 |
| Arki05 3,00 bpw | 10,09 Go | 0,0640 |

Le dernier point dépasse déjà le budget complet. Ces scores ne peuvent pas être comparés directement à 0,1064 sur tes documents.

Le dossier examine trois leviers documentés : **Arki05 combine compensation de type GPTQ et types par tenseur ; GSQ apprend les codes/échelles tandis que RCO choisit l’allocation ; Axolotl XQD calibre notamment la tête et les tokens d’assistant.** Les méthodes ne sont pas interchangeables. Les détails non accessibles restent inconnus : inspecte les configurations publiques et versions avant d’en attribuer un gain à un composant.

GSQ peut comporter de l’apprentissage local ; certaines expériences du papier utilisent une distillation globale des échelles. Ne qualifie pas tous ces modèles de « sans entraînement », ni tous les GGUF de « distillés » sans preuve. Arki05 ne décrit pas une distillation globale dans sa fiche ; XQD annonce ne pas faire de fine-tuning.

Sources de départ :
- https://huggingface.co/Arki05/Qwen3.8-27B-GGUF
- https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
- https://github.com/IST-DASLab/GSQ
- https://github.com/IST-DASLab/RCO
- https://huggingface.co/axolotl-ai-co/Qwen3.8-27B-GGUF

## 4. Objectifs chiffrés et vraie condition de réussite

**Objectif principal : battre le meilleur concurrent réellement mesuré au même budget pertinent et sous le même protocole.** Vise au moins **5 % de KL en moins**, puis **10 %** comme objectif renforcé. Ces marges sont des cibles choisies, pas des garanties. La différence doit être confirmée par une analyse appariée et accompagnée de garde-fous par domaine.

Sur le protocole Formic figé, les jalons internes sont :

| Jalon | KL moyenne, nat/token |
|---|---:|
| Première progression | ≤ 0,100 |
| Objectif central ambitieux | **≤ 0,080** |
| Objectif renforcé | ≤ 0,070 |
| Objectif très ambitieux | **≤ 0,060** |
| Exploration ultérieure | ≤ 0,050 |

Remesure le témoin sur tout nouvel ensemble. Ne prétends pas battre Arki05 en obtenant 0,08 sur un autre corpus. Si les seuils sont mal adaptés au protocole gelé, documente-le et privilégie le gain relatif comparable : **ne change pas les données, les masques ou les conditions pour embellir le score**.

« Meilleur au monde » est l’ambition, pas une conclusion prédéterminée. Une revendication finale doit nommer les concurrents, la date, le périmètre et les évaluations effectivement couverts. La KL ne remplace pas les capacités.

## 5. Ce qui rendra tes découvertes convaincantes

Commence par le plus petit test permettant de distinguer un écart de protocole d’un vrai défaut de compression. Choisis un concurrent fort proche du budget textuel disponible et, lorsque possible, un contrôle produit complet. Actualise brièvement les références accessibles : ce dossier n’est pas un inventaire exhaustif du marché. Actualise brièvement les références accessibles : ce dossier n’est pas un inventaire exhaustif du marché. N’épuise pas le budget à télécharger tous les modèles.

Sur les mêmes préfixes, utilise un professeur BF16 identifié, le même tokenizer, les mêmes positions et masques, et la même définition de `KL(P_BF16 || P_candidat)` en logarithme naturel. Documente contexte, moyenne, cache, runtime et précision numérique. Si les moteurs diffèrent, mesure des contrôles haute précision ; ne soustrais pas arbitrairement leurs KL. Une mesure top-K + queue n’est pas interchangeable avec le plein vocabulaire.

Sépare calibration, sélection et confirmation par documents/épisodes et vérifie les parentés entre jeux, pas seulement les doublons textuels. Les ensembles déjà consultés restent des données de développement. Gèle les choix avant la confirmation ; rapporte les différences appariées, leur incertitude, les domaines et les erreurs extrêmes.

Pour attribuer les gains, distingue architecture originale/candidate, recette de référence/candidate et effort d’apprentissage. Une baisse de KL grâce à plus de bits ou plus de récupération n’est pas automatiquement un gain architectural. Mesure à coût comparable ou publie la frontière taille–qualité et les coûts non égalisés.

Tu peux exploiter factorisations, rotations, partage, protection de sous-espaces, meilleure allocation, formats hybrides, optimisation de codes ou nouvelles cibles. **Le choix t’appartient.** Pour chaque finaliste, relie le mécanisme à une faiblesse observée et à une expérience réfutable ; ne cumule pas toutes les briques avant de comprendre leurs effets.

Aucun domaine n’est délibérément sacrifié. Vérifie séparément texte, langues, raisonnement/code/outils, long contexte, Vision et MTP selon les ressources. Une absence de test reste une limite. Évalue la taille et la qualité après export/décodage ; compte tout le package, sans confusion avec la VRAM. Ne pénalise pas artificiellement un concurrent en lui imposant des accessoires BF16 quand Formic peut les quantifier.

## 6. Ressources, résultats et clôture

Établis ton plan à partir du matériel, du temps, du crédit et de l’espace réellement disponibles. Les anciennes estimations ne sont pas une autorisation de dépenses. Aucune campagne lourde, nouveau pod ou téléchargement massif sans budget autorisé. Chronomètre préparation, recherche, apprentissage, codage et évaluation ; sauvegarde progressivement.

Ne bloque pas la recherche sur un artefact externe inaccessible : déclare précisément ce qui manque, utilise un autre contrôle pertinent et limite la conclusion. Conserve les résultats négatifs et n’affiche pas de gain inventé pour satisfaire un seuil.

Livre : **rapport décisionnel sourcé ; description des solutions ; diff de recette et d’architecture ; code/configurations ; données de résultats par document ; bilan d’octets et de calcul ; protocole et manifeste de reproduction**. Sépare les résultats sans apprentissage, après adaptation locale et après distillation globale.

La question finale est : **qu’avons-nous réellement amélioré, quel concurrent battons-nous dans des conditions comparables, et qu’est-ce qui manque encore pour livrer Formic-Max complet sous 10 Go ?**
