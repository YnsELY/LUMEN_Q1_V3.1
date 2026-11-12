# FORMIC / LUMEN — Recherche autonome de co-conception architecture–compression

**Prompt V5 — Référence : LUMEN Q1-v3.1.** Remplace le mandat architectural V4 pour ce sprint.

## 1. Le projet et ton objectif

Formic vise à dériver du modèle préentraîné `Qwen/Qwen3.8-27B` une version beaucoup plus compacte, en préservant au maximum ses capacités générales. Lumen est notre recette de compression. Le premier objectif est **10 000 000 000 octets au maximum pour le modèle complet**, texte, Vision, MTP, poids transformés, correcteurs, tables, échelles et métadonnées nécessaires inclus.

Notre hypothèse est qu’une architecture dérivée pourrait mieux préserver les capacités sous compression Lumen que la structure BF16 originale. **Examine cette hypothèse sans la tenir pour acquise.**

Ta mission est de découvrir, concevoir et expérimenter des modifications architecturales qui fonctionnent particulièrement bien avec Lumen. Le résultat recherché est un meilleur couple **architecture + compression**, pas une architecture différente pour elle-même ni une accélération présentée comme un gain de capacités.

## 2. Ce dont tu disposes et où nous en sommes

Le pod contiendra le modèle BF16 complet et `LUMEN_Q1_V3_1_CORRIGE_ET_VERIFIE.zip`. La révision cible est `37561816e66b5b9153b1b64c10968c18c1e6f4a0` : vérifie l’identité des fichiers locaux, sans substitution silencieuse.

Lis le README, `RECETTE_MAX_PROPOSEE_V3_1.md`, le bilan V3.1 et les composants pertinents. Les anciennes versions ne remplacent pas les décisions actuelles.

Le projet Lumen articule quantification vectorielle E8, compensation d’erreur LDLQ, arrondi tenant compte des statistiques et du coût de codage, allocation de précision et récupération après compression. **Toutes ces composantes ne constituent pas encore un pipeline industriel intégré.**

La V3.1 fournit une référence CPU testée : algèbre, encodeur/décodeur et mesures. La calibration asymétrique régularisée montre un signal positif **synthétique, pas encore démontré sur Qwen**. BGCC et les polissages restent expérimentaux, désactivés par défaut. Le codec CPU n’est pas un kernel GPU ni le format produit définitif. Aucun Formic complet de 10 Go n’est encore validé.

Réutilise ces acquis sans refaire un audit exhaustif ni développer tout le moteur. Corrige toutefois un défaut qui invaliderait tes expériences.

## 3. Tu es responsable de la recherche

**Tu choisis les hypothèses, les mécanismes et les expériences.** Étudie le graphe exécuté, les poids et les activations. Consulte les sources primaires pertinentes, sans présumer que leurs résultats se transfèrent à notre modèle.

L’espace de recherche inclut les représentations équivalentes, les modifications de briques et **les reconstructions profondes**. Tu peux remettre en question dimensions, décompositions, partage, connexions, organisation des blocs ou mécanismes internes, en réutilisant autant que possible les poids préentraînés. Aucune architecture de remplacement n’est imposée.

**La reconstruction profonde exige une investigation réelle, pas une simple mention.** Réalise au moins un pilote structurel sur des blocs réels lorsque les ressources le permettent. Documente sinon le blocage concret, sans le présenter comme une impossibilité scientifique.

Pour chaque proposition, explique son intérêt pour Lumen, la réutilisation des poids, la récupération nécessaire et son critère d’abandon. Tu peux dépasser les pistes déjà suggérées. Privilégie quelques expériences approfondies plutôt qu’un catalogue superficiel.

## 4. Rechercher la synergie, sans figer une mauvaise solution

Évalue les structures **après quantification**, pas seulement en BF16. Analyse ensemble dommage structurel, erreurs de quantification, récupération et coût final. Des économies de paramètres peuvent être réinvesties en précision ; les facteurs, adaptations et conversions ajoutés doivent aussi être comptés.

Lumen est ton point de départ, pas un carcan. Adapte ses réglages ou une brique lorsque l’architecture le justifie, en isolant cette contribution : une meilleure compression seule n’est pas un gain architectural.

Recalcule les statistiques et adapte les interfaces après changement de base, dimensions ou états. Une transformation exacte exige preuve et vérification numérique. Une reconstruction approximative exige de mesurer dommage et récupération, pas une identité initiale au BF16.

## 5. Ce qui permettra de conclure

Compare à l’architecture originale avec la meilleure réalisation Lumen disponible, versionnée ; une référence historique absente reste non reproduite. Pas d’octets supplémentaires accordés au candidat, sauf courbe taille–qualité explicite. Rends comparables les efforts d’adaptation et distingue effets recette, architecture et combinaison.

Utilise des activations authentiques et contextualisées, sur préfixes communs. Sépare calibration, sélection et confirmation par documents/épisodes. Aucun benchmark final ni ses traces ne sert à sélectionner les candidats. Conserve les conditions de génération, sans davantage de tirages ou de réflexion.

N’abandonne délibérément aucun domaine : raisonnement, mathématiques, code, connaissances, langues, consignes, outils, agents, long contexte, vision et robustesse. Déclare les domaines et interfaces non testés. Une erreur locale plus faible ne prouve ni un gain de capacités ni un score de benchmark.

Distingue taille estimée, octets encodés, mémoire des poids et pic d’inférence. Évalue après export/décodage dès que possible. Un bloc greffé dans du BF16 n’est pas un modèle complet de 10 Go ; un codec de substitution ne valide pas Lumen.

## 6. Organisation et résultat attendu

Le GPU sera précisé séparément. Inspecte les ressources, chronomètre un essai et établis ton plan. Vise **10 à 15 heures maximum selon l’enveloppe autorisée**, rapport compris. Une récupération complète n’est pas présumée réalisable. Aucun dépassement ni ressource payante supplémentaire sans autorisation.

Travaille sur des copies, préserve les poids et résultats d’origine, sauvegarde progressivement et réserve du temps à la confirmation. Avance sans validation entre chaque étape dans les limites accordées.

Livre un dossier compact et reproductible : **rapport décisionnel, description des architectures effectivement examinées, code/configurations des prototypes, résultats bruts et bilan des coûts**. Identifie précisément ce qui a été publié, démontré, mesuré, estimé ou reste hypothétique.

La conclusion doit répondre : **quelle modification a réellement été essayée, pourquoi elle s’accorde avec Lumen, ce qu’elle gagne ou perd face au contrôle, et quelle intégration ou expérience suivante est justifiée.** Une conclusion négative solide est recevable. Une revendication de nouvelle architecture optimale sans preuve ne l’est pas.
