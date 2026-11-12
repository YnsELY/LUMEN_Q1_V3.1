# File A → B V3 — Mesures minimales avant promotion de la recette

**Statut : à transmettre manuellement. Toutes les demandes sont `proposed`.** B conserve son planning, son plafond de ressources et son mandat de reconstruction N3. Ce fichier n’autorise aucune location, téléchargement ou prolongation.

## Préalable

Exécuter `tools/run_verification.py` sur une copie du dossier. Retrouver les identités de `L0_historique` et `S0_historique`. Le codec CPU est un oracle de correction, pas un remplaçant transparent du moteur. Si aucun raccordement au vrai modèle n’est prêt, ne pas lancer une campagne complète en reprenant l’ancienne estimation de 2,65 heures.

## REQ-A2B-V3-01 — Captures conformes et reprise d’un contrôle

**Question : les entrées du bloc qui sera quantifié sont-elles fidèlement capturées ?**

Réutiliser une couche déjà chargée par B, de préférence un MLP contextualisé. Capturer sa vraie entrée, les matrices ou sorties nécessaires et l’origine du préfixe. Conserver pleine largeur, biais, normes, masques, documents et états initiaux. Une projection modifiée partiellement conserve les contributions restantes avant la non-linéarité.

Produire calibration, sélection et confirmation distinctes, avec les mêmes historiques entre BF16 et élève pour les comparaisons. Fournir le manifeste, les empreintes et le code de capture. Aucun volume de 256 séquences ni nombre de quatre couches n’est imposé avant un chronométrage.

**Minimum utile :** un seul MLP réel et un contrôle numérique d’origine. **Coût : à mesurer.**

Livrer une capture NPZ documentée. `tools/validate_capture.py` peut vérifier son schéma ; il ne certifie pas sa provenance. B joint donc le code du hook, le chemin d’exécution et son journal.

## REQ-A2B-V3-02 — Une brique à la fois, après export

**Question : la cible asymétrique seule améliore-t-elle le contrôle corrigé sur le vrai bloc, sous un budget comparable ?**

Comparer la référence correctement exécutée, la cible asymétrique régularisée, puis éventuellement BGCC-CG ou BGCC discret si les entrées et les ressources sont prêtes. Ne pas commencer par cinq améliorations combinées. Documenter le budget de recherche des réglages et les différences de calcul.

Mesurer les octets réellement encodés du sous-système, la NMSE après down, puis la KL sur les mêmes préfixes après greffe lorsque c’est possible. Évaluer les poids reconstruits, les échelles stockées et les biais. Le contrôle dispose du même plafond et d’un effort pertinent comparable. Un résultat utilisant seulement le codec CPU reste une preuve de référence CPU, pas une mesure du moteur Lumen.

**Minimum utile :** un bloc, un débit, quelques réglages déclarés et une confirmation distincte. **Coût : premier chronométrage obligatoire.**

Le polissage à coquille constante reste désactivé : aucune relance automatique n’est demandée. Une hypothèse nouvelle et des données nouvelles sont nécessaires pour le rouvrir.

## REQ-A2B-V3-03 — Interfaces MTP/Vision et approximation KL

**Question : les interfaces nécessaires à Max sont-elles exécutables et les métriques correctement interprétées ?**

Décrire d’abord ce qui est réellement chargé pour MTP et Vision. Vérifier le chemin MTP, ses paramètres propres et ceux partagés. Mesurer son acceptation sous le protocole réel, pas son gradient dans un forward qui l’ignore. Pour Vision, utiliser de vraies entrées et la bonne forme de MLP. Un test image ne valide pas la vidéo.

Sur quelques positions présélectionnées, comparer KL complète et Top-K + queue avec les mêmes préfixes, la normalisation complète et l’arrondi du cache. Ne pas déclarer le proxy universellement sans biais.

**Minimum utile :** un audit exécutable d’une interface disponible ou un petit test de métrique ; pas l’obligation de tout réaliser. **Coût : à mesurer.**

Un éventuel duel QERA sur la tête reste secondaire : rang 64 INT8 = 16 220 160 octets bruts, plus auxiliaires. Le contrôle de précision reçoit le même budget réel ; `+0,08 bpw` n’est pas ce contrôle.

## Règles communes

Rapporter checkpoint/révision, code, architecture A0/A1, recette Q0/Q1, données, dtypes des états/caches, métriques, effets, coûts et limites. B peut accepter, réduire, refuser ou déclarer une demande bloquée. Un test à 4k tokens ne valide pas 262k ; conserver une qualification longue ultérieure même sans dérive courte.

Une greffe dans un reste BF16 ne démontre pas un modèle complet de 10 Go. Ne pas promouvoir une brique sur une meilleure MSE seule. Ne pas additionner les résultats de A et B ; pour A1, recalculer les statistiques dépendant du nouveau graphe.
