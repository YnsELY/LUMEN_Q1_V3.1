# Lumen Q1-v3-ref — Matrice de résolution de l’audit

Ce tableau ne signifie pas « toute la recette27B est validée ». Chaque point reçoit soit un correctif exécuté de référence, soit une spécification corrigée, soit un retrait de conclusion en attente de données. Les originaux restent inchangés. Voir les tests dans `tests/` et le journal `results/tests_final.log`.

| ID | Problème | Traitement | Preuve ou limite |
|---|---|---|---|
| C01 | Inverse triangulaire faux | Corrigé | inverse_unit_upper ; contre-exemple et tests aléatoires |
| C02 | Un estimateur nommé codeur/encoded_bpw | Corrigé dans la branche référence | Vrais octets rANS, rangs, ESC, décodeur et len(artifact) ; pas le codec produit |
| C03 | Codeur sans table, 314688 vecteurs et overhead minuscule non livrés | Affirmations retirées | Nouveaux flux et nouveaux effectifs seulement ; aucune revalidation de l’ancien test |
| C04 | Limite de coquilles contradictoire et clamp dans la queue | Corrigé | Limite explicite64 par défaut ; ESC256bits cohérent, int32 vérifié |
| C05 | Uniformité sphérique présumée optimale | Reformulé + testé | Mode uniforme témoin ; mode conditionné avec table comptée ; gains/pertes publiés |
| C06 | Coûts de profils globaux,32lanes,FP8 non simulés | Non revendiqués | Format CPU distinct ; intégration GPU/SG512 à réaliser |
| C07 | RMSNorm sur128 au lieu de5120 | Corrigé dans le contrat | rmsnorm_slice exige la moyenne carrée complète ; anciennes expériences déclassées |
| C08 | Embeddings utilisés comme entrées du MLP0/31 | Tests de transfert retirés | Captures contextualisées exigées ; données absentes, aucune réparation fictive |
| C09 | Sous-matrice sans contribution des colonnes restantes | Contrat corrigé | Avant non-linéarité, conserver la contribution complète ; pilote réel à faire |
| C10 | Pool calibration/validation concaténé | Protocole corrigé | Sources/documents/épisodes disjoints ; pas interdiction artificielle de vocabulaire commun |
| C11 | BGCC Gram et RHS incohérents | Corrigé | Objectif ridge explicite, équations normales et référence dense |
| C12 | BGCC quantifié avec une autre métrique sans le dire | Séparé et testé | Gram conditionnel retourné ; arrondi commun reste approximation déclarée ; raffinement sortie complète distinct |
| C13 | Ncal≥10n présenté comme condition démontrée | Retiré | Rang, diversité, conditionnement et validation mesurés ; aucun seuil universel |
| C14 | Grams par canal coûteux, temps extrapolés | Réalisation alternative fournie | Opérateur matrice-free CG et raffinement8×8 ; activations et calcul restent coûteux |
| C15 | BGCC porté à Vision en changeant une dérivée | Corrigé dans la recette | Vision séquentielle GELU : calibration affine asymétrique propre, pas gate×up |
| C16 | Fisher annoncé totalement sans biais | Requalifié | Covariance de u uᵀ correcte ; approximations Kron/temps/blocs restent approximations |
| C17 | Nombre de pas onde frontale et argmin général exact | Requalifié | Ordonnancement de dépendances et référence dense ; pas optimalité discrète globale |
| C18 | ABCD ni bidirectionnel ni garanti décroissant | Corrigé | Balayages inversés, incumbent, variation exacte objectif à table fixe |
| C19 | Pénalité V-step sans cardinalité de coquille | Corrigé | ShellModel.bits inclut shell, rang et ESC ; coût physique vérifié séparément |
| C20 | Rollback limité à V, snapshot partiel | Corrigé au niveau du garde-fou | Artefact complet, P et V évalués, sentinelles obligatoires ; boucle entraînement globale non livrée |
| C21 | Water-filling à sensibilités manuelles annoncé optimal pourQwen | Requalifié | ω historique marqué hypothèse ; optimum de modèle seulement |
| C22 | RCO spécifié comme déjà implémenté | Retiré | DP exacte pour tableau additif, RCO joint reste une recherche distincte |
| C23 | 623Mo annoncés comme progrès contreQ0 | Retiré | Différence contre format intermédiaire, pas contreQ0 déjà≈83Mo auxiliaires |
| C24 | Total10Go calculé annoncé final | Corrigé | Capacité conditionnelle vs coût réel des flux ; pas de package27B prétendu |
| C25 | BudgetQERA confond octets etbits | Corrigé et testé | 16220160o bruts,0.1020618557bpw avant auxiliaires ; duel+.08 abandonné |
| C26 | MTP supposé chargé mais simplement gradientnul | Corrigé dans le contrat | Vérifier présence des paramètres, branche et distribution ; chemin non réalisé ici |
| C27 | Cache topK ambigu et KL complète confondue | Corrigé et testé | Log-probas normalisées, int32, tail, arrondi déclaré ; borne pour agrégation exacte seulement |
| C28 | VRAM et durées2.65h/24.8h sans moteur | Retiré comme conclusion | Chronométrage de vrais kernels nécessaire ; temps CPU locaux sans extrapolationGPU |
| C29 | Pas de dérive4k utilisé pour exclure contexte long | Retiré | Sentinelle longue indépendante ; distribution des positions à contrôler |
| C30 | Toute baisse même minuscule qualifiée de régression assurée | Corrigé | Marges préenregistrées et incertitude appariée ; pas de compensation cachée entre domaines |
| C31 | Courbe dB/KL convertie en scores ou dominanceconcurrentielle | Interdit | Aucun score prédit ni classement produit déduit |
| C32 | SiLU tronque directement x à±20 | Corrigé supplémentaire | SiLU stable sans changer sa fonction, test x>20 |
| C33 | Cholesky répare silencieusement les pivots | Corrigé supplémentaire | Rejet SPD explicite et régularisation choisie publiquement |
| C34 | RéférenceL0 renommée après modification | Corrigé | L0_historique distinct deLref3 ; grilles et versions conservées |
| C35 | Absence de terme linéaire alors que le point a déjà changé | Nouvelle piste formalisée | Cible quadratique à gradient frais et rayon de confiance ; pas de gainQwen démontré |
| C36 | Améliorationcalibration assimilée à améliorationgénérale | Contre-exemple exécuté | Polissage iso-coquille : tailleconstante mais cinqrejets, données publiées |

## Ce qui ne peut pas être corrigé par un changement de texte

Les gains originaux sur les tranches BF16 doivent être remesurés avec l’inverse réparé, les bons états cachés et le codeur réellement utilisé. L’absence d’échantillon/captures empêche cette réexécution ici. Les nombres historiques restent dans `originals/`, explicitement non promus ; ils ne sont pas remplacés par des nombres synthétiques portant le même identifiant.

Le codec GPU et l’encodeur final du produit ne deviennent pas implémentés parce qu’un oracle CPU existe. Les quantificateurs/allocateurs corrigés doivent être raccordés aux vrais layouts, aux normes et au runtime. Toute incidence sur le matériel et les performances reste une mesure à produire.

## Limites identifiées par notre propre développement

Le premier plafond CG de160itérations ne satisfaisait pas `1e−8` ; le résultat était marqué non convergé. Une nouvelle exécution à512itérations maximum atteint le critère sur les cinq graines, sans changer les scores quantifiés. L’ancien JSON est archivé.

Un premier essai de très petites matrices avait un contrôle moins fortement ajusté. Il est conservé sous un nom exploratoire et n’alimente pas les conclusions principales. Le protocole principal emploie de nouvelles graines, un contrôle recherché et une confirmation séparée.

Une variante de polissage qui baisse sa perte d’apprentissage peut grossir le fichier. La variante iso-coquille corrige ce problème de taille, mais pas son sur-ajustement : les cinq essais réservés sont négatifs. Elle est donc désactivée par défaut. Aucun résultat défavorable n’a été supprimé pour fabriquer une solution gagnante.
