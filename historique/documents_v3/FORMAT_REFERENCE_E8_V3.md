# Format CPU `LE8CPU3` / matrice `LMATv3` — spécification de référence

**Ce format n’est pas Slot-CP-rANS, ni une implémentation CUDA de Lumen.** Son rôle : fournir des octets réels, un décodeur indépendant et des oracles de test. Voir `lumen_ref/entropy.py`.

## 1. Codes et co-ensembles

Le code stocké est `u=2z` pour `z∈E8`. Les huit coordonnées sont entières, de même parité et leur somme est divisible par4. La coquille est `m=Σu_i²/8` ; `m=0` possède un point, les autres `240·Σ_{d|m}d³`.

Le rang suit un ordre déterministe : co-ensemble entier d’abord en mode0, puis demi-entier, puis ordre lexicographique des coordonnées admissibles. `e8.rank/unrank` réalise cette bijection par comptage dynamique. Il utilise de la mémoire cache CPU ; aucune affirmation de décodage GPU sans table n’est associée.

## 2. En-tête de flux : 48 octets little-endian

Structure Python `struct.Struct('<8sBBBBIQQQQ')` :

| Champ | Taille |
|---|---:|
| magic `LE8CPU3\0` | 8 |
| mode :0uniforme,1conditionné par co-ensemble | 1 |
| limite de coquilles, défaut64, maximum128 | 1 |
| précision de fréquence, défaut14, intervalle1..16 | 1 |
| réservé=0 | 1 |
| taille groupe de rangs, défaut128, maximum4096 | 4 |
| nombre de vecteurs | 8 |
| longueur flux rANS | 8 |
| longueur flux de rangs | 8 |
| longueur échappements | 8 |

Ensuite : table de fréquences entières `uint32`, flux rANS, flux de rangs, échappements, SHA-256 de32octets du corps complet précédent. La somme de contrôle détecte une corruption accidentelle ; elle n’authentifie pas un expéditeur.

Mode0 : symboles0..limit−1 pour les coquilles, puis ESC. Mode1 : `2m+parité`, puis ESC=`2limit`. Toutes les fréquences sont strictement positives ; le lissage et le coût des symboles impossibles éventuels sont donc visibles dans le modèle. L’arrondi déterministe des fréquences garantit une somme `2^precision`.

Avec limite64, la table occupe260octets en mode0 ou516en mode1. Ce sont des tables **par flux**, pas une revendication globale3,5Kio sur27B.

## 3. Flux rANS

Une lane CPU, état initial de normalisation `L=2^23`. L’encodeur parcourt les symboles en sens inverse ; le décodeur les restitue dans l’ordre. État final stocké en `uint32 little-endian`, puis octets de renormalisation dans l’ordre de lecture. Le décodeur exige l’état terminal `L` et la consommation exacte de la section, même pour une liste vide.

Un coût `−log2(f/2^precision)` est une longueur de modèle, **pas** la taille physique rANS. Le conteneur et l’état sont comptés séparément et réellement.

## 4. Flux de rangs

Pour chaque groupe de vecteurs non ESC, combiner les rangs `r_j` avec leurs radices `M_j` : `v←v·M_j+r_j`. Poser `P=∏M_j` ; le groupe occupe exactement `ceil(bit_length(P−1)/8)` octets big-endian, y compris les zéros de tête nécessaires. `P=1` n’émet aucun octet. Le décodeur obtient les radices des symboles de coquille déjà décodés, lit la longueur ainsi déterminée, puis inverse par divisions euclidiennes.

L’écart à `log2(P)` est inférieur à8bits par groupe. Ce résultat porte sur ce seul flux, pas sur tout le conteneur. Les grands entiers Python sont une réalisation de référence, pas une preuve de débit GPU.

## 5. Échappements

Une coquille hors limite émet ESC et huit int32 little-endian =32octets. Coût utilisé dans le modèle : coût du symbole ESC +256bits. Aucune saturation à la dernière coquille. L’encodeur refuse les coordonnées hors int32. Le décodeur valide l’appartenance à E8 et rejette un échappement non canonique qui aurait dû être dans les coquilles normales.

## 6. Conteneur de matrice : 24 octets d’en-tête

Structure `<8sIIQ` : magic `LMATv3\0\0`, lignes `m`, colonnes `n`, longueur du payload E8. Suivent `m` échelles FP16 little-endian, le payload `LE8CPU3`, puis un SHA-256 externe de32octets. `n` doit être multiple de8. Les poids reconstruits sont `W[i,j]=scale[i]·u[i,j]/2`.

Les échelles sont arrondies à FP16 **avant le choix des codes et les évaluations**. Zéros, NaN, infinis ou underflows non représentables sont refusés selon le contrat. Une ligne entièrement nulle reçoit une échelle1 et des codes0.

Aucun Hadamard, D_in, profil d’état, codebook, facteur FP8 ni bias n’est implicitement ajouté à ce conteneur. L’appelant doit déclarer les coordonnées de la matrice ; les expériences fournies sont en base canonique. Toute conversion requise par un moteur complet doit être sérialisée et budgétée.

## 7. Invariance de longueur à coquilles constantes

À modèle de fréquence, mode, groupe, échelles et dimensions fixes, conserver chaque coquille (et chaque co-ensemble en mode1) conserve les symboles rANS et les radices. L’état/flux rANS reste identique, les mots de rang gardent leur longueur, ESC garde32octets, les checksums gardent32octets. **La longueur totale est donc exactement conservée.**

Les rangs et les poids changent ; la qualité peut empirer hors calibration. Cette propriété n’est pas vraie en général pour un autre codec dont les probabilités dépendent du voisinage ou pour des échelles réoptimisées.

## 8. Périmètre non implémenté

Pas de multi-lane CUDA, pas d’indexation aléatoire complète des embeddings, pas de packager18shards, pas de tables globales14profils, pas de compression des activations ni du cache. Limites de ressources des décodeurs à configurer pour un usage réel. Tests de corruption et de formes fournis, sans prétention de certification de sécurité exhaustive.
