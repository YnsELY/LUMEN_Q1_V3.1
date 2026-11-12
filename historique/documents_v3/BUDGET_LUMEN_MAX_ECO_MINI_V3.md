# Budget LUMEN Max / Éco / Mini — calculs corrigés V3

**Il n’existe pas de package Qwen V3 encodé dans cette livraison.** Le tableau est dérivé de l’inventaire rapporté, sans relecture des 18 en-têtes. Les grands groupes de paramètres protégés ne fournissent pas leurs alignements individuels exacts : le script indique une borne conservative supplémentaire, pas un total physique certifié.

## Plafonds et inventaire

| Profil | Plafond complet applicable | Paramètres uniques rapportés | Paramètres quantifiables rapportés | Paramètres protégés, octets BF16 bruts | Plafond effectif bits/paramètre |
|---|---:|---:|---:|---:|---:|
| Max | 10 000 000 000 | 27 781 427 952 | 27 750 330 368 | 62 195 168 | 2.87962160 |
| Eco | 9 000 000 000 | 27 781 427 952 | 27 750 330 368 | 62 195 168 | 2.59165944 |
| Mini_text | 8 000 000 000 | 26 895 998 464 | 26 869 760 000 | 52 476 928 | 2.37953613 |

Mini signifie ici texte seul, sans Vision et sans MTP. Ce n’est pas le même périmètre que Max. Éco et Mini exigeraient leurs propres allocations, réglages et validations, pas une règle de trois sur les scores.

## Scénarios de capacité, pas résultats encodés

La capacité disponible pour les symboles est calculée après déduction des octets protégés, auxiliaires, garde et overhead supposé. L’overhead n’est pas une mesure du codec CPU transposée au modèle. Les mêmes scénarios servent à montrer la sensibilité du budget, pas à certifier des choix matériels.

| Profil | Auxiliaires supposés (Mo) | Garde (Mo) | Overhead supposé (bpw) | Débit modèle de symboles disponible (bpw) |
|---|---:|---:|---:|---:|
| Max | 84.067 | 18.5 | 0.0025 | 2.83285025 |
| Max | 100.000 | 50.0 | 0.0200 | 2.80167591 |
| Max | 150.000 | 100.0 | 0.0500 | 2.74284742 |
| Eco | 84.067 | 18.5 | 0.0025 | 2.54456539 |
| Eco | 100.000 | 50.0 | 0.0200 | 2.51339105 |
| Eco | 150.000 | 100.0 | 0.0500 | 2.45456257 |
| Mini_text | 84.067 | 18.5 | 0.0025 | 2.33319828 |
| Mini_text | 100.000 | 50.0 | 0.0200 | 2.30157580 |
| Mini_text | 150.000 | 100.0 | 0.0500 | 2.24180255 |

Aucune allocation familiale de qualité n’est présentée comme mesurée. Les anciennes sensibilités `omega` sont conservées dans le fichier source avec leur statut d’a priori. Pour fermer le budget, utiliser les coûts entiers de fichiers effectivement encodés et leur inventaire complet.

## Calculs corrigés vérifiables

- Correcteur de tête rang64 INT8 : `64 × (248320 + 5120) = 16 220 160` octets, avant ses échelles et en-têtes. Sur `248320 × 5120` poids, cela vaut `0,1020618557 bpw`, pas `0,0128`. Le comparateur `+0,08 bpw` est trop petit.
- Cache Top-128 : `128 × (2 + 4) + 4 = 772` octets par position, hors métadonnées. Pour 4 194 304 positions : `3 238 002 688` octets. Avec un cache Top32 MTP analogue : `4 060 086 272` octets.
- Tous les Grams FP32 par canal pour un up `[17408,5120]` : `17408 × 5120² × 4 = 1 825 361 100 800` octets. Ce stockage est évité par la référence matrice-free ; l’activation et le calcul restent à payer.
- Un Gram local8×8 en float64 :512octets. Cela ne représente pas toute la mémoire d’une expérience de raffinement.

Le gain historique623Mo comparait un format intermédiaire lourd, pas Q0 déjà prévu autour de83Mo d’auxiliaires. Il est retiré des progrès marginaux revendiqués.

## Reproduction

`python budget_lumen_max_v3.py` produit le JSON correspondant. `continuous_waterfill` résout un modèle de distorsion sous hypothèses ; `allocate_options` traite un tableau additif fini à coûts entiers ; `account_files` compte et hache les fichiers physiques. Aucun des trois ne démontre la préservation des capacités sans évaluation séparée.
