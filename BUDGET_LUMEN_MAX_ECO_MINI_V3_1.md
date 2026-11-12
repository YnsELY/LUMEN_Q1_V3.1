# Budgets LUMEN — révision 3.1

**Tous les budgets de modèle ci-dessous sont conditionnels à l’inventaire rapporté.** Les 18 en-têtes et les poids de la révision cible n’ont pas été relus ici. Aucun package complet n’a été produit. Les six fichiers CPU réels relèvent du codec de référence, pas du format GPU du projet.

## 1. Corrections

| Poste | Valeur / décision |
|---|---:|
| Auxiliaires du scénario de base Max/Éco | 84 066 560 octets, hypothèse héritée du format proposé |
| Auxiliaires du scénario de base Mini texte | **80 529 024 octets** |
| Différence qui était facturée à tort à Mini | **3 537 536 octets** |
| Réserve maximale d’alignement protégé Max/Éco | 86 233 octets |
| Réserve maximale d’alignement protégé Mini | 57 023 octets |
| Correcteur tête rang 64 INT8, données brutes | **16 220 160 octets** |
| Même correcteur, exemple de schéma par ligne décrit ci-dessous | **16 717 056 octets** |

Les bornes d’alignement sont `nombre_de_tenseurs × 127` pour un alignement 128 octets : elles ne sont pas des paddings mesurés. Elles sont désormais retranchées de la capacité, ce qui est conservateur.

## 2. Scénario 1, pas une allocation optimisée par les mesures

Formule :

```
capacité_payload_octets = plafond − protégés_bruts − réserve_alignement
                         − auxiliaires_supposés − garde_technique
bpw_modèle_symboles = 8 × capacité_payload_octets / paramètres_quantifiables
                     − surcoût_codage_supposé_bpw
```

Avec garde de 18 500 000 octets et surcoût **supposé** de 0,0025 bpw :

| Profil | Plafond complet | Capacité calculée du modèle de symboles |
|---|---:|---:|
| Max | 10 000 000 000 octets | **2,832825 bpw** |
| Éco | 9 000 000 000 octets | **2,544541 bpw** |
| Mini texte | 8 000 000 000 octets | **2,334235 bpw** |

Ces nombres ne sont ni le bpw global du fichier, ni une prévision de qualité, ni la sortie d’une allocation mesurée sur Qwen. Deux autres scénarios d’auxiliaires/réserve/surcoût sont conservés dans `results/budget_faits_et_scenarios.json`.

Les anciens « +623 Mo » ne sont pas réattribués à une amélioration de Q0. Les budgets texte/Vision/MTP doivent être évalués avec leurs qualités respectives, pas comme un transfert gratuit des octets vers le texte.

## 3. Correcteur de tête : unités et convention d’échelles

Pour `m=248320`, `n=5120`, `r=64`, facteurs INT8 :

```
poids_bruts = r*(m+n) = 16 220 160 octets
échelles_FP16_par_ligne = 2*(m+r) = 496 768 octets
descripteur_choisi = 128 octets
total_exemple = 16 717 056 octets
```

Rapporté aux `1 271 398 400` poids de la tête : brut **0,1020619 bpw**, exemple complet **0,1051885 bpw**. Ce schéma n’est pas un minimum universel : une autre granularité d’échelles possède un autre coût. Le contrôle doit recevoir les octets du format effectivement choisi et encodé, sans se contenter de `+0,08 bpw`.

## 4. Cache professeur et types variables

Le cache normal Top-128 avec valeurs FP16, IDs int32, queue FP32 vaut `128*2 + 128*4 + 4 = 772` octets par position, hors conteneur. La version corrigée peut promouvoir les valeurs non représentables vers FP32/FP64, ou les rejeter selon l’option. **Le comptage doit alors utiliser les dtypes et `actual_array_bytes` retournés**, pas systématiquement 772.

Un cache FP16 peut introduire un biais de classement à très faible écart ; l’oracle FP64 groupé sert de contrôle. La queue agrégée ne reconstitue pas une KL complète.

## 5. Ce qui est réellement encodé

`results/codec_*.bin`, les matrices dans `results/matched_budget/` et `results/demo_cpu/` sont des fichiers binaires CPU avec tailles et hashes réels. Les poids sont relus avant évaluation. Les dimensions des expériences sont petites ; ne pas extrapoler leurs bpw ou leurs frais fixes à 27B.

La DP de `allocate_options` est exacte pour les pertes additives et coûts entiers fournis. Le water-filling optimise une fonction hypothétique déterminée par ses entrées. Aucun de ces deux outils ne rend les sensibilités automatiquement mesurées ou les effets de groupes additifs.
