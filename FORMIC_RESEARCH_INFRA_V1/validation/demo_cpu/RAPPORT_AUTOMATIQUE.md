# Rapport automatisé — mesures, pas conclusions scientifiques

Périmètre : texte, préfixes imposés, positions échantillonnées. Les octets encodés concernent uniquement les matrices traitées. Les snapshots sont denses. Aucun package complet de 10 Go ni conservation des capacités générales n’est démontré.

| Variante | KL moyenne par document | Octets des tenseurs locaux encodés | Paramètres uniques chargés | Secondes |
|---|---:|---:|---:|---:|
| A0_full | 0 | 0 | 7328 | 0.00 |
| A0_local_E8 | 0.00011144 | 623 | 7328 | 0.03 |
| A1_example_rank8 | 0.000397062 | 0 | 7200 | 0.01 |
| A1_example_rank8_E8 | 0.000527942 | 998 | 7200 | 0.05 |

Sélection à faire par le chercheur sur `select`. Geler les identifiants avec `freeze`, puis seulement lancer `confirm`. Interpréter les différences à tailles et efforts comparables ; ne pas additionner les gains.
