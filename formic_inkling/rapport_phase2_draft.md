### 11.1 Cadre

* Demande de l'opérateur à 16h25 (après le gel de phase 1, moitié conf de P-I déjà lue) : « trouver des solutions pour que la KL de
  Formic V1.5 soit plus basse », crédit ajouté.
* Règle écrite et commitée **avant** tout résultat (`results/selection_rule_phase2.lock.json`, commit `88bde90`) : candidats L0, V1a,
  V4, C1 = V1a + V4 (C1 seulement si V4 bat L0 sur dev) ; sélection sur P-I dev ; **confirmation sur P-I2**, jeu neuf scellé
  (48 fenêtres : 36 brutes code / math / encyclopédique / multilingue / long / web + 6 agentiques glaive + 6 instructions UltraChat
  test, réponses des jeux de données ; disjoint par identifiant ; gelé avant tout résultat, commit `1eea8e1`) ; **remplacement de
  Formic V1.5 seulement si le gain contre L0 sur P-I2 a un IC 95 % entièrement sous 0**.
* La moitié conf de P-I n'est plus aveugle pour la phase 2 : ses chiffres de phase 2 sont donnés à titre d'information seulement.

### 11.2 V4 — mesuré (P-I dev)

| | L0 | V4 |
|---|---:|---:|
| Multiplicateur experts | 0,757082 | 0,787197 |
| Multiplicateur attention, embed (et tête visée) | 0,757082 | 0,393599 (tête ajustée 0,365758) |
| bpw idéal : experts / attention / embed / unembed | 2,49 / 2,49 / 2,48 / 2,56 | 2,44 / 3,44 / 3,41 / 3,52 |
| Octets texte réels | 8 474 167 056 | 8 473 584 587 (−0,008 %) |
| KL P-I dev | 0,08024 | **0,06117** |
| top-1 / routage / top-1 expert changé | 91,61 % / 87,52 % / 15,20 % | 93,04 % / 89,33 % / 13,02 % |
| contre L0 [IC 95 %] | — | **−23,8 %** [−26,7 ; −20,9], 53/54 fenêtres |
| contre UD-IQ2_XXS [IC 95 %] | −6,9 % [−14,8 ; +0,7] | **−29,0 %** [−36,5 ; −21,9], 51/54 |

Par domaine (V4) : agentique 0,1389, code 0,0167, math 0,0121, raisonnement 0,1524, encyclopédique 0,0328, multilingue 0,0336,
long 0,0414, instructions 0,0924, web 0,0303 — tous meilleurs que L0. V4 est aussi en dessous de UD-Q2_K_XL (0,0712), plus gros de
652 Mo de texte. Lecture : ce que l'allocation uniforme de F1 coûte sur Inkling vient surtout de l'attention et du vocabulaire
(3,5 % des poids), que le cœur MoE de 24 G poids ne compense pas : retirer 0,05 bit aux experts pour donner +1 bit à ces chemins
paie massivement. C'est l'inverse du résultat de Qwen, où les allocations non uniformes essayées avaient perdu.
