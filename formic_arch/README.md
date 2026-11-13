# formic_arch — recherche architecture × Lumen sur le vrai Qwen3.8-27B

Sprint du 13 novembre 2026 sur un pod RunPod A100 80 Go (2,11 $/h, renseigné par l'opérateur).
Conclusion et décision : **`RAPPORT_ARCHITECTURE_X_LUMEN.md`**. Ce dossier contient le code exécuté, les résultats bruts et les limites.

## Contenu

| Chemin | Rôle |
|---|---|
| `RAPPORT_ARCHITECTURE_X_LUMEN.md` | Rapport décisionnel (hypothèses, mesures, contrôle équitable, décision, suite) |
| `docs/GRAPHE_AVANT_APRES.md` | Graphe réel inspecté, inventaire, structures essayées (biais A_b, profondeur, largeur) |
| `docs/REPONSE_DOSSIER_PISTES.md` | Réponse mesurée, piste par piste, au `DOSSIER_PISTES_ARCHITECTURALES_LUMEN.md` ; organisation de la suite |
| `docs/SOURCES_PRIMAIRES.md` | Sources primaires, chaque entrée marquée VÉRIFIÉ / NON VÉRIFIÉ |
| `lab/formic_q.py` | Portage GPU (PyTorch) de `lumen_ref` : E8, sélection par candidats, LDLQ bloc-8, coût idéal du codec mode 0 |
| `lab/harness.py` | Banc : restauration BF16, professeur, KL exacte, statistiques (Fisher diag., moments), quantification du modèle, biais |
| `lab/seq.py` | Passe séquentielle couche par couche sur activations contextualisées : Grams, élagage de largeur + moindres carrés, passe E8-LDLQ complète |
| `lab/lab_server.py`, `lab/submit.sh` | Serveur d'expériences : un seul BF16 chargé en VRAM, tâches exécutées en file |
| `lab/budget.py`, `lab/summarize.py` | Budget du paquet complet (en-têtes réels) ; tableaux et bootstrap apparié par document |
| `lab/build_report.py`, `lab/fill_report.py`, `lab/*_TEMPLATE.md` | Régénération des tableaux du rapport à partir des fichiers bruts |
| `jobs/*.py` | Les tâches réellement exécutées, dans l'ordre |
| `tests/test_gpu_port.py` (+ `.log`) | Parité du portage GPU avec la référence CPU inchangée (7 contrôles, tous PASS) |
| `tests/test_codec_real_weights.py` | Encodage/décodage **physique** par le codec CPU de référence sur de vrais poids Qwen |
| `data/build_corpus.py`, `data/data_manifest.json` | Corpus cal / sélection / confirmation (144 / 72 / 72 documents × 1024 tokens, 6 domaines) |
| `data/build_long.py`, `data/build_big.py`, `data/vision/` | 12 documents × 8192 tokens ; 600 documents de comparaison ; 24 invites image+texte (scripts et manifestes) |
| `results/` | Résultats bruts : `variant_*.json` (sélection, par document), `big_*.json` (600 documents), `long_*.json`, `diag_*.json`, budget, verrous (`selection.lock.json`, `big_comparison.lock.json`), `summary_confirm.json` |

## Reproduire

```bash
pip install "transformers==5.22.2" safetensors accelerate          # PyTorch CUDA du pod conservé (2.8.0+cu128)
python3 -B formic_arch/tests/test_gpu_port.py                       # parité avec lumen_ref (≈15 s)
python3 -B formic_arch/data/build_corpus.py                         # corpus (réseau requis ; graine 1234)
nohup python3 -B formic_arch/lab/lab_server.py > server.log &       # charge le BF16 (≈13 s, 52,7 Go VRAM)
formic_arch/lab/submit.sh formic_arch/jobs/001_teacher.py           # puis les tâches dans l'ordre des numéros
python3 -B formic_arch/lab/summarize.py                             # tableau de sélection
python3 -B formic_arch/lab/summarize.py --unseal                    # confirmation, seulement après le verrou
```

Chemins codés en dur : modèle `/root/models/qwen3.8-27b`, travail `/workspace/work` (hors dépôt : logs, corpus tokenisé,
statistiques, résultats de confirmation scellés).

## Garde-fous de méthode

* Calibration, sélection, confirmation, contexte long et jeu de 600 documents sont des ensembles de documents disjoints (aucun `source_id` commun).
* Le témoin équitable est la passe **E8-LDLQ complète** ; l'arrondi sans rétroaction ne sert qu'au criblage.
* Sur 72 documents, une différence de KL n'est résolue qu'à ±15 % du témoin ; les effets plus fins sont tranchés sur 600 documents, variantes déclarées à l'avance.
* Toute transformation « exacte » est vérifiée numériquement dans le dtype réel avant quantification (elle ne l'est pas tout à fait en BF16).
* Chaque variante est évaluée sur la sélection (affichée) et sur la confirmation (**écrite dans un fichier scellé, jamais affichée**) ;
  les finalistes sont figés par un commit (`results/selection.lock.json`) avant toute lecture de la confirmation.
* Toutes les variantes quantifiées visent le même nombre total de bits sur le chemin texte (2,824 bpw idéal × 26,893 G poids,
  vecteurs d'échelle, biais et en-têtes compris) ; l'écart réel au budget est rapporté pour chacune.
* Les poids évalués sont les poids décodés (codes E8 × pas FP16), convertis en BF16 dans le modèle.
* Les octets sont un coût **idéal** du codec de référence (entropie des coquilles + rang + échappements) ; le surcoût physique
  mesuré sur de vrais poids est de 0,005 bpw (hors pas et en-tête), couvert par la marge de 0,009 bpw retenue.
* KL, accord top-1 et perplexité ne sont pas des pourcentages de capacités. Vision : une seule mesure diagnostique (24 invites) ; MTP et génération : non évalués.
