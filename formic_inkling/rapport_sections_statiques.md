## 1. Méthode de comparaison

* **Professeur.** Le checkpoint publié (`thinkingmachines/Inkling-2-Mini`, révision `b85c3cf`, SHA-256 identiques à ceux du dépôt)
  déquantifié exactement : W4A16, aucun `input_scale` dans le fichier. **Correctif du chargeur** : Transformers 5.24 arrondissait en
  BF16 le facteur FP32 par expert NVFP4 avant la déquantification (les 40 tenseurs NVFP4 correspondent exactement à cette erreur ;
  22,5 % des poids changent après correction). Le professeur corrigé a des experts NVFP4 **identiques bit à bit** à ceux de
  UD-Q8_K_XL d'Unsloth (déquantifiés en FP32 puis arrondis en BF16).
* **Graphe.** `Inkling2LoopForConditionalGeneration` de Transformers 5.24.0 (sans `trust_remote_code`), experts en `grouped_mm`.
  Professeur, Formic et concurrents passent dans **le même graphe**.
* **Concurrents exécutés dans notre graphe.** Les tenseurs quantifiés des GGUF d'Unsloth sont déquantifiés (lecteur d'en-têtes
  maison ; types K/IQ par `gguf` 0.19 ; NVFP4, Q2_0, F8_E4M3 par une implémentation maison écrite d'après la référence de
  llama.cpp) et écrits dans le modèle Transformers. L'échelle FP32 par expert (`ffn_*_exps.scale`) est appliquée. Le rangement est
  vérifié tenseur par tenseur avant toute mesure (cosinus, rapport des normes, égalité bit à bit). Les tenseurs F32 des GGUF
  (normes, convolutions, contrôleurs) ne sont pas chargés : ce sont ceux du professeur.
* **Protocole principal P-I** : 106 fenêtres de 4096 tokens lues depuis un contexte vide (`<|begin_of_text|>` en tête), positions
  2048–4095 scorées (217 088 positions), 9 domaines. KL(professeur ‖ candidat) sur les 200 058 entrées utiles du vocabulaire
  (l'unembed est chargé non rembourré), logits `30·tanh(z/30)` pour les deux, en nats, moyenne par token ; log-softmax par
  morceaux de 1024 positions ; professeur rejoué depuis ses états finaux mis en cache avec sa tête d'origine. Accord top-1 ; accord
  de routage (part des 8 experts communs et part des tokens dont l'expert top-1 change, pour chacune des 48 applications par
  token) ; IC 95 % appariés par bootstrap des fenêtres (méthode V7).
* **Moitiés.** Alternance par domaine : dev 54 fenêtres, conf 52 fenêtres, **scellée** (`/workspace/work_ink/sealed/`) jusqu'au gel.
* **Budget égal.** Octets texte de UD-IQ2_XXS **mesurés sur fichier** : 8 474 257 664 octets de tenseurs (métadonnées exclues).
  Formic : octets réels de ses fichiers texte (E8 + petits tenseurs), mesurés sur disque.
* **Disjonction.** Calibration et P-I disjointes par identifiant de document (intersection vide, `protocol_ink.json`) et par le
  filtre de 13 mots de `build_cal2.py` (bloc `tool_declare` exclu). P-W et P-D32 utilisent des plages d'indices distinctes.
* **Gel.** Règle de sélection écrite et commitée avant toute variante (`results/selection_rule.lock.json`, commit `9829a60`) ; choix
  figé avant lecture de la moitié conf (`results/selection.lock.json`).
* **KL publiées par Unsloth** (0,0726 / 0,0595 / 0,0269) : corpus, domaines et commande inconnus. Elles ne sont citées que comme
  repères « publié, protocole inconnu » et ne sont jamais comparées à nos chiffres.
