# Ce que l’infrastructure fait réellement

Tous les scripts acceptent `--help`. Ils appellent `formic_lab/cli.py` : les corrections se font dans le code partagé, pas dans 21 copies différentes.

| Script | Fonction disponible | Périmètre / limite |
|---|---|---|
| `check_environment.py` | CPU/RAM/disque, versions, GPU visible et mémoire disponible | Diagnostic ; ne décide pas d’un GPU à louer |
| `verify_checkpoint.py` | Lecture des en-têtes safetensors, formes/dtypes/offsets, cohérence index, hashes optionnels | Sans chargement des poids ; la provenance du manifeste reste à auditer |
| `inspect_architecture.py` | Arbre des modules chargés, dimensions, paramètres et stockage dédupliqué | Ne prouve pas qu’un chemin MTP/Vision s’exécute |
| `load_model.py` | Chargement local et un court forward | Backend HF écrit, non validé sur Qwen ici ; pas de fallback silencieux vers CPU/quantification |
| `prepare_data.py` | Validation des sources/splits/contenus ; tokenisation CPU optionnelle avec tokenizer local | Pas de corpus téléchargé ; pas de preuve automatique de décontamination sémantique |
| `collect_activations.py` | Hooks entrée/sortie de module sur forward réel, pleine largeur, batch 1, positions traçables | Collecteur texte sans padding ; exécution multiple du même module, états spéciaux et multimodal demandent adaptation |
| `analyze_redundancy.py` | Spectre et rang effectif d’une matrice NPZ | Diagnostic descriptif borné, pas preuve que des canaux peuvent être supprimés |
| `analyze_sensitivity.py` | Erreur réelle d’une projection W/Q sur X | Pas de Fisher global ni carte automatique des 600 groupes |
| `generate_architecture_candidates.py` | Développe une petite grille explicitement choisie par le chercheur | Ne cherche ni ne choisit la nouvelle architecture à sa place |
| `apply_architecture.py` | Substitution et sauvegarde des seuls modules modifiés | Exemples identité / faible rang / canaux SwiGLU ; plugins libres |
| `quantize_with_lumen.py` | Vrai E8/LDLQ + export/décodage LMAT pour une petite matrice complète | Oracle CPU lent, sans Hadamard automatique ; **pas L0 historique**, pas CUDA ni 27B |
| `recover_candidate.py` | Optimisation locale bornée sur X/Y, sélection d’un état, sauvegarde | CLI linéaire ; runner aussi MLP/structure locale ; pas de distillation globale ou QAT complet |
| `evaluate_fidelity.py` | KL exacte du vocabulaire fourni, accord top-1 | Sur logits explicitement alignés ; aucune conversion en points benchmark |
| `benchmark_candidates.py` | Exécution multi-variantes en mémoire avec un seul modèle de base ; captures/recovery/quantification optionnelles | « benchmark » signifie ici banc expérimental, pas suite officielle de capacités |
| `compare_architectures.py` | Différences appariées par document, bootstrap par source, coûts locaux | Identifie les écarts de taille ; n’annonce pas une victoire iso-octets si elle n’est pas établie |
| `monitor_experiments.py` | Logs, supervision et arrêt du groupe de processus | Coût estimé d’après les paramètres fournis ; **ne stoppe pas RunPod** |
| `save_results.py` | Archive + manifeste listant les exclusions | Captures/poids exclus par défaut ; pas de téléversement |
| `generate_report.py` | Rapport des mesures réellement terminées | Pas de texte scientifique inventé ni extrapolation des gains |
| `freeze_candidates.py` | Gel des identifiants et hashes avant confirmation | Garde méthodologique, pas barrière contre un opérateur qui modifie le code |
| `confirm_candidates.py` | Rechargement des poids locaux retenus et tests sur `confirm`, sans réapprentissage | Une confirmation déjà consommée ne doit pas devenir un nouveau jeu de sélection |
| `check_release.py` | Contrôle du manifeste de la livraison | Intégrité seulement |
| `run_selftest.py` | Tests CPU et fichiers locaux | Aucun GPU/Qwen vérifié |
| `run_demo.py` | Démonstration complète sur un petit modèle causal généré localement | Fixture synthétique ; la factorisation d’exemple peut être moins bonne et plus coûteuse |

## Fichiers et ressources

Aucun modèle n’est téléchargé. Aucun appel API IA/RunPod n’est exécuté. Le modèle original reste séparé des sorties. Les plugins sont du code Python de confiance exécuté avec les privilèges de l’utilisateur : **ce kit n’est pas une sandbox de sécurité**.

La référence Lumen est copiée sans modification dans `vendor/lumen_ref/`. Le ZIP source complet est conservé dans `reference/` avec son empreinte. `reference/validate_capture_v3_1.py` est aussi une copie inchangée, utilisée dans un test du raccordement de schéma.

L’option `scalar_proxy` est exécutable avec PyTorch, y compris sur CUDA après validation locale. Elle simule une quantification scalaire et sauvegarde des tenseurs denses : son rapport porte **`UNIFORM_SCALAR_PROXY_NOT_LUMEN`**. Elle ne valide ni la compacité ni les capacités de Lumen. Elle est absente des expériences de démonstration livrées.

## Limites importantes du backend standard

- Un seul modèle chargé. Professeur et candidat sont évalués séquentiellement par substitutions réversibles ; cela évite de présumer que deux BF16 de 55,6 Go tiennent en VRAM.
- Pas d’offload automatique ni de `trust_remote_code=True`. Un runtime spécifique peut être installé via une factory après lecture du code.
- Chaque document commence sans cache et utilise `use_cache=False`. Ce protocole est adapté à un pilote textuel court, pas à la mesure de génération avec états persistants et MTP.
- Le forward garde peu de logits lorsque `logits_to_keep` est explicitement disponible. Sinon, une limite empêche une allocation contexte × vocabulaire trop grande.
- L’état temporaire du modèle n’est pas un nouveau checkpoint complet. Les sauvegardes sont celles des modules changés et sont **denses** ; elles ne doivent pas être comptées comme un modèle produit à 10 Go.
- La référence E8 utilise un Gram dense et Python. Ses seuils bloquent les grosses matrices avant de consommer des heures : les lever sans portage/chronométrage serait risqué.

## Statut du matériel et des versions

Les tests de livraison ont utilisé un CPU, Python 3.13.5, NumPy 2.3.5 et PyTorch 2.10.0+cpu. Le contrat du chargeur HF a été testé par un mock, pas par chargement d’un Qwen. Ni Transformers ni Accelerate n’étaient disponibles pour une intégration réelle ici. Les informations de GPU et les appels CUDA du kit restent à exercer sur le pod.
