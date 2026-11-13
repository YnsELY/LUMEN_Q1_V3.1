# FORMIC — Infrastructure de recherche architecture × Lumen

**Version 1.0 — 13 novembre 2026**  
**Référence de compression : LUMEN Q1-v3.1.** Ce paquet prépare le travail de l’agent ; il ne choisit pas l’architecture à sa place.

## Commencer ici

Transmettre à l’agent **ce dossier complet** et `PROMPT_AGENT_REVERSE_ENGINEERING_MAX_V6.md`. L’archive Lumen originale est déjà incluse dans `reference/`. Le BF16 doit être prépositionné séparément sur le pod.

La chaîne de démonstration est réellement exécutable sur CPU : modèle causal miniature → captures pleine largeur → substitution d’une projection → quantification E8 de petites matrices → fichier binaire → poids redécodés → fidélité → gel → confirmation. **Ce n’est pas un test Qwen, un résultat de capacité, ni une architecture recommandée.**

```bash
# À exécuter depuis n’importe quel dossier, en adaptant ces deux chemins.
KIT=/chemin/FORMIC_RESEARCH_INFRA_V1
WORK=/chemin/travail_formic        # hors du dossier des poids et hors du KIT

python -B "$KIT/scripts/check_release.py"
python -B "$KIT/scripts/run_selftest.py"
python -B "$KIT/scripts/run_demo.py" --out "$WORK/demo_001" --with-confirmation
```

Python 3.11+ est visé ; environnement effectivement testé : **Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, safetensors 0.7.0** sur Linux. Aucun modèle téléchargé et aucun GPU utilisé pour les tests de livraison. Voir `validation/`.

## Installation et dépendances

Sur un pod GPU, **conserver une installation PyTorch CUDA compatible avec son image et son pilote** : ne pas la remplacer par le wheel CPU de notre environnement. Le kit n’installe rien automatiquement.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Seulement si ces outils manquent ou si un environnement isolé a été choisi :
python -m pip install -r "$KIT/requirements-tools.txt"
```

Le backend `hf_local` demande en plus `transformers` et `accelerate` dans une version prenant réellement en charge le checkpoint local. **Ces deux bibliothèques n’étaient pas installées ici : leur contrat d’appel est testé avec un mock, pas leur intégration réelle à Qwen.** L’agent doit choisir/épingler les versions et vérifier le premier forward. Aucun script ne tente de télécharger automatiquement un modèle ou du code distant.

## Ce qui est fourni

- Des scripts simples partageant le même code dans `formic_lab/` : inspection, capture, diagnostics, substitutions, récupération locale, comparaison, reprise, budget et archivage.
- Un chargeur **offline** pour un modèle Transformers local ; une factory Python permet de brancher un autre runtime.
- Des exemples de modifications réversibles : identité, factorisation linéaire, sélection de canaux couplés d’un SwiGLU. **Ce sont des exemples techniques, pas des découvertes.**
- Un raccordement à la référence **Lumen CPU inchangée**, borné aux petites matrices. Les résultats distinguent fichiers E8, snapshots denses et mémoire d’exécution.
- Une séparation calibration / sélection / confirmation, un gel explicite des finalistes, des logs versionnés, une reprise vérifiant les artefacts.
- Un superviseur qui arrête un arbre de processus selon une durée et un coût **estimé** renseignés par l’opérateur.

La liste exacte et les limites de chaque script figurent dans `docs/CAPACITES_ET_LIMITES.md`.

## Premier passage sur le pod

1. Avant d’allumer le GPU si possible : lire l’archive Lumen, préparer le corpus, choisir les premières hypothèses et modifier la configuration. Le kit n’exécute pas d’agent IA et ne consomme aucune API de modèle.
2. Après démarrage : renseigner le vrai tarif et le vrai solde dans une copie de `configs/budget_operator_template.json`. Le tarif doit inclure les frais de fonctionnement connus. Le temps de facturation commence au démarrage du pod, **pas** au premier calcul.
3. Copier `configs/pod_template.json` hors du kit. Ajuster les chemins du checkpoint et du corpus, le runtime et le GPU. Cette configuration n’exécute initialement qu’un contrôle de parité, **aucune reconstruction prédéfinie**.
4. Faire `check_environment.py`, `verify_checkpoint.py`, puis `load_model.py` et `inspect_architecture.py`. Les en-têtes identifient les formes sans charger les poids. Un nom de dossier portant un commit n’est pas une preuve cryptographique de provenance : un manifeste de hashes de confiance permet un contrôle plus fort.
5. Choisir un petit pilote puis employer `benchmark_candidates.py`. Après choix et gel des variantes seulement, exécuter `confirm_candidates.py`.

```bash
python -B "$KIT/scripts/check_environment.py" --workdir "$WORK" --out "$WORK/environment.json"
python -B "$KIT/scripts/verify_checkpoint.py" --path /workspace/models/... \
  --out "$WORK/checkpoint_headers.json"
python -B "$KIT/scripts/load_model.py" --config "$WORK/pod.json" --out "$WORK/smoke.json"
python -B "$KIT/scripts/inspect_architecture.py" --config "$WORK/pod.json" --out "$WORK/architecture.json"

# Lancement supervisé après avoir rempli budget.json. Pas de tarif inventé par le kit.
python -B "$KIT/scripts/monitor_experiments.py" --budget "$WORK/budget.json" \
  --cwd "$WORK" --out "$WORK/supervision_001" -- \
  python -B "$KIT/scripts/benchmark_candidates.py" --config "$WORK/pod.json" --out "$WORK/run_001"

# Une reprise ne relance pas les variantes terminées et inchangées.
python -B "$KIT/scripts/benchmark_candidates.py" --config "$WORK/pod.json" --out "$WORK/run_001" --resume
```

**Attention : le superviseur arrête les scripts, pas le pod RunPod. La facturation GPU continue tant que le pod reste actif.** Il n’appelle aucune API de facturation/arrêt et ne garantit pas le solde réel. Arrêter le pod depuis RunPod, exporter les résultats au préalable et tenir compte du stockage restant. Prévoir une réserve. Le budget de l’agent IA est séparé.

## Confirmation et sauvegarde

```bash
python -B "$KIT/scripts/compare_architectures.py" --run "$WORK/run_001" \
  --control TEMOIN --candidate CANDIDAT --out "$WORK/comparaison.json"
python -B "$KIT/scripts/freeze_candidates.py" --run "$WORK/run_001" --ids TEMOIN CANDIDAT
python -B "$KIT/scripts/confirm_candidates.py" --run "$WORK/run_001"
python -B "$KIT/scripts/save_results.py" --run "$WORK/run_001" --dest "$WORK/resultats.tar.gz"
```

L’archivage exclut les tableaux et poids par défaut, et **énumère les exclusions**. Ajouter `--include-arrays` et une limite `--max-file-bytes` explicite pour inclure des captures/prototypes. Vérifier l’archive avant l’arrêt du pod. Les données privées restent locales.

## Liberté de recherche

L’agent peut modifier, remplacer, supprimer ou ajouter des fichiers et choisir d’autres outils. Le code ne doit pas le forcer à faire une SVD, à utiliser tel rang, ni à reproduire une expérience inutile. **Les interfaces sont un point de départ et non un cahier des charges scientifique.**

En cas de modification volontaire, conserver la version reçue, nommer la nouvelle branche, relancer les tests concernés et lancer un **nouveau** dossier d’expérience. Les hashes du run décrivent le code exécuté ; une reprise avec un code modifié est refusée pour ne pas mélanger les résultats. Un échec de l’ancien manifeste après une modification assumée est normal, mais ne doit pas être masqué.

## Ce qui reste à faire sur les vraies ressources

Compatibilité du runtime Qwen ; contrôle des captures contextualisées ; nouvelles architectures inventées par l’agent ; portage GPU de Lumen si nécessaire ; calibration/récupération à l’échelle réelle ; collecteurs Vision/MTP et tests de récurrence longue ; package complet sous 10 Go. La mémoire GPU, les vitesses et les gains du modèle réel ne sont **pas** validés par cette livraison.
