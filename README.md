# LUMEN Q1-v3.1 — Référence CPU corrigée après contre-vérification

**12 novembre 2026.** Ce paquet remplace les outils CPU de la livraison Q1-v3. Il ne remplace ni Q0 dans le projet, ni un moteur d’inférence Qwen, ni le mandat architectural de l’agent B.

## À lire

| Fichier | Rôle |
|---|---|
| `BILAN_CONTRE_VERIFICATION_ET_CORRECTIONS_V3_1.md` | Verdict sur l’audit, erreurs confirmées, nuances et résultats nouveaux |
| `MATRICE_D01_D25_V3_1.md` | Réponse à chacun des 25 constats |
| `RECETTE_MAX_PROPOSEE_V3_1.md` | Intégration par composants et conditions de promotion |
| `FORMAT_REFERENCE_E8_V3_1.md` | Contrat binaire exact du codec CPU |
| `BUDGET_LUMEN_MAX_ECO_MINI_V3_1.md` | Budgets conditionnels et calculs d’unités |
| `CAPTURE_SCHEMA_V2.md` | Entrées attendues pour des captures authentiques |
| `DEMANDES_MESURES_POD_V3_1.md` | Pilote limité, sans détourner le chantier architectural |
| `PROMPT_REVERIFICATION_V3_1.md` | Mandat de nouvel audit indépendant |

## Démarrage

Python 3.11 ou supérieur est visé ; cette livraison a été exécutée sur **Python 3.13.5 / NumPy 2.3.5 / Linux CPU**. La compatibilité de cette révision sur Windows n’a pas été exécutée ici. Aucun GPU, téléchargement de poids ni accès réseau n’est requis par les tests. La dépendance est épinglée dans `requirements.txt`.

Depuis un environnement disposant de NumPy :

```bash
python -B /chemin/LUMEN_Q1_V3_1_CORRIGE/tools/check_release.py
```

Cette commande vérifie le manifeste, copie le paquet dans un répertoire temporaire, y exécute les tests et vérifie que le paquet source n’a pas changé. Elle ne modifie pas la livraison. Les caches Python `__pycache__` sont exclus du contrôle ; tous les autres fichiers, y compris les fichiers ajoutés, sont contrôlés. SHA-256 établit une identité, **pas l’authenticité d’un expéditeur ni la justesse scientifique**.

Vérification de l’intégrité seule :

```bash
python -B /chemin/LUMEN_Q1_V3_1_CORRIGE/tools/verify_manifest.py
```

Démonstration de bout en bout, avec sorties dans un autre dossier :

```bash
python -B /chemin/LUMEN_Q1_V3_1_CORRIGE/tools/demo_end_to_end.py --output-dir /chemin/sorties_demo
```

La démonstration est **synthétique** : calibration, quantification, fichier binaire, décodage, mesure et acceptation/rejet. Elle ne lance pas Qwen.

## Reproduire les expériences sans écraser les preuves livrées

```bash
python -B /chemin/LUMEN_Q1_V3_1_CORRIGE/tools/run_experiments.py --output-dir /chemin/sorties_experiences
```

Les cinq graines et la grille sont figées dans le script. Les choix utilisent `select`, jamais `confirm`. Les candidats revendiqués ne peuvent pas dépasser les octets réellement consommés par leur contrôle. Les temps comprennent la résolution de cible ; le calcul n’est pas égalisé entre méthodes.

`tools/run_verification.py`, `tools/check_cold_streams.py`, `budget_lumen_max_v3.py` et le script historique iso-coquille peuvent réécrire `results/` : **les exécuter sur une copie**. Le nom de `budget_lumen_max_v3.py` est conservé pour compatibilité ; son contenu est corrigé. Toute régénération change légitimement les métadonnées et parfois les durées : ne pas conserver un ancien manifeste comme s’il décrivait les nouveaux fichiers.

## Ce que le paquet contient réellement

- `lumen_ref/` : algèbre, calibration, quantification, codec, budget, métriques et garde-fous CPU.
- `tests/` : tests historiques et 49 tests de régression supplémentaires ; lecteur indépendant attribué au contre-auditeur.
- `results/` : exécutions de cette révision, flux binaires et expériences fraîches.
- `contre_audit_recu/` : les quatre pièces reçues, sans modification.
- `historique/` : documents et résultats antérieurs ; **ne pas les confondre avec les conclusions actuelles**.
- `originals/` : ancienne recherche V2, conservée notamment pour le contre-exemple de l’inverse.
- `correctifs/CORRECTION_CODE_V3_V3_1.patch` : différences de code par rapport à l’archive V3 complète, distinctes du patch reçu.

## État de preuve

**107 tests CPU réussis ; six flux historiques décodés à froid ; expériences synthétiques à taille contrôlée.** Aucun checkpoint Qwen complet compressé, aucun benchmark Formic et aucun kernel GPU ne sont livrés. Les défauts reproduits sont corrigés dans le périmètre testé ; cela n’est pas une garantie d’absence de tout défaut futur.

La priorité expérimentale est la cible asymétrique, isolée, avec vraies activations et contrôle fort. BGCC et polissage restent désactivés par défaut. Le gain moyen synthétique publié n’est ni un gain en points de benchmark ni un pourcentage de capacités conservées.
