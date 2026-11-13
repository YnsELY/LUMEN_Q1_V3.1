# Guide d’extension pour l’agent

## Tu peux tout adapter

Ce kit est du temps de préparation économisé, **pas une prescription de recherche**. Il est légitime de modifier la collecte, changer de backend, remplacer le moteur de quantification, ajouter des candidats ou supprimer une étape inutile. Conserver les versions et les résultats d’origine ; identifier la branche expérimentale ; réexécuter les tests touchés. Ne pas modifier une ancienne mesure pour rendre un candidat gagnant.

Le répertoire `examples/` contient des factories exécutables et des grilles explicites. Il ne contient aucune proposition de nouvelle architecture reconnue meilleure.

## Trois interfaces remplaçables

### Backend

Une configuration peut déclarer :

```json
{"backend":"plugin", "factory":"/workspace/mes_outils/runtime.py:build"}
```

`build(config)` retourne `formic_lab.models.Bundle(model, tokenizer, metadata, ...)`. Le bundle peut fournir `forward_fn(bundle, token_ids, keep_last)`. Ses métadonnées doivent au minimum donner `synthetic`, `checkpoint`, `revision`, `vocab_size`. Voir `examples/custom_backend.py`.

Pour le multimodal, les états récurrents ou MTP, **étendre aussi le contrat de données et de forward**. Ne pas forcer de faux tokens textuels dans un protocole qui ne représente pas l’expérience.

### Architecture

```json
{"kind":"plugin", "target":"chemin.reel.du.module", "factory":"/workspace/mes_outils/architecture.py:build"}
```

La factory reçoit une copie du module ciblé et la spécification. Elle retourne un `torch.nn.Module`. Le parent du modèle est remplacé temporairement puis restauré même sur exception. Tout état nécessaire à la reconstitution doit appartenir à `state_dict()` ou à la spécification versionnée. Une architecture couvrant plusieurs blocs peut remplacer leur parent commun ou demander une nouvelle transaction : les remplacements imbriqués ambigus sont refusés par le helper courant.

### Quantification

Une entrée de `quantization` peut contenir :

```json
{"target":"chemin.reel", "kind":"plugin", "factory":"/workspace/mes_outils/quant_gpu.py:quantize", "options":{}}
```

`quantize(module_copy, X_cal, output_dir, options)` retourne `(new_module, report_dict)`. `X_cal` est un tableau `[positions, largeur complète]` capturé après les modifications déjà actives. Le rapport doit préciser méthode, coûts réels ou estimés, dtypes, transformations, portée et limites. Les matrices sont traitées dans l’ordre de la configuration : la suivante voit les précédentes déjà modifiées. Tout export doit être décodé et comparé avant revendication de compacité/fidélité.

## Ajouter une hypothèse sans modifier le cœur

Le candidat à tester est un objet `variants[]` avec `id`, `changes[]`, éventuellement `recovery` et `quantization[]`. Un objet vide `changes: []` fournit un témoin original. Le témoin quantifié et le candidat quantifié doivent avoir des scopes et des budgets comparables. La structure d’exemple low-rank à rang 8 n’est pas une recommandation pour Qwen.

Les modules ciblés proviennent d’`inspect_architecture.py`. **Ne pas copier les chemins `blocks.0...` de la fixture dans Qwen sans vérifier le graphe réel.**

## Captures et raccordement Lumen

Les captures enregistrent `X`, `Y`, positions, documents, sources, hash des tokens, chemin exact, largeur et identité du modèle. Les valeurs ne sont pas des embeddings substitués aux activations profondes : elles viennent du hook du module effectivement exécuté.

`pair_captures(reference, student)` vérifie l’alignement des exemples. `export_lumen_schema2(...)` rassemble des paires `cal/select/confirm` et peut produire un `input_pair` ou une capture `linear` compatible avec le validateur Lumen fourni. Cette fonction ne prouve ni l’authenticité du forward ni les invariances d’une architecture modifiée. Recalculer les captures après changements de dimensions/bases ; ne pas forcer des formes incompatibles.

Les chemins Vision, MTP, états récurrents et séquences avec plusieurs appels du même module nécessitent un collecteur explicite. Le collecteur standard échoue au lieu d’enregistrer des positions ambiguës.

## Reprise et gel

La reprise exige code, plugins, configuration, données, modèle et artefacts inchangés. Un candidat terminé n’est pas réexécuté ; un candidat interrompu repart dans un nouveau dossier d’essai. Ce n’est pas une reprise à l’instruction près d’un entraînement.

Une modification de code → **nouveau run**. `freeze_candidates.py` fige les candidats choisis ; `confirm_candidates.py` recharge leurs états, sans refaire la sélection ni réentraîner. Si la confirmation inspire une nouvelle variante, ce jeu devient du développement ; fournir un autre jeu réservé.

## Aucun classement scientifique automatique

La faible erreur d’une matrice ou d’un bloc n’établit pas la conservation des capacités. Le rapport automatique ne décide pas du gagnant. Les temps sont des temps de forward imposé, pas des tokens/s de génération. Les intervalles resamplent les sources/épisodes, pas chaque token comme s’il était indépendant.

L’agent reste responsable de la nouveauté, des hypothèses, des contrôles, de l’équité des octets/adaptations, des domaines non testés et du choix de l’expérience suivante.

### Persistance des modules de quantification personnalisés

Le plugin peut retourner un `nn.Linear` de mêmes dimensions, avec les poids redécodés : le runner sait le reconstituer. S’il retourne un autre type (module packed, buffers particuliers, kernel personnalisé), fournir aussi `restore_factory: "/.../quant_gpu.py:restore"`. Cette fonction reçoit `(module_original_copy, artifact_directory, options, quantization_report)` et retourne le bon module, sans entraînement et sans consultation du jeu de confirmation. Le runner recharge ensuite son `state_dict` sauvegardé avec `strict=True`. Le code de restauration est lui aussi empreinté et copié dans le snapshot d’expérience.

Un plugin qui a besoin de plusieurs fichiers ou de transformations non persistantes doit les sérialiser explicitement. Sans constructeur de restauration, un module custom n’est pas promu comme un candidat rechargé à froid.
