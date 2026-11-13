# Données, identité et coût

## Corpus JSONL minimal

Une ligne par document/épisode. `source_id` identifie le groupe à ne jamais répartir entre calibration/sélection/confirmation (document parent, dépôt ou épisode). Le champ `license` décrit la provenance et le droit d’usage **déclarés**, sans certification juridique automatique.

```json
{"id":"doc-1","source_id":"source-A","split":"cal","license":"texte original fourni par opérateur","text":"Texte de calibration..."}
{"id":"doc-2","source_id":"source-B","split":"select","license":"texte original fourni par opérateur","text":"Texte de sélection différent..."}
{"id":"doc-3","source_id":"source-C","split":"confirm","license":"texte original fourni par opérateur","text":"Texte réservé indépendant..."}
```

Ou utiliser `input_ids: [1, 2, 3, ...]` à la place de `text`. Les exemples ci-dessus sont un schéma, pas un corpus représentatif. Le kit refuse les IDs répétés, sources communes et contenus exactement dupliqués entre splits. Il ne détecte pas toutes les paraphrases ni toutes les contaminations par benchmarks.

Tokenisation locale avant la location GPU :

```bash
python -B scripts/prepare_data.py --input corpus.jsonl \
  --tokenizer /chemin/tokenizer_local --max-length 256 --out corpus_tokenise.json
```

Le chargement de tokenizer est local et ne charge pas les poids. Les tokens sont revalidés contre le vocabulaire du modèle pendant l’exécution. La longueur est tronquée explicitement ; ce pilote ne certifie pas le contexte long.

## Identité de checkpoint

`verify_checkpoint.py` vérifie headers, dtypes, formes, offsets, taille et cohérence de l’index. Avec `--full-hash`, il calcule aussi les SHA-256 des shards : cela lit tous leurs octets et peut prendre du temps.

Un manifeste **de provenance connue de l’opérateur** peut être fourni avec `--identity` :

```json
{"repo_id":"Qwen/Qwen3.8-27B","revision":"37561816e66b5b9153b1b64c10968c18c1e6f4a0","sha256":{"config.json":"sha256_reel","model.safetensors.index.json":"sha256_reel","model-00001-of-00018.safetensors":"sha256_reel_pour_chaque_shard"}}
```

Ce schéma doit lister tous les shards réellement chargés. Une empreinte que l’on vient de calculer sur un fichier ne prouve pas, seule, qu’il vient de l’éditeur officiel. Le nom de dossier `snapshots/<revision>` et `_commit_hash` sont enregistrés comme déclarations, pas comme provenance certifiée. Le chargeur refuse une révision contradictoire et un inventaire de paramètres différent de celui attendu. Son override diagnostique ne doit pas être utilisé pour présenter un autre modèle comme Qwen cible.

## Coût du pod

Le budget contient : début réel de facturation (ISO avec fuseau), solde disponible à ce moment, tarif total horaire connu, réserve et durée maximale depuis ce début. Les valeurs sont **à renseigner** ; aucun ancien prix RunPod n’est codé en dur.

Le coût estimé est `tarif × temps écoulé depuis le démarrage`. Cela inclut les temps sans GPU actif : préparation, réflexions et attentes si le pod reste allumé. Les éventuels frais non proportionnels, variations de tarif et dépenses d’autres pods/API ne sont pas connus du kit.

**Le moniteur termine uniquement son groupe de processus. Il ne stoppe ni ne supprime le pod, ne vide pas un solde, n’appelle aucune API RunPod.** Arrêter la location séparément. Garder une réserve suffisante pour le rapport, l’export et l’intervention humaine. Les limites sont des protections opérationnelles, pas une garantie financière.

## Sécurité opérationnelle

Les sorties doivent être hors du checkpoint. Aucun téléchargement, suppression de poids, installation distante, location ou envoi de données n’est automatique. Le kit n’enregistre pas les variables d’environnement ni les secrets. Ne pas passer de token d’API en clair dans une commande enregistrée par le superviseur.
