# Bilan de livraison — infrastructure FORMIC

**13 novembre 2026.** Cette livraison construit les outils de préparation du sprint, sans modifier la recette Lumen originale et sans chercher à imposer une nouvelle architecture.

## Livré

- **24 points d’entrée Python** dans `scripts/`, avec code partagé dans `formic_lab/`.
- Référence Lumen Q1-v3.1 inchangée : 8 fichiers de bibliothèque, ZIP original complet et validateur de capture source.
- Configuration de démonstration CPU ; configuration pod initiale limitée à la parité du runtime ; budget opérateur sans tarif ni solde inventés.
- Exemples de candidats/plugins, documentation d’extension, nouvelle version V6 du prompt et manifeste des fichiers.

## Vérifié lors de cette livraison

**87 tests d’infrastructure passent.** Ils couvrent les contrats de fichiers, en-têtes, sources et partitions, hooks, retrait des hooks, substitutions réversibles, interfaces plugins, factorisation exacte à plein rang, canaux couplés, petit encodage E8 réel, récupération locale, métriques, contrôle de budget, arrêt de processus, reprise, gel et confirmation.

Un test d’intégration exerce **20 commandes de la CLI** sur fichiers locaux et modèles miniatures. Le backend Transformers est testé **avec un mock pour ses arguments offline et garde-fous** ; ce n’est pas une validation du runtime Transformers réel.

Les **107 tests de la référence Lumen inchangée** ont également été relancés avec succès sur une copie temporaire de son archive ; ses 112 fichiers manifestés sont conformes.

La démonstration complète est enregistrée dans `validation/demo_cpu/`. Elle a effectivement écrit et relu des poids E8, sauvegardé des états locaux, comparé les logits et exécuté une confirmation après gel. L’architecture de factorisation fournie pour exercer les interfaces n’est pas recommandée scientifiquement : sur cette fixture elle n’est pas une amélioration. Les métriques de la démonstration ne sont pas des résultats Formic.

## Non testé / non livré comme solution complète

- Aucun poids Qwen téléchargé, chargé ou modifié ici ; aucune mesure de capacités Qwen.
- Aucun GPU/CUDA utilisé ; aucune durée A100/H100 garantie.
- Pas d’intégration réelle de Transformers/Accelerate dans cet environnement, où ces bibliothèques sont absentes.
- Pas de quantifieur GPU Lumen, de moteur d’inférence 27B, de distillation globale ni de package multimodal à 10 Go.
- Pas de collecteur multimodal/MTP/états persistants universel ; le collecteur standard est un pilote textuel sans padding, par document et sans cache.
- Pas d’arrêt ou de contrôle du compte RunPod. Le superviseur gère les processus et une estimation de dépense ; le pod reste à arrêter séparément.

## Ce qui est volontairement laissé à l’agent

La sélection des hypothèses et des architectures, leurs justifications, leur implémentation spécifique, le choix des observations, le portage si nécessaire, les contrôles d’équité et les conclusions scientifiques. Les scripts, limites initiales, exemples et interfaces peuvent être modifiés librement avec traçabilité et tests adaptés.

## Comment relancer les contrôles

```bash
python -B scripts/check_release.py
python -B scripts/run_selftest.py
python -B scripts/run_demo.py --out /chemin/hors_du_kit/demo_neuf --with-confirmation
```

Le manifeste atteste l’intégrité de la livraison. Les tests attestent les cas couverts. **Ni l’un ni les autres ne garantissent une absence de tout défaut ou un gain sur le modèle réel.**
