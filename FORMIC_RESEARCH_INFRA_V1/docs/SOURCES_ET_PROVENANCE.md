# Sources et provenance

## Base du projet

- `RECETTE_MAX_PROPOSEE_V3_1.md` et le bilan V3.1, dans l’archive Lumen fournie par le projet : statut CPU, limites GPU, calibration asymétrique, options BGCC et polissage, exigences de capture/confirmation.
- `PROMPT_AGENT_REVERSE_ENGINEERING_MAX_V5.md`, fourni précédemment : mandat autonome de co-conception, plafond de 10 Go complets et investigation structurelle réelle.
- L’archive d’origine Lumen est conservée dans `reference/`. Les empreintes des fichiers réutilisés sont dans `reference/PROVENANCE_VENDOR.json`. Aucune modification du quantifieur Lumen n’est faite dans cette livraison.

## Documentation primaire consultée le 13 novembre 2026

- [PyTorch 2.10 — Module, hooks et gestion des sous-modules](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html) : contrat des hooks ; ne valide pas les modules Qwen.
- [Transformers — chargement des modèles](https://huggingface.co/docs/transformers/main_classes/model) : chargement local, configuration, dtype et choix de runtime. L’agent doit épingler une version effectivement compatible.
- [Transformers — sorties de modèle](https://huggingface.co/docs/transformers/main_classes/output) : accès aux logits ; les détails du forward doivent être inspectés localement.
- [Accelerate — Big Model Inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) : placement mémoire/offload. Le kit ne l’active pas silencieusement et n’assimile pas ce chemin d’inférence à un entraînement complet.
- [Safetensors](https://huggingface.co/docs/safetensors/en/index) : conteneurs de tenseurs sans pickle ; pas preuve de provenance éditoriale.
- [RunPod — gestion des pods](https://docs.runpod.io/pods/manage-pods) : cycle de vie du pod ; le kit ne réalise aucune opération sur le compte.

Les tests et la démonstration décrivent le code livré, non des résultats publiés. Aucune nouvelle performance de Qwen, aucun prix actuel de GPU, aucune accélération du pipeline complet n’est revendiquée.
