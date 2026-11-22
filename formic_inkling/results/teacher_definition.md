# Définition du professeur — FORMIC-Ink (écrite en C0, avant toute passe)

**Date : 22 novembre 2026, 10h55 UTC.** Checkpoint `thinkingmachines/Inkling-2-Mini`, révision `b85c3cfa615fc413cc9af3379f491b963f3b586a`.
SHA-256 mesurés, identiques aux SHA LFS du dépôt :
- `model-00001-of-00001.safetensors` : `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`
- `mtp.safetensors` : `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`

## Activations quantifiées ?
**Mesuré : aucun tenseur `input_scale`** parmi les 1 244 tenseurs du fichier principal (`checkpoint_headers_inkling.json`,
champ `input_scale_tensors = []`). `hf_quant_config.json` porte `required_activation_quantization: null` sur les groupes lus.
La question W4A4 ne se pose donc pas pour ce checkpoint.

## Définition retenue
**Professeur = W4A16** : poids NVFP4 (experts des couches 0–19) et MXFP8 (experts des couches 20–23) déquantifiés exactement,
tous les autres tenseurs tels que publiés (BF16 / FP32), activations en BF16 avec les calculs FP32 prévus par le graphe
(routeur, résidus d'attention, softcaps du MLP, convolutions). Graphe : `Inkling2LoopForConditionalGeneration` de
Transformers 5.24.0, sans `trust_remote_code`.

Déquantification (vérifiée dans le code de Transformers, `Inkling2LoopDequantizeExperts`, et dans les en-têtes) :
- NVFP4 : `w = e2m1(code) × fp8_e4m3(weight_scale du bloc de 16) × (weight_amax[e] / weight_outer_scale_denominator)`,
  deux codes par octet, **élément pair dans le quartet bas** ; `w13` stocké entrelacé (gate₀, up₀, gate₁, up₁…) et
  désentrelacé en `[gate ; up]` (`Interleave(dim=1)`), forme `[128, 2304, 2304]` ; `w2` `[128, 2304, 1152]`.
- MXFP8 : `w = fp8_e4m3(valeur) × 2^(scale_u8 − 127)` par bloc de 32.
- `unembed` : seules les 200 058 premières lignes sont chargées (`Inkling2UnpadVocab`) : les logits de remplissage
  n'existent pas dans le graphe ; la KL porte exactement sur les 200 058 entrées utiles.
- Tête : `logits = 30 · tanh(z / 30)` (softcap final), appliqué au professeur et au candidat lors du rejeu de la tête.

## Précision du professeur (règle déclarée d'avance, § 4.1.3 du mandat)
Le facteur FP32 par expert (`amax/denominator`) peut sortir de BF16. Règle : mesurer `KL(professeur_FP32 ‖ professeur_BF16)`
(experts déquantifiés en FP32 à la volée depuis les codes, contre experts arrondis en BF16 par Transformers) sur la moitié dev
de P-I. **Si ≤ 1·10⁻⁴, le professeur BF16 est accepté.** Sinon, les experts restent déquantifiés en FP32 à la volée.
Résultat : `teacher_check.json`.
