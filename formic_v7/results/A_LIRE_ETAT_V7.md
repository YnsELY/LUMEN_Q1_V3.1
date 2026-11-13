# État de la campagne V7 au 13 novembre 2026, 15:15 UTC — interrompue (limite d'usage atteinte)

Fait :
- `docs/AUDIT_PIPELINE.md` : audit de ce que le pipeline GPU exécute réellement, et écarts visibles avec Arki05 2,82 bpw.
- Arki05 2,82 bpw téléchargé (9,478 Go, rév. `cdcbea9996`), en-tête et allocation par tenseur lus.
- Protocole public Arki05 (108 fenêtres redistribuées sur 128) implémenté ; cache professeur BF16 construit ; auto-contrôle BF16 : KL 0.
- Chargement des poids concurrents dans notre graphe (`lab/v7.py`, `lab/gguf_dequant.py`).

**NON VALIDE — ne pas utiliser :** `competitor_Arki05_2.82bpw.json`, s'il existe. La correspondance des tenseurs est fausse pour
`attn_qkv` (cosinus ≈ 0,40 avec le BF16 au lieu de ≈ 0,94 : l'ordre des blocs q/k/v du GGUF diffère de `in_proj_qkv`) et pour au
moins un autre tenseur (cosinus minimal 0,04, probablement `ssm_alpha` / `ssm_beta`). Les KL mesurées sur ces poids ne décrivent pas le concurrent.

Reste à faire, dans l'ordre :
1. Corriger la correspondance (`_target` dans `lab/v7.py`) : retrouver la permutation de lignes de `attn_qkv` et l'appariement
   `ssm_alpha`/`ssm_beta` ↔ `in_proj_a`/`in_proj_b` par corrélation avec le BF16 ; exiger un cosinus > 0,9 sur tous les tenseurs.
2. Mesurer Arki05 2,82 bpw sur nos 600 documents et sur P-A108 ; comparer P-A108 à leur 0,0823 publié (128 fenêtres).
3. Ré-exécuter notre témoin E8-LDLQ et le mesurer sur P-A108 : premier chiffre comparable à budget texte égal (9,48 Go).
4. Transplanter leur allocation par tenseur dans notre quantifieur pour séparer allocation et quantifieur.
