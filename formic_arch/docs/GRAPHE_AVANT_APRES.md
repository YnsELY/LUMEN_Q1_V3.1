# Graphe réel de Qwen3.8-27B (chemin texte) — avant / après

Source : en-têtes safetensors des 18 shards (`results/checkpoint_headers.json`) et code réellement exécuté
(`transformers 5.22.2`, `models/qwen3_5/modeling_qwen3_5.py`, classe `Qwen3_5ForConditionalGeneration`).
Statut : **mesuré** (inventaire, premier forward, parité chemin texte / enveloppe multimodale : écart max de logits 0,0).

## Inventaire (27 781 427 952 paramètres dans le checkpoint, 1199 tenseurs)

| Bloc | Tenseurs | Paramètres | Part |
|---|---|---:|---:|
| MLP SwiGLU ×64 : `gate_proj`, `up_proj` [17408×5120], `down_proj` [5120×17408] | 192 | 17,11 G | 61,6 % |
| Gated DeltaNet ×48 : `in_proj_qkv` [10240×5120], `in_proj_z` [6144×5120], `out_proj` [5120×6144], `in_proj_a/b` [48×5120], conv1d, norm | 336+ | 5,56 G | 20,0 % |
| Attention complète ×16 : `q_proj` [12288×5120] (requête + porte de sortie), `k_proj`/`v_proj` [1024×5120], `o_proj` [5120×6144] | 64+ | 1,68 G | 6,0 % |
| `embed_tokens` [248320×5120] | 1 | 1,27 G | 4,6 % |
| `lm_head` [248320×5120] (non lié) | 1 | 1,27 G | 4,6 % |
| Tour Vision (27 blocs, largeur 1152) + merger | — | 0,46 G | 1,7 % |
| MTP (1 couche + `fc` [5120×10240]) | 15 | 0,42 G | 1,5 % |

Le runtime HF charge 1184 tenseurs (27 356 728 560 paramètres) : **les 15 tenseurs `mtp.*` ne sont pas chargés ni exécutés**
par cette classe. Vision est chargée mais n'a pas été exercée (pas d'images dans le protocole).

## Une couche (64 couches, motif 3 × Gated DeltaNet puis 1 × attention complète)

```
h ──► RMSNorm(1+w) ──► mélangeur ──(+)──► RMSNorm(1+w) ──► SwiGLU ──(+)──► h'
 │                         ▲   │                              ▲   │
 └─────────────────────────┘   └──────────────────────────────┘   (résiduel)

mélangeur GDN  : x ─► in_proj_qkv ─► conv1d depthwise(k=4)+SiLU ─► q,k (l2-norm), v ─► règle delta à portes (état récurrent)
                 x ─► in_proj_a, in_proj_b (48 portes par tête : décroissance g, taux beta)
                 x ─► in_proj_z ─► porte SiLU appliquée après RMSNorm(128) de la sortie ─► out_proj ─► résiduel
mélangeur attn : x ─► q_proj (requête + porte sigmoïde de sortie), k_proj, v_proj (4 têtes KV, 24 têtes Q), RoPE partiel
                 sortie × sigmoid(porte) ─► o_proj ─► résiduel
SwiGLU         : down_proj( SiLU(gate_proj x) * up_proj x )
```

Aucun `nn.Linear` du chemin texte n'a de biais (497 linéaires sans biais).

## Fait mesuré déterminant : composante constante « massive » des activations

Sur le split de calibration (144 documents × 1024 tokens, poids BF16) :

* l'entrée normalisée des lecteurs du flux résiduel (`gate/up`, `in_proj_*`, `q/k/v`) est dominée par quelques dimensions :
  les 8 plus grandes portent 36 à 81 % du second moment `tr(H)` (rapport max/médiane jusqu'à 3,9·10⁴) ;
* cette énergie est presque entièrement **indépendante de l'entrée** : `|µ|²/tr(H)` = 0,36 (gate L30), 0,60 (in_proj_qkv L30),
  0,59 (q_proj L31) ; sur les 8 dimensions massives la part du carré de la moyenne est 94–97 %.
  (`results/diag_mean_share.json`, `results/diag_structure.json`)

C'est le phénomène publié des « massive activations » (Sun et al., 2024 ; voir `SOURCES_PRIMAIRES.md`), ici **mesuré** sur ce modèle.
Conséquence pour Lumen : un codec E8 à pas par ligne donne la même précision à la colonne d'une dimension massive qu'aux autres,
alors que l'erreur de cette colonne est multipliée par une entrée ~100× plus grande et **toujours de même signe** :
l'erreur de sortie correspondante est un décalage constant, donc exactement représentable par un biais.

## Architecture candidate A_b (seule modification retenue)

```
avant :  y = W x                      (497 linéaires sans biais)
après :  y = Q x + b ,  b = (W − Q) µ   (b : un vecteur FP16 par linéaire, 16 bits par ligne de sortie)
```

* `µ` = moyenne de l'entrée réelle du linéaire sur la calibration (entrées de l'élève dans la passe séquentielle).
* En BF16 (`Q = W`) le biais est nul : la modification est **exacte** tant que rien n'est quantifié ; après quantification,
  la sortie du linéaire est exacte sur la direction moyenne, et LDLQ ne travaille plus que sur la covariance centrée `H − µµᵀ`.
* Coût : 4,43 M valeurs FP16 ≈ 8,9 Mo (0,0026 bpw), **compté dans le budget** des variantes A_b (multiplicateur de pas relevé d'autant).
* Coût d'exécution : une addition de vecteur par linéaire ; mémoire +8,9 Mo.
* L'embedding (table de consultation) n'a pas de biais.

Les autres structures essayées (suppression de couches, réduction de largeur du SwiGLU, tête dérivée de l'embedding) sont décrites
dans le rapport ; aucune n'est retenue.
