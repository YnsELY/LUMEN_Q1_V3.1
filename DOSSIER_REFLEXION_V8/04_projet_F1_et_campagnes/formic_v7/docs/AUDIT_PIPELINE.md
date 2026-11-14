# Audit de maturité — ce que le pipeline GPU exécute réellement (témoin `A0_Q0ldlq`)

**13 novembre 2026.** Répond au §2 du mandat V7 et au tableau de questions du §6 du dossier concurrents.
Chaque ligne renvoie au code exécuté (`formic_arch/lab/`), pas à la recette V3.1.

| Question | Réponse factuelle | Code |
|---|---|---|
| Quels poids sont quantifiés ? | 498 matrices du chemin texte, 26,893 G poids : les 6–7 linéaires de chaque couche (dont `in_proj_a/b` 48×5120), `embed_tokens`, `lm_head`. | `harness.qmats` |
| Quels poids restent BF16 dans le banc ? | Gains des RMSNorm, `conv1d` (2,0 M), `A_log`, `dt_bias`, normes internes : 3,1 M poids. **Toute la tour Vision (460 M)**. **MTP (425 M) n'est pas chargé.** | `harness.qmats`, `lab_server.py` |
| Que compte le « 2,821 bpw » ? | Coût **idéal** du codec de référence mode 0 sur ces 498 matrices : entropie des coquilles + rang dans la coquille + échappements (256 bits) + pas FP16 par ligne + 656 octets par matrice. Dénominateur : 26,893 G. Ni Vision, ni MTP, ni tokenizer, ni conteneur. Soit 9,484 Go idéaux pour le texte. | `formic_q.matrix_bits`, `budget.py` |
| Un fichier a-t-il été écrit ? | **Non.** Encodage physique vérifié sur 7 extraits de ≤ 330 k poids (aller-retour exact, +0,005 bpw). Aucun paquet, aucun décodeur à l'échelle. | `tests/test_codec_real_weights.py` |
| L'allocation varie-t-elle par tenseur ou par profondeur ? | **Non : uniforme.** Un seul multiplicateur de pas global ; chaque matrice sort à 2,818–2,823 bpw (2,84 pour `in_proj_a/b`). | `seq.ldlq_pass`, `harness._scales` (règle `lumen`) |
| Quel pas ? | Par ligne de sortie : multiplicateur × RMS de la ligne, arrondi FP16. Pas d'échelle par colonne. | `formic_q.row_steps` |
| Quel objectif optimise LDLQ ? | L'erreur de **sortie locale du linéaire** `tr((W−Q) H (W−Q)ᵀ)`, `H = E[xxᵀ]` de l'entrée réelle du linéaire, amortissement 1 % de la diagonale moyenne. Ni bloc, ni logits, ni métrique de sortie. | `formic_q.ldl_prepare`, `ldlq_rows` |
| Cible symétrique ou asymétrique ? | **Symétrique** : entrées de l'élève (préfixe déjà quantifié), cible `W·x_élève`. La cible asymétrique de la recette (`W·x_professeur`) n'est pas exécutée. | `seq.Seq.gram`, `seq.ldlq_pass` |
| Arrondi | E8 par blocs de 8 colonnes, 7 candidats (zéro + 3 rétrécissements × 2 cosets), choix par la métrique 8×8 du bloc. `rate_lambda = 0` : aucun terme de débit. | `formic_q.e8_candidates`, `ldlq_rows` |
| Embeddings | E8 au plus proche **sans rétroaction** (table de consultation, pas de Gram). Même débit que le reste. | `seq.ldlq_pass` (bloc `-1`) |
| Tête de sortie | LDLQ avec le Gram des états finaux normalisés de l'élève ; même débit que le reste (2,82 bpw) ; pas de statistique propre à la sortie, pas de correcteur. | `seq.ldlq_pass` (bloc `64`) |
| Données de calibration | 144 documents × 1024 tokens, 6 domaines (prose, 7 langues, code Python, math, chat, science), texte brut sans `<|endoftext|>` initial. Aucune réponse générée par le BF16, aucun appel d'outil. | `data/build_corpus.py` |
| Apprentissage | **Aucun.** Ni optimisation de codes, ni d'échelles, ni distillation. Une seule passe. | — |
| BGCC, polissages, Hadamard | Non exécutés. | — |
| Mesure | Poids décodés (codes × pas) convertis en BF16 et exécutés dans le graphe HF ; KL plein vocabulaire à toutes les positions. | `harness.evaluate` |
| Captures | Entrées réelles de chaque linéaire, préfixe quantifié, vérifiées bit à bit contre le forward. | `jobs/030b_seq_parity.py` |

## Écarts visibles avec les concurrents avant toute mesure commune

| Point | Témoin Formic | Arki05 2,82 bpw (en-tête GGUF lu, rév. `cdcbea9996`) |
|---|---|---|
| Budget texte | 9,484 Go (idéal) | 9,467 Go de tenseurs, 9,478 Go de fichier |
| Allocation | uniforme 2,82 | par tenseur : blocs 2,71 en moyenne ; `output` 4,50 ; `token_embd` 2,625 ; `attn_v` 4,54 ; `attn_k` 4,00 ; `attn_output` 3,34 ; `ssm_out` 3,15 ; `ffn_gate` 2,52 |
| Profondeur | uniforme | couches 18–31 à 2,9–3,4 bpw ; couches 36–62 à 2,2–2,6 |
| Format | réseau E8 + codage entropique (non exporté) | types llama.cpp (IQ1_M … Q6_K), exécutables |
| Calibration | 147 k tokens, texte brut | non auditée ici |
| Évaluation | 1024 tokens, toutes positions, depuis un contexte vide | 4096 tokens, seconde moitié seulement, `<|endoftext|>` initial, réponses du BF16 |

La mesure commune (leurs poids dans notre graphe, et nos poids sur leurs fenêtres) est dans `results/`.
