# Journal du pod — FORMIC-Ink, 22/11/2026 (résumé)
1. Professeur : checkpoint Inkling-2-Mini b85c3cf déquantifié exactement ; correctif du chargeur Transformers 5.24 (facteur NVFP4 arrondi en BF16) ; = UD-Q8_K_XL bit à bit (KL 0).
2. Plancher numérique du graphe : ~0,017–0,019 de KL entre exécutions équivalentes (grouped_mm / boucle / FP32).
3. P-I : 106 fenêtres de 4096, 9 domaines, dev 54 / conf 52 (scellée) ; calibration 1,18 M tokens (6 experts sur 3 072 sous le seuil).
4. L0 (F1 transposée) : multiplicateur 0,757082, 8 474 167 056 octets texte (budget UD-IQ2_XXS −0,001 %), passe 4 328 s, pic 77,1 Gio.
5. P-I dev : L0 0,0802, V1a 0,0785, UD-IQ2_XXS 0,0862, UD-Q2_K_XL 0,0712, UD-IQ3_XXS 0,0376.
6. Sélection gelée (commit 8f0adfa) : L0 (départage par domaines contre V1a).
7. P-I conf : Formic 0,0785 contre UD-IQ2_XXS 0,0819 (−4,2 %, IC [−13,4 ; +5,0]) → niveau « Échec » ; 7/9 domaines gagnés, agentique perdu.
8. Paquet /workspace/work_ink/packages/L0 : 8 860 641 865 octets, verify OK, relu depuis les fichiers (KL identique à 6 décimales).
9. Vision 4,93 bpw : KL 0,0109 (tour seule) ; drafter 3,98 bpw non évalué ; llama.cpp non testé.
10. Phase 2 lancée à la demande de l'opérateur : V4 (attention/vocabulaire +1 bit) puis C1, confirmation sur P-I2 (nouveau jeu scellé).
11. Phase 2 : V4 sélectionné (P-I dev 0,0612), confirmé sur P-I2 scellé : 0,0584 contre 0,0779 UD-IQ2_XXS (−25,0 %, IC [−39,6 ; −11,7]) → Formic V1.5 = V4, paquet /workspace/work_ink/packages/V4, 8 860 059 425 octets, verify OK, relu.
12. Diagnostic agentique : V4 2–3× meilleur que UD-IQ2_XXS sur les jetons générés par le modèle ; l'écart vient des réponses d'outil / système / jetons spéciaux (surtout experts → calibration).
