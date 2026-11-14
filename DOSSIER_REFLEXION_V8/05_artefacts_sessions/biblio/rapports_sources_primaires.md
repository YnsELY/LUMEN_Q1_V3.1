# Rapports des 4 agents web-fetch (session 78f73527, 21h23-21h24)

Texte brut des rapports rendus par les agents. Les pages ont été tronquées vers 40 000 caractères : les passages marqués « summary » ne sont pas vérifiés mot à mot.


---
## massive_activations_outliers (a01039cce855562c0)

Source notes: Sources 1 and 3 were fetched as HTML pages, and the fetch truncated both. For 1, I read verbatim through Section 4.3 (about the first 40% of the page). The rest came as a model summary, which I did not rely on. For 3, I read the abstract and opening sections verbatim, but the kurtosis and cause sections were not in the verbatim text. Quotes below are short fragments of under 125 characters.

**1. Massive Activations in Large Language Models (Sun, Chen, Kolter, Liu)**
URL read: https://arxiv.org/html/2402.17762v2 (header: "arXiv:2402.17762v2 [cs.CL] 14 Aug 2024"). I did not check v1.
Venue: the abstract footnote says "Published as a conference paper in First Conference on Language Modeling (COLM), 2024."

a. Definition (Section 2, "Massive Activations", Table 1): the paper gives a "loose but broad definition". An activation counts as massive if "its magnitude surpasses 100" and is "at least or around 1,000 times larger than the median magnitude of its hidden state." Table 1 gives LLaMA2-7B top-1 2622.0, top-2 1547.0, median 0.2. The intro says these are "more than 4 orders of magnitude larger than the median." The 1,000x figure is a rule of thumb, not a hard threshold.

b. Intervention (Section 3, Table 3, LLaMA2-7B; the same experiment on LLaMA2-13B is in the same table). The intervention is applied at layer 2 for LLaMA2-7B and layer 4 for LLaMA2-13B.
- LLaMA2-7B, Original: WikiText 5.47, C4 7.85, PG-19 8.57, zero-shot 68.95%.
- LLaMA2-7B, Set to zero: inf, inf, inf, 36.75%.
- LLaMA2-7B, Set to mean: 5.47, 7.86, 8.59, 68.94%.
- LLaMA2-13B, Original: 4.88, 7.22, 7.16, 71.94%. Set to zero: 5729, 5526, 4759, 37.50%. Set to mean: 4.88, 7.22, 7.16, 71.92%.
The text says zeroing gives "exploding perplexity numbers." Zeroing ordinary activations of similar magnitude to the median "leads to no performance drop."

c. Normalization, residual norm, bias terms.
- The paper says "massive activations act as fixed but important biases in LLMs" (end of Section 3). The abstract says they "function as indispensable bias terms in LLMs."
- Section 4.2 mentions the normalization step. "We hypothesize that the attention LayerNorm may play a pivotal role in this process." Footnote 3 says LLaMA2 uses RMSNorm.
- I did not verify any residual-stream-norm claim in the verbatim text.
- LLaMA2-7B: feature dimensions 1415 and 2533 (Section 2.2, Figure 1). They first appear at layer 2 and stay "nearly constant values until layer 30" (Section 2.1).

d. Tokens (Section 2.2, verbatim): "the starting word token" and "the token representing the first period (.) or newline token (\n) in the sequence." Section 2 summary: "Starting token and the first 'strong' delimiter token." Section 3 "Why these layers and tokens?" gives the reasoning that the start token occurs in every training sequence.

e. Venue: COLM 2024 (see above).

**2. Understanding and Minimising Outlier Features in Neural Network Training (He, Noci, Paliotta, Schlag, Hofmann)**
URL read: https://arxiv.org/abs/2405.19279. Title and authors match your query exactly. The HTML body was not read.
a. Abstract, verbatim:
- "Outlier Features (OFs) are neurons whose activation magnitudes significantly exceed the average over a neural network's (NN) width."
- "little is known behind why OFs emerge during training, nor how one can minimise them."
- It uses metrics such as "the kurtosis over neuron activation norms" and studies "how architectural and optimisation choices influence OFs."
- It names a "default OF-prone combination of Pre-Norm model and Adam."
- It introduces "the Outlier Protected block" (an unnormalised transformer block) and "a previously unknown benefit of non-diagonal preconditioning optimisers."
The abstract does not explicitly state that normalisation layers, Adam, or learning rate cause OFs. The causal framing is in the body, which I did not read.
b. arXiv v2, submitted 29 May 2024 (v1), revised 6 Nov 2024 (v2). Comment: "NeurIPS 2024 camera ready." Venue: NeurIPS 2024. Licence CC BY 4.0.

**3. Privileged Bases in the Transformer Residual Stream (Elhage, Lasenby, Olah)**
URL read: https://transformer-circuits.pub/2023/privileged-basis/index.html
a. Main claim and cause (verbatim abstract):
- "individual coordinates in the residual stream should have no special significance"
- "we provisionally conclude that the per-dimension normalizers in the Adam optimizer are to blame for the effect."
- "We confidently rule these out as being the source of the observed basis-alignment." (Layer normalization and floating-point precision were the two alternatives tested.)
Outlier definition (verbatim): an "outlier" is "a single scalar value in the residual stream whose absolute value is >6."
Measurement: the abstract does not name a measure. The "privileged basis" test and kurtosis were not in the verbatim text I read. The model summary said they use kurtosis, with 3 as the Gaussian reference, and mentioned "20–60 outlier dimensions" out of d_model=1280. I did not verify these numbers, so treat them as unconfirmed.
b. Published March 16, 2023. Authors: Nelson Elhage, Robert Lasenby, Christopher Olah. Affiliation: Anthropic.

Pages not read: the rest of sources 1 and 3 (the main text after Section 4.3, the appendices, and sections 5–7 of source 1; the causal analysis in source 3 after the first ~40k characters). Claims about those parts are unconfirmed.

---
## quip_quip_sharp (a342436ec991919e0)

Structured report. Quotes are short; formulas are as rendered on the pages.

PAGES READ
- https://ar5iv.labs.arxiv.org/html/2307.13304 (ar5iv render; text cut at ~40k of 65k chars, the rest came only as a model summary, so sections after Sec. 5.2 are not verified)
- https://arxiv.org/abs/2307.13304 (metadata)
- https://arxiv.org/html/2402.04396v2 (read in two offset chunks, 0-39956 and 39956-79912; the rest of the paper, including Sec. 4.3 and Appendix B, was not read verbatim)

1) QuIP (Chee, Cai, Kuleshov, De Sa), arXiv 2307.13304

a. Definitions
- Proxy objective, Eq. (1): ℓ(Ŵ) = E_x[‖(Ŵ−W)x‖²] = tr((Ŵ−W)H(Ŵ−W)^T). (Sec. 3, intro to Sec. 3.1). H is the second moment of calibration inputs, the "proxy Hessian".
- LDLQ update, Sec. 3.1 (unnumbered display): Ŵ_k = Q(W_k + (W_{1:(k−1)} − Ŵ_{1:(k−1)}) a_k), with Q nearest or stochastic rounding.
- Matrix form, Eq. (2): Ŵ = Q(W + (W − Ŵ)U), U strictly upper triangular, Q acting elementwise.
- Eq. (3): tr((Ŵ−W)H(Ŵ−W)^T) = tr(η(U+I)^{-1} H (U+I)^{-T} η^T), with η the quantization error.
- LDL choice, Eq. (4): H = (Ǔ+I) D (Ǔ+I)^T, D non-negative diagonal, Ǔ upper unit triangular. Setting U ← Ǔ makes the (U+I) factors cancel. (Note: Algorithm 3, line 3 prints "(Ǔ+I)D(Ǔ+I)^{-1}", which looks like a typo for the transpose.)

b. Theorem 1 (Sec. 3.2). Quoted: "LDLQ is worst and average-case optimal amongst rounding methods which specify the linear feedback U as a function of H (not of W), and when rounding to the integers."
Formula: (m/4) tr(D) = L_worst(LDLQ, H) ≤ L_worst(A, H), and (m/c) tr(D) = L_avg(LDLQ, H) ≤ L_avg(A, H), with c = 12 for nearest and c = 6 for stochastic rounding.
Worst/average losses are defined in Eqs. (5) and (6).
Lemma 3 (Sec. 3.3) gives the baselines: L_worst(Stoch, H) = (m/4) tr(H), and L_avg({Near, Stoch}, H) = (m/c) tr(H).

c. Lemma 2 (Sec. 3.3). Assumes H is μ-incoherent (Definition 1: |Q_ij| ≤ μ/√n for H = QΛQ^T). Statement: tr(D) ≤ (μ²/n) tr(H^{1/2})².
Comparison: tr(D) < tr(H̃) for non-diagonal H̃ ⪰ 0 (Sec. 3.2). Combining Lemma 2 with Theorem 1 and Lemma 3 gives, for rank-k H with μ²k < n: L_worst(LDLQ, H) ≤ (mμ²/4n) tr(H^{1/2})² ≤ (mμ²k/4n) tr(H) ≤ (m/4) tr(H). This is the chain in Sec. 3.3, unnumbered display.
Theorem 4 (Sec. 3.3) says that without incoherence, no spectral bound separates LDLQ from the baselines.

d. Theorem 6 (Sec. 5.1): "OTPQ [sic] falls within the class of adaptive rounding procedures with linear feedback ... and is equivalent to LDLQ in Section 3." (Typo "OTPQ" is in the paper.)

e. Version and venue
- arXiv listing: v1 submitted 25 Jul 2023; v2 revised 15 Jan 2024 (the "this version"). Categories cs.LG, cs.CL. The ar5iv text matches v2 content, but ar5iv does not label its version.
- The abs page shows no journal-ref or venue. The QuIP# paper's own bibliography cites QuIP as "Thirty-seventh Conference on Neural Information Processing Systems, 2023" (NeurIPS 2023). That is a citation in another paper, not a statement on the QuIP arXiv page.

2) QuIP# (Tseng, Chee, Sun, Kuleshov, De Sa), arXiv 2402.04396v2

a. BlockLDLQ
- Block LDL: H = L^T D L, with L unit block lower triangular and D block diagonal, with block size g (Sec. 4.1). U = L^T − I. Per the text, the update is Ŵ_k = Q(W_k + (W_{:(k−1)} − Ŵ_{:(k−1)}) A_k), where A_k holds the k−g+1 through k columns of U. Ŵ = Q(W + (W − Ŵ)(L^T − I)) is the matrix form stated in Sec. 4.1.
- Theorem 4.1 (Sec. 4.1), quoted: "Suppose that HH is μ-incoherent, and that we use a (possibly stochastic) vector quantizer Q that satisfies E[(Q(x)−x)(Q(x)−x)^T] ⪯ σ²I for any x ∈ R^g."
Bound: E[tr((Ŵ−W)H(Ŵ−W)^T)] ≤ (g m μ² σ²/ n) tr(H^{1/2})².
The text says the naive block-independent bound is g m σ² tr(H), and that the improvement is "the same factor achieved in the scalar case in QuIP."
- Dependence: g is the block size, n the Hessian dimension, μ the incoherence, and the bound depends on tr(H^{1/2})². (The text calls the block size d in places, and the model-summary of the rest of the paper calls it d; the theorem uses g.)
- Lemma B.1 (μ²/n tr(H^{1/2})² bound on tr(D)) is in Appendix B, which I did not read verbatim. Not verified.

b. E8 and E8P (Sec. 4 intro, Sec. 4.2, Sec. 6.3)
- Sec. 4 intro: "E8 lattice, which achieves the highest density unit ball packing in R^8". Sec. 4.2 notes the E8 "codebooks achieve lower MSEs than all other presented codebooks ... due to the underlying packing density and high dimensionality of E8" (Figure 3 caption).
- Sec. 4.2 E8P: "2-bit 8 dimensional E8P codebook, which contains 2^16 entries but only requires lookups into a 2^8-entry table, with the remaining 8 bits being used to store signs and shifts." Also: "E8P requires only 1KiB of space" (Sec. 4.2 and Sec. 6.3).
- Sec. 4.2 construction: S = 227 elements of |D̂8| with norm ≤ √10 plus 29 padding elements of norm √12. The 16-bit codeword is 8 bits (entry of S), 7 sign-flip bits, and 1 bit for ±1/4.
- Sec. 4.3: 2 bits is E8P. 4 bits is residual VQ with E8P applied twice. 3 bits is E8P plus a 1-bit E8 codebook. Read from the page text, Sec. 4.3 verbatim.
- Bits overhead: Sec. 5 and Appendix F.1 (the latter from the model summary, not verbatim) say sign vectors add less than 0.01 bits per weight.

c. Shape and Gaussianity
- Sec. 4 intro (verbatim): "It follows from the central limit theorem that RHT-transformed weights follow a roughly ball-shaped Gaussian distribution."
- Abstract: "take advantage of the ball-shaped sub-Gaussian distribution that incoherent weights possess".
- Sec. 1: "Incoherence processing ... produces approximately Gaussian distributed weight matrices".
- Sec. 4 intro: scalar rounding "produc[es] a set of representable weight vectors that is shaped like a hypercube rather than a ball", which is the reason for vector quantization with a ball-shaped codebook.
- Sec. 4.2: "Increasing the vector dimension decreases the error for the half integer grid, as the resulting codebook is closer in shape to the source distribution."
- Appendix C.3 and C.4 (summary only): E8P is uniform, and the text says it beats a Gaussian-shaped K-means codebook. Treat as unverified.

d. Version and venue
- arXiv 2402.04396v2, footer "[cs.LG] 04 Jun 2024". I did not read the v1 date or the abs page.
- The page's keywords list "ICML". I did not verify the acceptance or venue from a source page.

Notes on what was not verified: QuIP Sec. 5.2 onward and the appendices (only the model summary), QuIP# Sec. 4.3 (read verbatim) and Appendix B (summary only).

---
## equalization_nagel_smoothquant_awq (ad51f9d3f29809335)

Fetched all three ar5iv pages (HTTP 200). Each page was truncated at about 40k characters. The first ~40k characters were verbatim; the rest came from a model-extracted summary, which I flag below. No page failed, so I did not use the arxiv.org/abs fallback.

**1) Nagel et al., "Data-Free Quantization Through Weight Equalization and Bias Correction"**
URL read: https://ar5iv.labs.arxiv.org/html/1906.04721

a. Scaling-equivariance identity: Section 4.1 ("Positive scaling equivariance"), Eq. (2):
   f(sx) = s f(x), "for any non-negative real number s" (the paper's wording).
   The ReLU definition is Eq. (3). Eq. (4) relaxes this to piecewise-linear f with f^(·) parameterized as a^_i = a_i, b^_i = b_i/s, c^_i = c_i/s.
   The network-level form is Eqs. (5)-(7): W^(2)_hat = W^(2) S, W^(1)_hat = S^-1 W^(1), b^(1)_hat = S^-1 b^(1), with S = diag(s) (Sec. 4.1.1).

b. Optimal per-channel scale: Section 4.1.2, Eq. (11):
   s_i = (1 / r_i^(2)) * sqrt(r_i^(1) r_i^(2))
   Objective, stated in the verbatim text before Eq. (9): "We want to find S such that the total precision per channel is maximized:" Eq. (9): max_S sum_i p_hat_i^(1) p_hat_i^(2). Precision is defined in Eq. (8) as p_hat_i^(1) = r_hat_i^(1) / R_hat^(1).
   Necessary condition, Eq. (10): argmax_j (1/s_j) r_j^(1) = argmax_k s_k r_k^(2). Eq. (11) "results in ∀i: r_i^(1) = r_i^(2)".
   Caveat: the paper's own text does not say Eq. (11) "maximizes" anything directly; it says it satisfies the condition from Eq. (10) and matches the ranges. The appendix derivation (Eqs. 20-28) is only in the model summary, which I could not verify verbatim, so do not cite those numbers as checked.

c. Version and venue: The ar5iv page shows no v1/v2/v3 label and no venue. Unknown from this page.

**2) Xiao et al., "SmoothQuant"**
URL read: https://ar5iv.labs.arxiv.org/html/2211.10438

a. Smoothing factor, Section 4, Eq. (4):
   s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
   Migration strength: "Here we introduce a hyper-parameter, migration strength α, to control how much difficulty we want to migrate from activation to weights, using the following equation:" Eq. (3) is the equivalence Y = (X diag(s)^-1)(diag(s) W).

b. Recommended alpha, Section 4 (verbatim): "We find that for most of the models, e.g., all OPT ... and BLOOM ... models, α = 0.5 is a well-balanced point to evenly split the quantization difficulty". For GLM-130B: "we can choose a larger α to migrate more quantization difficulty to weights (like 0.75)." Section 5.1 says α = 0.5 for OPT/BLOOM and α = 0.75 for GLM-130B. Table 6 (LLaMA) uses α = 0.8. Table 7 gives per-model α: Llama-2-7B 0.85, Llama-2-13B 0.85, Llama-2-70B 0.9, Falcon-7B 0.6, Falcon-40B 0.7, Mistral-7B 0.8, Mixtral-8x7B 0.8.
   alpha=0 and alpha=1: the paper does not label these. By substituting into Eq. (4), alpha=1 gives s_j = max|X_j| and alpha=0 gives s_j = 1/max|W_j|. Section 4 says the first "pushes all the quantization difficulties to the weights" and the second makes "the model performance is bad due to the activation quantization errors." These are my mapping of the paper's text, not its own statement.
   Caveat: the summary says the sweet spot is 0.4-0.6, and that alpha below 0.4 or above 0.6 is bad. I did not see this in verbatim text, so treat it as unverified.

c. Version and venue: The page shows no arXiv version. The keywords line reads "Machine Learning, ICML", so the venue is ICML according to that page.

**3) Lin et al., "AWQ: Activation-aware Weight Quantization"**
URL read: https://ar5iv.labs.arxiv.org/html/2306.00978

a. Search, Section 3.2:
   Eq. (4): s* = argmin_s L(s), with L(s) = || Q(W · diag(s)) (diag(s)^-1 · X) − W X ||.
   Eq. (5): s = s_X^alpha, alpha* = argmin_alpha L(s_X^alpha). Here s_X is the per-channel average activation magnitude.
   Grid: "We can find the best α by a fast grid search over the interval of [0,1]" (0 means no scaling, 1 is the most aggressive). Section 5.1: "We used a grid size of 20 to search for the optimal α." The paper also applies weight clipping to minimize quantization MSE.

b. Delta'/Delta argument, Section 3.2. Quantizer Eq. (1): Q(w) = Δ · Round(w/Δ), Δ = max(|w|)/2^(N-1).
   Eq. (2): Q(w·s)·(x/s) = Δ' · Round(w s / Δ') · x · (1/s).
   Eq. (3): Err(Q(w)x) = Δ · RoundErr(w/Δ) · x, and Err(Q(w·s)(x/s)) = Δ' · RoundErr(w s/Δ') · x · (1/s).
   Ratio stated in the text: "The ratio of the new error to the original error is Δ′/Δ · 1/s." Assumptions: RoundErr ≈ 0.25 unchanged, and Δ' ≈ Δ. "Given Δ′≈Δ and s>1, the relative error is smaller for the salient weight w."
   Table 2 (OPT-6.7B): average Δ'/Δ is 1.005, 1.013, 1.038, 1.213 for s = 1.25, 1.5, 2, 4. Wiki-2 PPL is best at s=2 (11.92). The paper also notes the error for non-salient channels grows when Δ' > Δ.

c. Venue and version: "Proceedings of the 7th MLSys Conference, Santa Clara, CA, USA, 2024. Best Paper Award." The page shows no arXiv version number.

Note: ar5iv renders one version of each paper. None of the three pages states which arXiv version it is.

---
## lattice_yaqa_gdn (ae312724b4c31a9ef)

Report for the three arXiv sources. Quotes are under 125 characters. Items marked (summary) come from a model-extracted summary of a truncated page, not verbatim text.

1) Agrell & Allen, "On the best lattice quantizers" (arXiv:2202.09605)
Read: https://arxiv.org/abs/2202.09605 ; https://arxiv.org/html/2202.09605v2 (truncated at 39,956 of 97,738 chars; rest is summary) ; https://arxiv.org/pdf/2202.09605 (returned binary, no readable text, so I used the v2 HTML instead; ar5iv not needed).
a. NSM values:
- E8: 0.071682099, Table I ("The best known lattice quantizers"), "Best previously reported" row, n=8 (summary). Matches 929/12960 ≈ 0.0716821.
- Z^n: 0.083333333 = 1/12. Table I's entry is the n=1 row; Corollary 8 proof states "G₂ = G(Q̂_ℤ) = 1/12" (summary).
- 1/(2πe): NOT found in any readable part of the paper. Do not attribute it to this paper.
b. NSM definition, Eq. (9), Section II: G(Q_Λ) ≜ E(Q_Λ) / (n V_Λ^{2/n}). E(Q_Λ) is the mean square error (Eqs. (7)/(8): E = (1/V_Λ)∫_{Ω(Q_Λ)} ‖x‖² dⁿx). V_Λ = |det B| (Eq. (4)).
c. Venue/version: IEEE Trans. Information Theory, vol. 69, no. 12, pp. 7650–7658, Dec. 2023, DOI 10.1109/TIT.2023.3291313 (arXiv abs page). arXiv v1 19 Feb 2022; v2 28 Jun 2023 (the HTML version). Primary class cs.IT.

2) Tseng, Sun, De Sa, "Model-Preserving Adaptive Rounding" (YAQA), arXiv:2505.22988
Read: https://arxiv.org/abs/2505.22988 ; https://arxiv.org/html/2505.22988v3 (truncated at 39,956 of 114,139 chars; rest is summary).
a. Quotes:
- Abstract: "most quantization algorithms minimize the immediate activation error of each layer as a proxy for the end-to-end error." Same abstract continues: "this ignores the effect of future layers, making it a poor proxy."
- Sec. 2.1.1: "In GPTQ and LDLQ, which are equivalent, columns in W∗ are iteratively rounded with linear feedback..."
- Sec. 1 (intro): GPTQ "rounds using the Hessian of the immediate layerwise activation error as a proxy". The paper does not literally say GPTQ/LDLQ ignore KL in one sentence; the KL argument is in the abstract, Sec. 1, and Eq. (1).
b. Theory:
- Eq. (1): θ̂ ← argmin_{θ∈C} E_{X∼D}[D_KL(M(θ*,X) ‖ M(θ,X))].
- Eq. (2): ℒ(W) ≈ ½ vec(W−W*)ᵀ (∇²ℒ(W*)) vec(W−W*). First-order terms vanish since D_KL is minimized at the original model.
- Eq. (3), LDLQ: W = 𝒬(W* + (W*−W)(L₁−I)). LDLQ minimizes tr((W*−W)H₁(W*−W)ᵀ), H₁ = E[xᵀx] (Sec. 2.1.1).
- Eq. (4)/(5): YAQA with H̃ = H_O ⊗ H_I; Eq. (5): W = vec⁻¹(𝒬(vec(W*) + vec(Δ)(L_O⊗L_I − I))), Δ = W*−W. Eq. (6) is expanded form.
- Theorem 3.4: bounds vec(Δ)H vec(Δ)ᵀ via cosine similarity c = ⟨H,H_O⊗H_I⟩/(‖H‖_F‖H_O‖_F‖H_I‖_F).
- Sketch power iteration: Eqs. (8)–(9) generic; Sketch A Eqs. (10)–(11); Sketch B Eqs. (12)–(13). Kronecker-factored: H ≈ H_O ⊗ H_I (Sec. 2.2, Sec. 3.2).
c. ≈30% claim: HTML v3 abstract: "empirically reduces the KL divergence by ≈30% over these methods." Intro: "Empirically, YAQA reduces the KL by ≈30% over LDLQ." NOTE: the arXiv abs page abstract for the same v3 instead says "reduces the error by ≈30% over these methods". (summary) Table 1 caption "YAQA reduces the KL by 30% over LDLQ" and Sec. 4.3 "≈10-20% over LDLQ" (finetuning) are in the unread part.
d. Version/venue: arXiv v3, revised 3 Jun 2026 (v1 29 May 2025; v2 26 Sep 2025). arXiv comments: "ICML 2026". Acceptance not otherwise verified.

3) Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv:2412.06464)
Read: https://arxiv.org/abs/2412.06464 ; https://arxiv.org/html/2412.06464 (HTML v3, truncated at 39,958 of 117,992 chars; rest is summary).
a. Equations:
- Eq. (10), Sec. 3.1, gated delta rule (state update): S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ, α_t ∈ (0,1), β_t ∈ (0,1). (The summary wrongly said Eq. (10) was missing; it is in the verbatim text.)
- Table 1, row "Gated DeltaNet": same update, S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ.
- Output: no explicit per-token gated output equation is in readable text. Sec. 2.1 gives the base recurrence output o_t = S_t q_t (unnumbered). Gated chunkwise output O_[t] in Sec. 3.3 is unnumbered. So the output equation is implicit; I did not find a numbered one.
b. Version/venue: arXiv v3, revised 6 Mar 2025 (v1 9 Dec 2024; v2 5 Mar 2025). arXiv comments: "ICLR 2025 camera ready."

Failures: none. The lattice PDF returned binary, so HTML v2 was used. Items marked (summary) and anything past the truncation points are unverified.
