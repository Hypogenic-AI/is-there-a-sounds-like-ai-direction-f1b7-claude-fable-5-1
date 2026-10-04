# Literature Review: Is there a "sounds like AI" direction in the residual stream?

Provenance: the four user-specified papers and two related finds were read in full
(all chunks; exceptions noted in the per-paper notes) and summarised in
`papers/notes/*.md` with page/section references. The remaining papers were
screened at abstract level and are characterised from their abstracts and general
knowledge of the work. Numbers quoted below come from the notes; values marked "≈"
were read off figures.

## Research Area Overview

Two literatures meet here and have not yet been joined.

1. **Machine-generated-text (MGT) detection from LLM activations.** Several 2025–26
   papers show that human-written vs AI-written text is linearly separable in the
   hidden states of a frozen "reader" LLM, that the separating direction transfers
   across domains and generators, and that it tracks a graded "machineness". All of
   this is *readout*: an LLM reads third-party text and a probe classifies it.
2. **Activation steering.** Directions found by difference-of-means between
   contrastive activations can causally control behaviours (refusal, sentiment,
   personas, the Assistant persona) when added to or ablated from the residual
   stream during generation.

**The gap.** No paper found takes the human-vs-AI readout direction, intervenes on it
*during the model's own generation*, and scores the result with an *independent*
detector while checking content and coherence. Nor has anyone tested whether that
direction is distinct from formality, domain, length, or the assistant persona.
Each neighbouring paper stops one step short (see table).

| Paper | Finds H-vs-AI direction | Steers generation along it | Scores steered text with a detector | Confound controls |
|---|---|---|---|---|
| Linear Probing (2608.24780) | yes (LR probe, Llama-3-8B) | no ("causality" = future work) | – | truncation ablation only |
| SV-Detect (2606.07313) | yes (mean-diff / LR / PCA, GPT-Neo) | no ("used only as probing features") | – | cross-domain transfer; regex-baseline |
| SAE features (2503.03601) | SAE features (Gemma-2-2b) | yes, per feature, for interpretation | no (GPT-4o describes the change) | lists length/punctuation features |
| Self-Recognition (2606.06315) | yes (LDA, own vs human summaries) | only with *random sparse* vectors (watermark) | no | length-matching, casing, punctuation |
| Base Models Look Human (2605.19516) | no (behavioural) | no (fine-tunes a paraphraser) | yes (GPTZero, Pangram) | prefix source |
| Assistant Axis (2601.10387) | persona axis, not H-vs-AI | yes | no | none relevant |

## Key Papers

### 1. Linear Probing Provides Robust and Efficient Detection of MGT (Quaremba et al., 2026; arXiv 2608.24780) — user-specified
- **Method**: Llama-3-8B(-Instruct in the code) reads raw text (no chat template, ≤1024 tokens); last-token residual state at each of 33 layers → StandardScaler → PCA-100 → L2 logistic regression. LLP = average of per-layer projections; CLP = one probe on concatenated layers.
- **Data**: DetectRL, MultiSocial, RAID, TSM; 1,500 train / 500 test per subset.
- **Results**: in-domain AUROC 0.90–1.00; OOD 0.81–0.99, up to ≈ +11.9 over the best of 16 baselines; 10–100 labelled samples suffice. Layer 16 best OOD. Cosine similarity of probe vectors is "high within-benchmark, moderate cross-benchmark" (heatmap only). Projection correlates with edit strength (r ≈ 0.7 with Levenshtein on APT-Eval) — "machineness spectrum", correlational.
- **Not done**: no steering; no formality/persona controls (authors themselves blame "stylistic cues associated with formal writing" for a failure on News).
- **Code**: `code/mgt_probes` (script broken as shipped; no vectors released).
- **Relevance**: defines the readout claim we take as the starting point; recommends mid layers (~16/32) and shows small extractors work.

### 2. SV-Detect (Vishnyakov & Gaintseva, 2026; arXiv 2606.07313) — user-specified
- **Method**: frozen GPT-Neo-2.7B, mean-pooled activations per layer; unit directions by mean-difference, LR weights (default) or PC1 of paired differences; feature = cosine of activation with direction per layer → LR head.
- **Results**: DetectRL in-distribution AUROC 99.8–100; cross-domain LR direction 81.8–99.4 vs mean-difference 71.9–98.1 vs PCA 50.8–92.7.
- **Interpretation**: logit lens gives AI-side tokens "endeavors, utilization, leveraging, captivating", human-side "anyway, basically, maybe, really" — i.e. polished/formal vs casual register. A regex surface baseline (em-dashes, connectives, markdown, length) already reaches 76–91 AUROC; the direction adds 9–24 points.
- **Hint of persona confound**: detectors trained on chat-model output do poorly on non-chat generators in RAID (TPR@5%FPR 10–34% vs 65–79%).
- **Not done**: no steering of generation despite the name.
- **Code**: `code/sv-detect`.

### 3. Feature-Level Insights into ATD with Sparse Autoencoders (Kuznetsov et al., 2025; arXiv 2503.03601) — user-specified
- **Method**: Gemma-2-2b + Gemma Scope residual SAEs; token-summed feature activations → XGBoost / single-feature thresholds; layer 16 best. Data: COLING-2025 Task 1 + RAID.
- **Features (L16)**: 3608 sentence complexity, 4645 assertive-vs-hedging, 6587 wordy introductions, 8264 repetition, 14161 formality, 14953 second-person advice, etc. Length-sensitive features (1033, 16028) and punctuation features are flagged.
- **Steering**: `x' = x + λ·A_max·d_i`, λ ∈ [−4, 4], all tokens; GPT-4o describes how 50 continuations change. No detector on steered text, no random-direction or fluency control; steered model/layer/decoding unstated.
- **Relevance**: evidence that "AI-ness" decomposes into several stylistic features (complexity, hedging, verbosity, formality) rather than one clean concept, and that features are strongly domain-dependent — supports testing "distinct direction vs relabelled style".

### 4. The Assistant Axis (Lu et al., 2026; arXiv 2601.10387) — user-specified
- **Method**: axis = mean(default-Assistant response activations) − mean(role-play activations) per layer; 275 roles × 5 system prompts × 240 questions, LLM-judge filtered; mean over response tokens. Models: Gemma-2-27B (layer 22/46), Qwen-3-32B (32/64), Llama-3.3-70B (40/80). Cosine with PC1 of persona space > 0.71 at the middle layer.
- **Steering**: additive at one middle layer on all tokens, scaled relative to mean residual norm; activation capping reduces persona-jailbreak harm ≈ 60% without capability loss.
- **Base models**: the persona space is near-identical in the base model; steering a base model toward "Assistant" yields helpful professional human archetypes — the "being an AI" association is attributed to post-training.
- **Caveat for us**: the non-Assistant pole is theatrical/mystical/poetic role-play, *not* ordinary human prose. So "assistant axis" ≠ "AI-vs-human text" a priori; their relation is an empirical question (cosine, projection-out).
- **Code/vectors**: `code/assistant-axis`; pre-computed axes only for the three large models (`datasets/assistant_axis_vectors/`).

### 5. Base Models Look Human To AI Detectors (Xu et al., 2026; arXiv 2605.19516)
- GPTZero / Pangram judge base-model continuations overwhelmingly human and instruct-model continuations not: Llama-3-8B base 96.7% / 98.8% human vs instruct 30.3% / 17.1% (human prefixes, T=1.0, top-p 0.95). Continued plain-LM fine-tuning of the instruct model restores human-likeness.
- **Relevance**: the strongest prior that detector-visible "AI-ness" is largely the post-training/assistant register. Motivates a **base-minus-instruct** control direction and holding prompts fixed across steering conditions.

### 6. LLM Self-Recognition: Steering and Retrieving Activation Signatures (Ardoin et al., 2026; arXiv 2606.06315)
- Mid-layer mean-pooled LDA separates the model's own summaries from human ones at AUROC ≥ 98.6 (Llama 1B/3B/8B, Ministral-8B), robust to length matching, lowercasing, punctuation stripping. Steering only with random 99.7%-sparse vectors (α=5) as a watermark, recoverable by re-encoding the text.
- **Relevance**: (a) a random norm-matched vector is a required control — random steering alone leaves a detectable trace; (b) the "read back the steered text with the probe" check is a useful closed-loop measurement.

### 7. Steering methodology (screened)
- **Arditi et al. 2024 (refusal direction)**: diff-of-means over all layers/positions, select by causal effect, activation addition (`x + α r`) and directional ablation (`x − r̂ r̂ᵀx` at every layer); the canonical necessity + sufficiency test. `code/refusal_direction`.
- **Turner et al. 2023 (ActAdd), Panickssery et al. 2023 (CAA), Chen et al. 2025 (persona vectors)**: contrastive mean-difference vectors added at a mid layer; coefficient sweeps; persona vectors add LLM-judge trait and coherence scores.
- **Marks & Tegmark 2023**: mean-difference directions are more causally implicated than logistic-regression probe directions even when LR classifies better — relevant because both MGT papers default to LR probes (and SV-Detect finds LR transfers better for *readout*).
- **Tan et al. 2024; Wu et al. 2025 (AxBench)**: steering effects are unreliable across inputs and trade off against fluency; evaluate with concept score × fluency × instruction-following, and compare to prompting baselines.
- **Konen et al. 2024 (style vectors)**: style attributes can be steered by activation vectors — precedent that register-like properties are steerable.

### 8. Detection benchmarks, detectors and evasion (screened)
- **Datasets**: HC3 (Guo et al. 2023), MAGE (Li et al. 2023), RAID (Dugan et al. 2024), HAP-E (Reinhart et al. 2024 — Biber-feature analysis showing instruction-tuned models deviate from human style much more than base models).
- **Detectors**: Binoculars (Hans et al. 2024) and Fast-DetectGPT (Bao et al. 2023), zero-shot perplexity/curvature-based; RADAR (Hu et al. 2023), adversarially trained RoBERTa. RAID protocol: threshold at fixed FPR on human text, report TPR.
- **Evasion in text space**: DIPPER paraphrasing (Krishna et al. 2023), Adversarial Paraphrasing (Cheng et al. 2025). These are the text-space counterparts of what activation steering would do.

## Common Methodologies
- **Direction extraction**: diff-of-means (Arditi, CAA, persona vectors, Assistant Axis, SV-Detect) vs logistic-regression weights (Linear Probing, SV-Detect default) vs PCA of paired differences. Pooling: last token (Linear Probing) vs mean over tokens (SV-Detect, Self-Recognition, Assistant Axis). Middle layers are consistently best for OOD readout and for steering.
- **Intervention**: add `α·v` at one mid layer on all positions (α scaled to residual norm), and/or project the unit direction out of every layer.
- **Evaluation of steering**: target metric + coherence/fluency judge + dose-response over α.

## Standard Baselines
- For readout: Binoculars, Fast-DetectGPT, RoBERTa detectors, RepreGuard.
- For steering: random direction with matched norm; prompting ("write like a human"); unrelated-concept direction. For evasion: paraphrasing.

## Evaluation Metrics
- Detection: AUROC, TPR at 1% / 5% FPR, mean detector score / "% judged human".
- Steering: dose-response slope of detector score vs α; coherence (LLM-judge 0–100, perplexity under a separate LM, repetition rate); content preservation (embedding cosine to unsteered output, LLM-judge on-topic/answer-correctness).
- Geometry: cosine between directions per layer; change in effect after projecting out a confound direction.

## Datasets in the Literature
RAID (2608.24780, 2503.03601, 2605.19516, SV-Detect), MAGE (2605.19516), DetectRL / MIRAGE / MultiSocial / TSM / APT-Eval (probe papers; not downloaded), XL-Sum (Self-Recognition), HC3 and HAP-E (user-specified).

## Gaps and Opportunities
1. **Causality is untested.** Readout direction → generation intervention → independent detector has not been reported.
2. **Distinctness is untested.** No paper controls for formality, length, domain or assistant persona; SV-Detect's own logit-lens reading *is* a formality reading, and 2605.19516 suggests detectors key on instruction-tuning artefacts.
3. **Readout ≠ causal direction.** LR probes (best readers) may not be the best steering vectors (Marks & Tegmark); a head-to-head is cheap.
4. **Reading third-party text vs writing own text.** Probe directions are computed while *reading* text; whether the same direction governs *writing* is the crux.

## Recommendations for Our Experiment

**Model.** A mid-size instruct model that fits one 48 GB A6000 with hooks and has a base sibling: `meta-llama/Llama-3.1-8B-Instruct` + `meta-llama/Llama-3.1-8B` (HF access verified). This matches the extractor family of 2608.24780, the base/instruct pair in 2605.19516, and the Llama-3-8B base/instruct texts in HAP-E. Optional replication: `Qwen/Qwen2.5-7B-Instruct` or `google/gemma-2-9b-it` (access verified).

**Datasets.**
- Direction extraction: **HAP-E** (human chunk-2 vs LLM continuation of the same chunk-1; topic/genre/length matched; 6 genres) as primary; **HC3** as an easy but confounded contrast; **RAID no-attack** and **MAGE** for transfer of the readout.
- Generation prompts: HAP-E chunk-1 continuations (has a matched human reference), HC3 questions, RAID `title`/`prompt`.
- Confound directions: Pavlick formality scores (formality), HAP-E genre labels (domain), token count (length), base-vs-instruct activations and/or a reduced Assistant-Axis pipeline (persona), Biber features as interpretable covariates.

**Independent detectors** (must not share the direction's model): Binoculars (falcon-7b / falcon-7b-instruct), Fast-DetectGPT, RADAR (`TrustSafeAI/RADAR-Vicuna-7B`), and one supervised RoBERTa-style detector (e.g. `Hello-SimpleAI/chatgpt-detector-roberta`, `desklib/ai-text-detector-v1.01`). All exist on HF and are ungated; none were run in this phase. GPTZero/Pangram are paid APIs with no key available (`pangram/editlens_roberta-large` is gated and not accessible).

**Methodological cautions.**
- *Perplexity confound*: Binoculars / Fast-DetectGPT score low-perplexity text as AI. Any steering that degrades fluency raises perplexity and will "look human". A **norm-matched random direction** and coherence metrics at every α are mandatory; report detector change *at matched coherence*. Include a trained (non-perplexity) detector.
- *Read vs write*: compute the direction at the hook point and token positions where it will be applied; test both mean-pooled and last-token variants, and both diff-of-means and LR.
- *Chat template*: probe papers feed raw text; generation uses the chat template. Extract with the text placed where generated tokens would be (e.g. as the assistant turn) as well as raw.
- *Prompts fixed across conditions*; same decoding (the base-models paper uses T=1.0, top-p 0.95); enough samples for CIs (≥ 200 prompts per condition), paired tests.
- *Distinctness tests*: (i) cosine of AI direction with formality / length / genre / base-vs-instruct / assistant directions per layer; (ii) steer with the AI direction after projecting those out — does the detector effect survive?; (iii) steer with the confound directions alone — do they reproduce the effect?; (iv) measure formality classifier score, length and Biber features of steered outputs.
- *Prompting baseline*: "write so it doesn't sound like AI" as a non-mechanistic comparison (AxBench practice).
