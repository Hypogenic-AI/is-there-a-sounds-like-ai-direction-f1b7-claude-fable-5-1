# Planning: Is there a "sounds like AI" direction in the residual stream?

## Motivation & Novelty Assessment

### Why This Research Matters
Human-vs-AI text is linearly readable from LLM activations, and that readout is now the
strongest family of machine-text detectors. Whether the same direction *controls* how
AI-like a model's own writing is matters for detector evasion (a single vector that
hides AI authorship), for interpretability (does a model represent "this text was
written by a model", or just a register?), and for the relation between "sounding like
AI" and "being the Assistant".

### Gap in Existing Work
From `literature_review.md`: Quaremba et al. (2608.24780), SV-Detect (2606.07313) and
Kuznetsov et al. (2503.03601) read the direction out but never intervene on it during
generation and score the result with an independent detector. None controls for
formality, fluency, domain, length or the assistant persona. Xu et al. (2605.19516)
show behaviourally that base-model text looks human to detectors, which suggests the
detector-visible signal may be chat-tuning register rather than model authorship.

### Our Novel Contribution
(1) A causal test: add / ablate the human-vs-AI readout direction while
Llama-3.1-8B-Instruct writes, score the output with detectors that never saw those
activations, and compare with equal-norm random, confound-direction and prompt
baselines at matched coherence. (2) A disentanglement test against formality,
fluency (low perplexity), length, genre, chat-tuning register and an assistant-axis
proxy, by geometry, projection-out and steering with the confounds themselves.
(3) A base-vs-instruct comparison of readout and of steering.

### Experiment Justification
- **E1 Readout and generality** — needed to establish that the direction we later
  steer is the one the literature describes (held-out AUROC, cross-genre,
  cross-generator, cross-dataset transfer), and to pick layers.
- **E2 Causal steering** — the central open question; without it the work is a replication.
- **E3 Disentanglement** — a causal effect is uninterpretable if the direction is
  formality or persona under another name.
- **E4 Base vs instruct** — tests whether the direction pre-exists chat tuning and
  whether chat tuning moves the model's own writing along it.

## Research Question
Is the linear direction that separates human-written from AI-written text in the
residual stream of Llama-3.1-8B(-Instruct) causal for generation — does moving along
it change how AI-like the output reads to independent detectors with content and
coherence held — and is it distinct from formality, fluency, length, domain and the
assistant persona?

## Hypothesis Decomposition
- **H1 (readout)**: a diff-of-means direction fit on matched HAP-E pairs separates
  held-out human vs AI text, across genres, generators and datasets (HC3, RAID, MAGE).
- **H2 (causal)**: steering the instruct model toward the human side lowers
  independent-detector AI scores more than (a) equal-norm random directions,
  (b) confound directions, at matched coherence; steering toward the AI side raises
  them. Ablating the direction lowers them.
- **H3 (distinct)**: the direction is not explained by formality / fluency / length /
  genre / chat register / assistant axis: low cosine, readout survives projection-out,
  and the steering effect survives projection-out while confound directions alone do
  not reproduce it.
- **H4 (base vs instruct)**: the direction exists in the base model (similar AUROC,
  aligned direction) and the instruct model's own writing sits further to the AI side.

Alternative explanations tracked: detector drop caused by incoherence (perplexity
detectors reward it); direction = "Reddit vs encyclopedic"; direction = register of
chat-tuned generators (then base-model text should look human along it).

## Proposed Methodology

### Approach
Model: `meta-llama/Llama-3.1-8B-Instruct` (steered) and `meta-llama/Llama-3.1-8B`
(base), bf16 on one RTX A6000. Directions are unit vectors in the residual stream
after a block; activations are mean-pooled over text tokens (first 256 tokens, so
length is equalised).

### Experimental Steps
1. **Data** (`src/prep_data.py`): HAP-E matched continuations (train/test split by
   document, balanced over 6 genres, human chunk-2 vs 6 generators); HC3 pairs; RAID
   no-attack (human + 11 generators, 8 domains); MAGE test sample; Pavlick formality.
   Generation prompts: HAP-E continuation prompts, HC3 questions, RAID prompts, each
   with a matched human reference; dev set for tuning, disjoint test set.
2. **Activations** (`src/extract.py`): mean-pooled residuals per layer for all texts,
   in both models; plus per-text mean token NLL (fluency).
3. **E1** (`src/analyze_readout.py`): diff-of-means and logistic-regression directions
   per layer on HAP-E train; AUROC on held-out HAP-E, leave-one-genre-out,
   per-generator, HC3, RAID (chat vs non-chat generators), MAGE.
4. **Confound directions** (`src/directions.py`): formality (Pavlick top vs bottom
   tercile), fluency (low vs high NLL human text), length (long vs short window),
   genre subspace, chat register (Llama-3-8B-Instruct text minus Llama-3-8B base
   text), assistant-axis proxy (default assistant vs role-play responses of the model
   itself, reduced role set).
5. **E2** (`src/steer.py`, `src/score.py`): pilot on dev prompts to pick the layer and
   the coefficient range; then on test prompts generate with: no steering; AI
   direction (diff-of-means; LR) at several negative and positive coefficients;
   3 random directions at equal norm; each confound direction at equal norm;
   AI direction with confounds projected out; directional ablation at all layers;
   prompt baselines ("write like a human"; "write like an AI").
6. **Scoring**: detectors independent of Llama activations — Binoculars-style score
   (Qwen2.5-7B / -Instruct pair), desklib DeBERTa detector, RADAR, and an LLM judge
   (OpenRouter). Detectors are first validated on unsteered output vs matched human
   references; only detectors with AUROC > 0.8 there count as primary. Coherence:
   LLM-judge coherence, repetition, external-LM perplexity. Content: embedding
   similarity to the unsteered output for the same prompt and LLM-judge on-task score.
7. **E3**: cosines, variance of the AI direction inside the confound span, readout
   AUROC after projection-out, steering effect of confounds and of the residualised
   direction; formality classifier and length of steered outputs.
8. **E4**: same readout analysis in the base model; cosine of base and instruct
   directions; projection of each model's own continuation activations; steering the
   base model toward "AI" on raw continuation prompts.

### Baselines
Unsteered output; equal-norm random directions (3 seeds); confound directions at equal
norm; prompt instruction; human references (floor for detector scores).

### Evaluation Metrics
Detector score and fraction flagged at the threshold giving 5% FPR on human
references; coherence (judge 0-100, distinct-3, external NLL); content (embedding
cosine, judge on-task); readout AUROC; cosine between directions.

### Statistical Analysis Plan
Paired comparisons over prompts (same prompt, same sampling seed across conditions);
bootstrap 95% CIs over prompts (2,000 resamples); Wilcoxon signed-rank for steered vs
unsteered and for AI direction vs random at the same coefficient; Holm correction
across the family of direction comparisons. "Matched coherence": for each direction,
the largest coefficient whose mean judge coherence stays within 10 points of
unsteered (and a coherence-vs-detector frontier plot).

## Expected Outcomes
Support for H2: monotone dose-response on ≥ 2 independent detectors that exceeds the
random-direction band at matched coherence. Refutation: detector scores move only where
coherence collapses, or random directions do as well. Support for H3: effect survives
projection-out and confounds do not reproduce it; refutation: formality / chat register
direction has cosine near 1 or reproduces the effect.

## Timeline and Milestones
Setup + data 30 min; extraction 30 min; readout + directions 40 min; steering pilot
30 min; main generation 60–90 min; scoring 45 min; analysis 45 min; report 40 min.

## Potential Challenges
- Detectors weak on Llama-3.1 output → validate first, report all, mark primaries.
- Coherence collapse → frontier analysis instead of fixed coefficient.
- No assistant axis for 8B models → reduced-role proxy, stated as a limitation.

## Success Criteria
All four experiments run with the stated controls and the result — positive or
negative — is supported by paired statistics on ≥ 200 test prompts.

## Direction budget
Kept (from `resources.md`): D1 causal steering, D2 distinctness, D3 readout/generality
(base-vs-instruct folded into D2/D3). Pruned: SAE decomposition (different model
family), cross-family replication (compute), commercial detectors (no access),
paraphraser comparison (not about a linear direction), full 275-role assistant axis
(cost).
