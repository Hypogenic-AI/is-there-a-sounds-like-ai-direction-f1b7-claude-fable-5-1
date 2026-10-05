# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-04T18:56:47.374760Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/is-there-a-sounds-like-ai-direction-f1b7-claude-fable-5-1`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase**: resource_finder — complete. Artifacts: `papers/` (25 PDFs, `papers/README.md`, deep-read notes in `papers/notes/`), `datasets/` (6 datasets + `download_datasets.py`, `datasets/README.md`), `code/` (6 repos, `code/README.md`), `literature_review.md`, `resources.md`, `.resource_finder_complete`.

**Key findings**
- Readout is established, causality is not: 2608.24780 (LR probe, Llama-3-8B, layer ~16 best OOD) and 2606.07313 (mean-diff / LR / PCA) find a transferable human-vs-AI direction but never steer generation; 2503.03601 steers SAE features only to interpret them; 2606.06315 steers only random vectors. No paper found scores steered generations with an independent detector.
- Confound priors: SV-Detect's logit lens reads the direction as formal-vs-casual register; 2605.19516 reports mean human-probability 96.7% / 98.8% (GPTZero / Pangram) for base Llama-3-8B continuations vs 30.3% / 17.1% for instruct → detector "AI-ness" may be mostly post-training register. Assistant Axis's opposite pole is theatrical role-play, not human prose, so its relation to the H-vs-AI direction is open.
- Mean-diff directions tend to be more causal than LR probe directions (Marks & Tegmark), while LR reads out better (SV-Detect) → compare both.

**Decisions**
- Model: `meta-llama/Llama-3.1-8B-Instruct` (+ base `Llama-3.1-8B`); HF access verified, fits the 48 GB A6000. Not loaded yet.
- Direction source: HAP-E (matched continuations, 6 genres, has Llama-3 base + instruct); HC3 / RAID no-attack / MAGE for transfer and prompts.
- Independent detectors: Binoculars, Fast-DetectGPT, RADAR, one RoBERTa-type detector (exist on HF, not yet run). No GPTZero / Pangram access.
- Top-3 directions kept (scoring table and pruned D4–D8 in `resources.md` → "Direction budget"): D1 causal add/ablate test with detectors + coherence + content; D2 distinctness vs formality / length / genre / base-vs-instruct / assistant-axis proxy (cosine, projection-out, steer with confounds); D3 read↔write consistency (layer sweep, mean-diff vs LR, cross-dataset transfer, closed-loop projection).

**Next phase (experiment_runner) — concrete steps**
1. `source .venv/bin/activate`; install torch (CUDA), transformers, accelerate, scikit-learn, sentence-transformers (only CPU torch is installed now, used to inspect `.pt` files).
2. Extract per-layer activations on HAP-E human chunk-2 vs LLM continuations; build mean-diff and LR directions; held-out AUROC + transfer.
3. Steer generation (add α·v at a mid layer; ablate at all layers) on ≥ 200 fixed prompts; sweep α; score with detectors, coherence, content preservation.
4. Controls: norm-matched random direction, formality direction (Pavlick), base-minus-instruct direction, prompting baseline; projection-out reruns.

**Risks / open uncertainty**
- Perplexity-based detectors reward incoherence: any detector drop must be compared with the random-direction control at matched coherence.
- No assistant axis exists for 8B models; the proxy (reduced role set or base-vs-instruct) is weaker than the paper's axis.
- Deep reading was done by sub-agents (notes have page refs; a few appendix chunks of the two secondary papers were skipped). Cloned code was inspected, not executed; `mgt_probes` probe script is broken as shipped.
- MAGE labels are inverted (1 = human).
- This phase ran twice: the first session was cut off during a redundant RAID re-stream before writing the completion marker. The second session kept the complete first RAID file (`datasets/raid/train_none.csv`, 467,985 rows = 13,371 human docs × 35, all 8 domains), discarded the truncated copy, re-validated PDFs and dataset row counts, and spot-checked the sub-agent notes against the PDFs (2608.24780 layer 16, 2605.19516 detector numbers, 2503.03601 Gemma-2-2b, SV-Detect formal/casual reading — all consistent). No new searches were run.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase**: experiment_runner — complete. Deliverables on disk: `planning.md`, `REPORT.md`, `README.md`, `src/` (14 scripts), `results/` (tables, JSON, generations, scores, judge), `figures/` (5), `requirements.txt`.

**Key findings (layer 12, Llama-3.1-8B-Instruct / base; details and tables in `REPORT.md`)**
- Readout: direction separates chat-model text from human at AUROC ≥ 0.97 across HAP-E / HC3 / RAID-chat, but base-LLM text only 0.75–0.78, RAID non-chat 0.64, MAGE 0.60 (`results/readout_*.json`).
- Causal: steering toward human at matched coherence (c = −0.75) moves LLM-judge AI-likelihood 92.4 → 70.6 and desklib flag rate 99.5% → 85%; two random directions, formality, assistant-axis proxy and a prompt baseline do not. Binoculars (Qwen pair) moves the opposite way; RADAR does not respond (`results/table_main_test.csv`, `results/vs_random_main_test.csv`).
- Distinctness: cosine 0.96 with chat register (0.88 on disjoint generators); removing that span abolishes the steering effect. Formality 0.15, fluency 0.32, assistant-axis proxy −0.01 (`results/geometry.json`).
- Base vs instruct: direction exists in the base model (cosine 0.84); own-writing offset 0.45 SD (base) → 1.39 (instruct raw) → 3.9 (instruct chat). Adding it to the base model raises judge AI-likelihood 59 → 92.5 with coherence up (`results/table_base_cont.csv`).

**Decisions / deviations**: layer and coefficient grid chosen on dev prompts only; Binoculars with Qwen2.5-7B pair; judge `openai/gpt-5.6-terra` with `anthropic/claude-sonnet-5` fallback on content-filter hits; third random direction and formality-only residual direction dropped for time; second-tier directions at two coefficients.

**Unresolved**
- Binoculars reversal is unexplained (no test run). Unsteered base-model text at T = 0.7 is flagged by desklib / Binoculars, unlike Xu et al. at T = 1.0; not rerun.
- OpenRouter key hit its $150 daily limit near the end: judge half of the truncation control covers 72 / 208 prompts for the main condition. No further judge calls are possible until the limit resets.
- `results/acts/` (25 GB activations) is git-ignored; regenerate with `src/extract.py` and `src/fit_acts.py`.
- Nothing was committed to git in this phase.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
