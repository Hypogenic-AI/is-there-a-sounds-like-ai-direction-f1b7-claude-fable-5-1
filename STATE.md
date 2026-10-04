# Research State

- Current phase: `None`
- Pipeline completed: `False`

## Previous phases

resource_finder (failed)

## Current phase context

- Phase: `resource_finder`
- Status: `failed`
- Started: `2026-10-04T18:25:28.691777Z`
- Next steps:
  - Review the literature and resource catalog before running experiments.

## Workspace check

- Root: `/workspaces/is-there-a-sounds-like-ai-direction-f1b7-claude-fable-5-1`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `literature_review.md`, `resources.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase**: resource_finder — complete. Artifacts: `papers/` (25 PDFs, `papers/README.md`, deep-read notes in `papers/notes/`), `datasets/` (6 datasets + `download_datasets.py`, `datasets/README.md`), `code/` (6 repos, `code/README.md`), `literature_review.md`, `resources.md`, `.resource_finder_complete`.

**Key findings**
- Readout is established, causality is not: 2608.24780 (LR probe, Llama-3-8B, layer ~16 best OOD) and 2606.07313 (mean-diff / LR / PCA) find a transferable human-vs-AI direction but never steer generation; 2503.03601 steers SAE features only to interpret them; 2606.06315 steers only random vectors. No paper found scores steered generations with an independent detector.
- Confound priors: SV-Detect's logit lens reads the direction as formal-vs-casual register; 2605.19516 reports base Llama-3-8B judged 97–99% human vs 17–30% for instruct (GPTZero / Pangram) → detector "AI-ness" may be mostly post-training register. Assistant Axis's opposite pole is theatrical role-play, not human prose, so its relation to the H-vs-AI direction is open.
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
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

<!-- NEURICO_AGENT_NOTES_END -->
