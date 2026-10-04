# Cloned Repositories

All shallow clones (`--depth 1`). None were executed in this phase (code inspection
only); the experiment runner should treat them as references to adapt, not as
installed packages.

## 1. mgt_probes — official code for "Linear Probing Provides Robust and Efficient Detection of MGT" (2608.24780)
- URL: https://github.com/gerritq/mgt_probes
- Location: `code/mgt_probes/`
- Key files: `src/inference.py` (hidden-state extraction: Llama-3-8B-Instruct, raw text, last token, all 33 layer states), `src/probes/probe_main.py` (StandardScaler → PCA-100 → L2 logistic regression per layer; LLP / CLP variants), `src/baseline/` (Binoculars, Fast-DetectGPT, RepreGuard, entropy/rank/LLR … usable as independent detectors).
- Notes: no data and no saved probe vectors ship with the repo. `probe_main.py` imports `OOD` from `src.utils`, which does not define it, so the script fails as shipped — reimplement (≈30 lines of sklearn). The probe direction lives in standardised PCA coordinates; back-project (`pca.components_.T @ w / scaler.scale_`) to get a residual-space vector for steering. No steering code.

## 2. sv-detect — official code for SV-Detect (2606.07313)
- URL: https://github.com/Atmyre/sv-detect
- Location: `code/sv-detect/`
- Key files: `src/extract/extract_activations.py` (mean-pooled per-layer activations), `src/extract/compute_steering_vectors.py` (mean-diff / logistic-regression / PCA-of-paired-differences directions), `src/extract/nb_pipeline.py`, `src/interpret/` (logit-lens token readout of directions), `src/baselines/`.
- Notes: readout only; reuse the three direction constructions and the logit-lens interpretation. Default backbone GPT-Neo-2.7B.

## 3. assistant-axis — official code for The Assistant Axis (2601.10387)
- URL: https://github.com/safety-research/assistant-axis
- Location: `code/assistant-axis/`
- Key files: `assistant_axis/` (`load_model`, `load_axis`, steering + activation-capping hooks), `pipeline/` (5-step pipeline to compute the axis for a new model: generate role responses → extract activations → judge → role vectors → axis), `data/` (275 roles, extraction questions), `notebooks/`.
- Notes: axis = mean(default-assistant activations) − mean(role activations), one vector per layer. Pre-computed axes exist only for Gemma-2-27B, Qwen-3-32B, Llama-3.3-70B (downloaded to `datasets/assistant_axis_vectors/`). For a smaller model the pipeline must be rerun (needs an LLM judge — OpenAI/OpenRouter key is available in env) or approximated with a reduced role set. See `papers/notes/2601.10387_notes.md`.

## 4. refusal_direction — Arditi et al. 2024 (2406.11717)
- URL: https://github.com/andyrdt/refusal_direction
- Location: `code/refusal_direction/`
- Key files: `pipeline/submodules/generate_directions.py` (diff-of-means at every layer/position), `select_direction.py` (pick direction by causal effect + KL sanity filter), `pipeline/utils/hook_utils.py` (activation addition and directional-ablation hooks on every block), `pipeline/model_utils/` (Gemma, Llama, Qwen wrappers).
- Notes: the cleanest template for the causal test — add `α·r̂` at one layer, or project out `r̂` at all layers, during generation.

## 5. persona_vectors — Chen et al. 2025 (2507.21509)
- URL: https://github.com/safety-research/persona_vectors
- Location: `code/persona_vectors/`
- Key files: `generate_vec.py` (mean-diff persona vector from contrastive system prompts), `activation_steer.py` (compact steering hook class with `response` / `prompt` / `all` position modes), `judge.py`.
- Notes: `activation_steer.py` is a drop-in steering context manager for HF models.

## 6. raid — RAID benchmark tooling (2405.07940)
- URL: https://github.com/liamdugan/raid (pip: `raid-bench`)
- Location: `code/raid/` — `.git`, `leaderboard/` and `web/` removed after cloning (2.6 GB of leaderboard submissions).
- Key files: `detectors/models/` (wrappers for Binoculars, Fast-DetectGPT, RADAR, GLTR, GPT-2/ChatGPT RoBERTa detectors, …), `raid/evaluate.py` (threshold at fixed FPR on human text → TPR).
- Notes: use the detector wrappers for the *independent* detectors and adopt the TPR@FPR=5%/1% protocol.
