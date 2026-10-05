# Resources Catalog

## Summary
Resources for testing whether the human-vs-AI readout direction in LLM activations is
(a) causal for generation and (b) distinct from formality / domain / length /
assistant persona. All four user-specified papers and all four user-specified
datasets were obtained. No staged local resources or code_references were listed in
the specification.

## Papers
Total papers downloaded: 25 (`papers/`). Deep-reading notes: `papers/notes/`.

| Title | Authors | Year | File | Key info |
|---|---|---|---|---|
| Linear Probing … Detection of MGT ★ | Quaremba et al. | 2026 | 2608.24780_linear_probing_mgt_detection.pdf | Shared linear MGT direction, Llama-3-8B, readout only |
| SV-Detect ★ | Vishnyakov, Gaintseva | 2026 | 2606.07313_sv_detect_steering_vectors.pdf | Per-layer H-vs-AI directions as detector features; formal-vs-casual logit lens |
| SAE features for ATD ★ | Kuznetsov et al. | 2025 | 2503.03601_sae_artificial_text_detection.pdf | Gemma-2-2b SAE features; steering for interpretation only |
| The Assistant Axis ★ | Lu et al. | 2026 | 2601.10387_assistant_axis.pdf | Persona axis; vectors released for 3 large models |
| Base Models Look Human To AI Detectors | Xu et al. | 2026 | 2605.19516_…pdf | Base 97–99% "human" vs instruct 17–30% (Llama-3-8B) |
| Steer-to-Detect | Liang, Li | 2026 | 2605.12890_steer_to_detect.pdf | Steers the observer LLM to improve detection |
| LLM Self-Recognition | Ardoin et al. | 2026 | 2606.06315_…pdf | Own-vs-human LDA probe; random-vector watermark steering |
| Refusal … Single Direction | Arditi et al. | 2024 | 2406.11717_…pdf | Diff-of-means, addition + ablation template |
| ActAdd / CAA / Persona Vectors / Style Vectors | Turner; Panickssery; Chen; Konen | 2023–25 | 2308.10248, 2312.06681, 2507.21509, 2402.01618 | Steering methods |
| Geometry of Truth / Linear Rep. Hypothesis | Marks & Tegmark; Park et al. | 2023 | 2310.06824, 2311.03658 | Mean-diff more causal than LR probes |
| Steering reliability / AxBench | Tan et al.; Wu et al. | 2024–25 | 2407.12404, 2501.17148 | Steering evaluation practice |
| RAID / HC3 / MAGE / HAP-E papers | Dugan; Guo; Li; Reinhart | 2023–24 | 2405.07940, 2301.07597, 2305.13242, 2410.16107 | Datasets |
| Binoculars / Fast-DetectGPT / RADAR | Hans; Bao; Hu | 2023–24 | 2401.12070, 2310.05130, 2307.03838 | Independent detectors |
| DIPPER / Adversarial Paraphrasing | Krishna; Cheng | 2023–25 | 2303.13408, 2506.07001 | Text-space evasion baselines |

★ = user-specified. See `papers/README.md` for the full list.

## Datasets
Total datasets downloaded: 6 (+ Biber companion). Re-download: `python datasets/download_datasets.py`.

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| HC3 ★ | HF `Hello-SimpleAI/HC3` | 24,322 questions | paired human / ChatGPT answers | `datasets/hc3/` | Confounded by register and length |
| HAP-E ★ | HF `browndw/human-ai-parallel-corpus` | 8,290 docs × 8 sources | matched continuations, 6 genres | `datasets/human_ai_parallel_corpus/` | Primary; has Llama-3 base + instruct |
| HAP-E Biber | HF `browndw/human-ai-parallel-corpus-biber` | 67 features per doc | style covariates | `datasets/human_ai_parallel_corpus_biber/` | Interpretable style axes |
| MAGE ★ | HF `yaful/MAGE` | 319k / 57k / 57k | detection, 10 domains | `datasets/mage/` | label 1 = human |
| RAID ★ (no-attack train) | HF `liamdugan/raid` | 467,985 rows (13,371 human + 454,614 generated), 802 MB | detection, 8 domains, 11 generators | `datasets/raid/train_none.csv` | Filtered stream of 11.8 GB train.csv |
| Pavlick formality | HF `osyvokon/pavlick-formality-scores` | 11,274 sentences | formality regression | `datasets/pavlick_formality/` | Formality control direction |
| Assistant Axis vectors | HF `lu-christina/assistant-axis-vectors` | 3 models, axis files | persona control | `datasets/assistant_axis_vectors/` | Large models only |

See `datasets/README.md` for schemas, loading code and caveats.

## Code Repositories
Total repositories cloned: 6.

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| mgt_probes | github.com/gerritq/mgt_probes | Linear-probe MGT detection + baselines | `code/mgt_probes/` | Probe script broken as shipped; no steering |
| sv-detect | github.com/Atmyre/sv-detect | Mean-diff / LR / PCA directions, logit lens | `code/sv-detect/` | Readout only |
| assistant-axis | github.com/safety-research/assistant-axis | Axis pipeline, steering / ablation / capping hooks | `code/assistant-axis/` | Needs LLM judge to recompute |
| refusal_direction | github.com/andyrdt/refusal_direction | Diff-of-means, addition, directional ablation | `code/refusal_direction/` | Template for the causal test |
| persona_vectors | github.com/safety-research/persona_vectors | Compact steering hook, persona vector extraction | `code/persona_vectors/` | `activation_steer.py` |
| raid | github.com/liamdugan/raid | Detector wrappers, TPR@FPR evaluation | `code/raid/` | `.git`, leaderboard, web removed |

See `code/README.md` for details.

## Resource Gathering Notes

### Search strategy
Paper-finder (7 queries; 2 failed with timeout / HTTP 500 and were re-queried with
different wording), arXiv metadata for the specified IDs, and two web searches for
work that steers generation along a human-vs-AI direction. Paper-finder returned
many persona- and refusal-steering papers (≥ 70 at relevance 3); only the
methodological anchors were downloaded rather than all of them, since the rest are
variations on refusal/persona steering with no bearing on AI-text detection.

### Selection criteria
User-specified items first; then papers that (i) read out or steer a human-vs-AI
direction, (ii) define the steering method to reuse, (iii) define datasets and
independent detectors, (iv) bear on the confounds named in the hypothesis.

### Challenges encountered
- arXiv API returned 503 for two IDs; metadata was taken from the abs pages instead.
- RAID is 16.7 GB; only no-attack train rows were kept. The RAID repo clone was 2.6 GB and was trimmed.
- Assistant Axis vectors exist only for 27B–70B models, which do not fit the A6000 unquantised.
- Commercial detectors (GPTZero, Pangram) need paid keys that are not available; Pangram's open EditLens model is gated and inaccessible with the current token.
- `uv add` could not build the placeholder project, so packages were added with `uv add --no-sync` (recorded in `pyproject.toml`) and installed with `uv pip install`.

### Gaps and workarounds
- No assistant axis for an 8B model → recompute with a reduced role set via `code/assistant-axis/pipeline`, and use a base-minus-instruct mean-difference direction as a second persona proxy.
- No GYAFC (licence-restricted) → Pavlick formality scores plus `s-nlp/roberta-base-formality-ranker`.
- Deep reading of the six key papers was delegated to sub-agents; their notes carry page references and flag uncertain items. Skipped chunks: appendix A.4 / qualitative examples of 2605.19516 and Appendix G of 2606.06315.
- Nothing in `code/` was executed, and no detector or LLM was loaded in this phase; model availability was checked via HF metadata only.

## Recommendations for Experiment Design

1. **Primary dataset**: HAP-E for extracting the direction (matched topic/genre/length); HC3, RAID and MAGE for readout transfer and as prompt sources.
2. **Baseline methods**: norm-matched random direction; formality direction; base-minus-instruct direction; assistant-axis proxy; prompting ("write like a human"); unsteered output.
3. **Evaluation metrics**: independent-detector score and TPR at 5% FPR vs steering coefficient; coherence (judge + external-LM perplexity + repetition); content preservation (embedding similarity, judge); cosine and projection-out analyses for distinctness.
4. **Code to adapt**: `refusal_direction` hooks (addition + ablation), `persona_vectors/activation_steer.py`, `sv-detect` direction constructions and logit lens, `raid` and `mgt_probes/src/baseline` detector wrappers.

### Direction budget (top 3 kept)
Scores 1–5 for literature evidence / relevance to hypothesis / expected information gain / feasibility on one A6000.

| # | Direction | Ev. | Rel. | Gain | Feas. | Total | Decision |
|---|---|---|---|---|---|---|---|
| D1 | **Causal steering test**: diff-of-means (and LR) H-vs-AI direction in Llama-3.1-8B-Instruct; add / ablate during generation; independent detectors + coherence + content, with random-direction control | 4 | 5 | 5 | 5 | 19 | **keep** |
| D2 | **Distinctness**: cosine and projection-out against formality, length, genre, base-vs-instruct, assistant-axis proxy; steer with the confound directions and with the residualised AI direction | 4 | 5 | 5 | 4 | 18 | **keep** |
| D3 | **Readout ↔ write consistency and generality**: layer sweep, mean-diff vs LR, direction transfer across HAP-E / HC3 / RAID / MAGE, closed-loop projection of steered outputs | 5 | 4 | 3 | 5 | 17 | **keep** |
| D4 | SAE-feature decomposition of the direction (Gemma Scope) | 4 | 3 | 3 | 2 | 12 | pruned — different model family and tooling; feature-level reading already in 2503.03601 |
| D5 | Cross-family universality (Gemma, Qwen, larger Llama) | 3 | 3 | 3 | 2 | 11 | pruned — compute; at most one replication if time remains |
| D6 | Commercial detectors (GPTZero, Pangram) | 4 | 4 | 3 | 1 | 12 | pruned — no API access |
| D7 | HIP-style paraphraser / fine-tuning comparison | 3 | 2 | 2 | 2 | 9 | pruned — not about a linear direction |
| D8 | Full 275-role Assistant-Axis recomputation | 4 | 3 | 2 | 2 | 11 | pruned — ≈ 331k generations; a reduced proxy is folded into D2 |

## Experiment-phase usage (added by experiment_runner)

- **Used**: HAP-E (direction fitting, held-out readout, continuation prompts), HC3, RAID no-attack and MAGE (transfer; HC3 and RAID also as generation prompts), Pavlick formality scores, role instructions and questions from `code/assistant-axis/data` (assistant-axis proxy), HAP-E Biber features (interpretation).
- **Not used**: the released Assistant Axis vectors (large models only); cloned repositories other than `assistant-axis/data` were consulted for method, not executed.
- **Models added**: Qwen2.5-7B / -Instruct (Binoculars-style score), desklib, RADAR, HC3 RoBERTa detector, formality ranker, all-mpnet-base-v2; judge `openai/gpt-5.6-terra` via OpenRouter.
- Results and process are in `REPORT.md`; the plan is in `planning.md`.
