# Downloaded Papers

25 PDFs. Deep-reading notes for the user-specified papers (and two closely related
finds) are in `papers/notes/`. `papers/pages/` holds 3-page PDF chunks of the
deep-read papers.

## User-specified (deep-read, see `notes/`)

1. [Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text](2608.24780_linear_probing_mgt_detection.pdf)
   - Authors: Gerrit Quaremba et al. · 2026 · arXiv 2608.24780 · code: `code/mgt_probes`
   - Why relevant: establishes the readout half of the hypothesis — a shared, transferable linear "MGT direction" in Llama-3-8B residual activations. No steering/causal test (listed as future work).
2. [SV-Detect: AI-generated Text Detection with Steering Vectors](2606.07313_sv_detect_steering_vectors.pdf)
   - Authors: Mikhail Vishnyakov, Tatiana Gaintseva · 2026 · arXiv 2606.07313 · code: `code/sv-detect`
   - Why relevant: per-layer human-vs-AI directions (mean-diff / LR / PCA) used as detector features; logit-lens shows a formal-vs-casual register reading. Readout only — never steers generation.
3. [Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders](2503.03601_sae_artificial_text_detection.pdf)
   - Authors: Kuznetsov, Kushnareva, et al. · 2025 · arXiv 2503.03601
   - Why relevant: Gemma-2-2b SAE features that detect AI text; steers with them but only to *interpret* features (GPT-4o descriptions), never scores steered text with a detector.
4. [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](2601.10387_assistant_axis.pdf)
   - Authors: Lu, Gallagher, Michala, Fish, Lindsey · 2026 · arXiv 2601.10387 · code: `code/assistant-axis`
   - Why relevant: the "assistant persona" direction that the AI-ness direction must be distinguished from; pre-computed axes released.

## Closely related finds (read main body)

5. [Base Models Look Human To AI Detectors](2605.19516_base_models_look_human_to_detectors.pdf) — Xu, Zhong, Raghunathan, Fang, Kolter · 2026 · arXiv 2605.19516. Detectors (GPTZero, Pangram) flag instruct-model text but not base-model text ⇒ "AI-ness" as seen by detectors may largely be instruction-tuning/assistant register. Key confound.
6. [Steer-to-Detect](2605.12890_steer_to_detect.pdf) — Liang, Li · 2026 · arXiv 2605.12890. Learns a steering vector injected into an observer LLM to improve class separability for detection (steering the *reader*, not the generator).
7. [LLM Self-Recognition: Steering and Retrieving Activation Signatures](2606.06315_llm_self_recognition_steering.pdf) — Ardoin, Schäfer, Wunder · 2026 · arXiv 2606.06315. Steering during generation leaves a signature recoverable from activations — a steering-as-watermark precedent, shows that steered generations are detectable in activation space.

## Steering / linear-representation methodology

8. [Refusal in Language Models Is Mediated by a Single Direction](2406.11717_refusal_single_direction.pdf) — Arditi et al. 2024. Template for our causal test: diff-of-means direction, activation addition and directional ablation, direction selection by causal effect. Code: `code/refusal_direction`.
9. [Steering Language Models with Activation Engineering (ActAdd)](2308.10248_activation_addition.pdf) — Turner et al. 2023.
10. [Steering Llama 2 via Contrastive Activation Addition](2312.06681_contrastive_activation_addition.pdf) — Panickssery (Rimsky) et al. 2023.
11. [Persona Vectors: Monitoring and Controlling Character Traits](2507.21509_persona_vectors.pdf) — Chen et al. 2025. Code: `code/persona_vectors`.
12. [Style Vectors for Steering Generative LLMs](2402.01618_style_vectors_steering.pdf) — Konen et al. 2024. Style (sentiment/emotion/formality-like) steering via activation vectors.
13. [The Geometry of Truth](2310.06824_geometry_of_truth.pdf) — Marks & Tegmark 2023. Diff-of-means directions are more causal than LR probe directions.
14. [The Linear Representation Hypothesis and the Geometry of LLMs](2311.03658_linear_representation_hypothesis.pdf) — Park et al. 2023.
15. [Analysing the Generalisation and Reliability of Steering Vectors](2407.12404_steering_vectors_reliability.pdf) — Tan et al. 2024. Steerability is highly variable; spurious biases.
16. [AxBench](2501.17148_axbench.pdf) — Wu et al. 2025. Evaluate steering by concept score × fluency × instruction-following; diff-of-means is a strong baseline.

## Detection, datasets, evasion

17. [RAID benchmark](2405.07940_raid_benchmark.pdf) — Dugan et al. 2024 (dataset + detector evaluation at fixed FPR).
18. [HC3: How Close is ChatGPT to Human Experts?](2301.07597_hc3.pdf) — Guo et al. 2023.
19. [MAGE: Machine-generated Text Detection in the Wild](2305.13242_mage.pdf) — Li et al. 2023.
20. [Do LLMs write like humans? (HAP-E corpus)](2410.16107_reinhart_llms_write_like_humans.pdf) — Reinhart et al. 2024. Biber-feature analysis; instruct models differ from humans far more than base models.
21. [Binoculars](2401.12070_binoculars.pdf) — Hans et al. 2024 (zero-shot detector).
22. [Fast-DetectGPT](2310.05130_fast_detectgpt.pdf) — Bao et al. 2023 (zero-shot detector).
23. [RADAR](2307.03838_radar_detector.pdf) — Hu et al. 2023 (adversarially trained RoBERTa detector).
24. [Paraphrasing evades detectors (DIPPER)](2303.13408_dipper_paraphrase_evades_detectors.pdf) — Krishna et al. 2023.
25. [Adversarial Paraphrasing](2506.07001_adversarial_paraphrasing.pdf) — Cheng et al. 2025 (detector-guided humanization; text-space baseline for evasion).
